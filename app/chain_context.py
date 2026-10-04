"""Reusable certificate-chain verification context shared by every deployment
entry point that reconstructs chains from the unordered PEM bag:

``/scenarios/normalize``, ``/deployments/precheck``,
``/deployments/policy-review`` and ``/deployments/revocation-review``
(and, for its primitives, ``/deployments/ocsp-precheck`` and
``/deployments/variants``).

A :class:`VerificationContext` is a **per-request** object that performs the
whole chain pipeline exactly once:

1. parses every PEM entry (each element may itself carry several concatenated
   PEM blocks) and de-duplicates certificates by their DER SHA-256
   fingerprint;
2. establishes the issuance relation by *cryptographically verifying* each
   child's signature with each candidate issuer's public key — never by
   comparing subject/issuer DNs;
3. decides, independently of any diagnostic limit, whether each referenced
   leaf has zero, exactly one or several simple leaf-first paths over those
   verified edges that terminate at a declared trust root.

The zero / unique / ambiguous verdict is computed without path enumeration
(one reachability search plus a polynomial uniqueness check); the bounded
(at most :data:`MAX_CANDIDATE_PATHS` = 64) candidate-path sample used for 422
diagnostics is produced on demand and can never influence the verdict.

Nothing in this module keeps module-level mutable state: a fresh context is
constructed per request, so trust roots, time validity, name constraints and
revocation rules of one request can never leak into another.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, ed448, padding, rsa

from .models import (
    NormalizeChainIssue,
    NormalizeFingerprintIssue,
    NormalizePemIssue,
)

# A single request element may itself carry several concatenated PEM blocks.
_PEM_BLOCK_RE = re.compile(
    rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----",
    re.DOTALL,
)

_RSASSA_PSS_OID = "1.2.840.113549.1.1.10"

# Enumerating *all* simple paths is factorial in the worst case. The zero /
# unique / ambiguous verdict below is therefore computed without enumeration
# (one path search plus a polynomial uniqueness check); this cap only bounds
# the candidate paths shown in 422 diagnostics, never the decision itself.
MAX_CANDIDATE_PATHS = 64

FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class NormalizeError(ValueError):
    """Chain/request failure shared by every PEM-bag entry point.

    Mapped to HTTP 422 with structured detail by ``app.main``; the extra
    issue lists (identity/eku) are carried opaquely for the routes that need
    them and omitted from the others' detail payloads.
    """

    def __init__(
        self,
        error: str,
        message: str,
        *,
        pem_issues: Optional[list[NormalizePemIssue]] = None,
        fingerprint_issues: Optional[list[NormalizeFingerprintIssue]] = None,
        chain_issues: Optional[list[NormalizeChainIssue]] = None,
        identity_issues: Optional[list] = None,
        eku_issues: Optional[list] = None,
    ) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.pem_issues = pem_issues or []
        self.fingerprint_issues = fingerprint_issues or []
        self.chain_issues = chain_issues or []
        self.identity_issues = identity_issues or []
        self.eku_issues = eku_issues or []


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafRef:
    """One endpoint's reference to a leaf, carrying its own diagnostic path.

    ``role`` is ``"current"`` / ``"target"`` (the only roles the chain
    diagnostics model admits); the revocation and pre-check routes only ever
    reference target leaves.
    """

    endpoint_id: str
    role: str
    raw_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return self.raw_fingerprint.strip().lower()

    @property
    def referred_by(self) -> str:
        return (f"endpoints['{self.endpoint_id}']."
                f"{self.role}_leaf_fingerprint")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def cert_fingerprint(cert: x509.Certificate) -> str:
    """Lower-case SHA-256 fingerprint of the DER encoding."""
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


def normalize_fingerprint(raw: str) -> Optional[str]:
    """Trim/lower-case a fingerprint; return None unless it is 64 hex chars."""
    fp = raw.strip().lower()
    return fp if FINGERPRINT_RE.match(fp) else None


def validity_utc(cert: x509.Certificate, attr: str) -> datetime:
    """Timezone-aware validity bound (cryptography >= 42 ``*_utc`` aware)."""
    # cryptography >= 42 exposes timezone-aware *_utc properties; older
    # versions expose naive datetimes that are already expressed in UTC.
    value = getattr(cert, f"{attr}_utc", None)
    if value is not None:
        return value
    return getattr(cert, attr).replace(tzinfo=timezone.utc)


def dn_str(name: x509.Name) -> str:
    return name.rfc4514_string()


def load_pem_bundle(
    pem_certificates: list[str],
) -> tuple[dict[str, x509.Certificate], list[NormalizePemIssue]]:
    """Parse every PEM entry and dedupe by DER fingerprint.

    Returns the unique certificates (keyed by lower-case SHA-256, in
    first-seen order) and one issue per unparseable input element.
    """
    certs: dict[str, x509.Certificate] = {}
    issues: list[NormalizePemIssue] = []
    for index, text in enumerate(pem_certificates):
        raw = text.encode("utf-8", errors="replace")
        blocks = _PEM_BLOCK_RE.findall(raw)
        if not blocks:
            issues.append(NormalizePemIssue(
                index=index,
                message="no PEM-encoded CERTIFICATE block found",
            ))
            continue
        for block in blocks:
            try:
                cert = x509.load_pem_x509_certificate(block)
            except ValueError as exc:
                issues.append(NormalizePemIssue(
                    index=index,
                    message=f"malformed PEM certificate: {exc}",
                ))
                continue
            fp = cert_fingerprint(cert)
            if fp not in certs:  # duplicate DER copy: ignore, keep first
                certs[fp] = cert
    return certs, issues


# ---------------------------------------------------------------------------
# Signature verification (the actual issuance relation)
# ---------------------------------------------------------------------------


def verify_signed_data(
    public_key,
    signature: bytes,
    tbs: bytes,
    hash_algo,
    signature_algorithm_oid: str,
    signature_algorithm_parameters,
) -> bool:
    """Verify ``signature`` over ``tbs`` with ``public_key``.

    Shared core of certificate issuance checks, CRL checks and OCSP response
    verification. InvalidSignature is the ordinary "did not sign" answer;
    unsupported key types or algorithms simply yield False so the bundle
    stays processable.
    """
    try:
        if isinstance(public_key, (ed25519.Ed25519PublicKey,
                                   getattr(ed448, "Ed448PublicKey", ()))):
            # PureEdDSA: the signed data itself identifies the algorithm.
            public_key.verify(signature, tbs)
        elif isinstance(public_key, rsa.RSAPublicKey):
            if signature_algorithm_oid == _RSASSA_PSS_OID:
                parameters = signature_algorithm_parameters or padding.PSS(
                    mgf=padding.MGF1(hash_algo),
                    salt_length=hash_algo.digest_size,
                )
                public_key.verify(signature, tbs, parameters, hash_algo)
            else:
                public_key.verify(
                    signature, tbs, padding.PKCS1v15(), hash_algo
                )
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, tbs, ec.ECDSA(hash_algo))
        elif isinstance(public_key, dsa.DSAPublicKey):
            public_key.verify(signature, tbs, hash_algo)
        else:
            return False
    except (InvalidSignature, UnsupportedAlgorithm, TypeError, ValueError):
        return False
    return True


def verify_signature(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    """True iff ``child`` was signed by ``issuer``'s private key."""
    return verify_signed_data(
        issuer.public_key(),
        child.signature,
        child.tbs_certificate_bytes,
        child.signature_hash_algorithm,
        child.signature_algorithm_oid.dotted_string,
        getattr(child, "signature_algorithm_parameters", None),
    )


def build_issuer_edges(
    certs: dict[str, x509.Certificate],
) -> dict[str, list[str]]:
    """Map every certificate to the fingerprints that cryptographically signed it.

    Every ordered pair is checked by signature verification alone: a matching
    subject/issuer DN is not a filter (cross-certified certs often carry an
    issuer DN unrelated to the wrapping certificate) and a matching DN never
    establishes an edge by itself.
    """
    fps = sorted(certs)
    edges: dict[str, list[str]] = {fp: [] for fp in fps}
    for child_fp in fps:
        child = certs[child_fp]
        for issuer_fp in fps:
            if verify_signature(child, certs[issuer_fp]):
                edges[child_fp].append(issuer_fp)
    return edges


# ---------------------------------------------------------------------------
# Chain reconstruction: exact zero / unique / ambiguous verdict
# ---------------------------------------------------------------------------
#
# The verdict never enumerates all simple paths (factorial in the worst case)
# and never lets the diagnostic cap influence the outcome:
#
# * ``_find_trust_path`` returns one simple path to a trust root, or None —
#   an ordinary reachability search, O(V + E).
# * ``_has_second_trust_path`` decides whether a *different* simple path to a
#   trust root exists. Any such path shares a (possibly empty) prefix with the
#   first path ``first`` and deviates at exactly one position i, so a second
#   path exists iff some verified edge ``first[i] -> alt`` with
#   ``alt != first[i + 1]`` and ``alt`` not on the prefix can still reach a
#   trust root without re-entering that prefix — one reverse reachability
#   sweep per prefix position, O(len(first) * (V + E)) overall.
#
# Both hold for arbitrary graphs: cycles, shared intermediates and any number
# of cross-signing branches are handled without unbounded work or storage.


def _find_trust_path(
    leaf: str,
    edges: dict[str, list[str]],
    trust_roots: frozenset[str],
) -> Optional[list[str]]:
    """One simple leaf-first path from ``leaf`` to a trust root, or None.

    A path stops at the first trust root it reaches (trust roots terminate a
    chain even when they have outgoing verified edges, e.g. self-loops).
    """
    if leaf in trust_roots:
        return [leaf]
    visited = {leaf}
    path = [leaf]

    def dfs(node: str) -> bool:
        for nxt in edges.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            if nxt in trust_roots or dfs(nxt):
                return True
            path.pop()
        return False

    return list(path) if dfs(leaf) else None


def _has_second_trust_path(
    first: list[str],
    edges: dict[str, list[str]],
    trust_roots: frozenset[str],
) -> bool:
    """True iff a simple path to a trust root other than ``first`` exists."""
    reverse: dict[str, list[str]] = {}
    for child, issuers in edges.items():
        for issuer in issuers:
            reverse.setdefault(issuer, []).append(child)

    # ``blocked`` is the shared prefix first[0..i]: a deviating path must not
    # re-enter it. Everything else that can still reach a trust root is found
    # by one reverse sweep from the trust roots over verified edges.
    blocked: set[str] = set()
    for i in range(len(first) - 1):
        node = first[i]
        blocked.add(node)
        reachable = {root for root in trust_roots if root not in blocked}
        stack = list(reachable)
        while stack:
            cur = stack.pop()
            for child in reverse.get(cur, []):
                if child not in blocked and child not in reachable:
                    reachable.add(child)
                    stack.append(child)
        nxt = first[i + 1]
        for alt in edges.get(node, []):
            if alt != nxt and alt not in blocked and alt in reachable:
                return True
    return False


def _enumerate_paths(
    leaf: str,
    edges: dict[str, list[str]],
    trust_roots: frozenset[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Up to :data:`MAX_CANDIDATE_PATHS` simple leaf-first paths from ``leaf``.

    Diagnostic sample for 422 responses, deliberately bounded: the traversal
    stops once the cap is hit, so the result may omit paths. Returns
    (valid_paths, dead_end_paths): sampled paths that terminate at a declared
    trust root, and sampled maximal simple paths that run out of verified
    edges (or would re-enter a visited node) without reaching one. The
    zero/unique/ambiguous verdict never reads this — see above.
    """
    valid: list[list[str]] = []
    dead_ends: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str], visited: set[str]) -> None:
        if len(seen) >= MAX_CANDIDATE_PATHS:
            return
        if node in trust_roots:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                valid.append(list(path))
            return
        progressed = False
        for nxt in edges.get(node, []):
            if nxt in visited:
                continue
            progressed = True
            visited.add(nxt)
            path.append(nxt)
            walk(nxt, path, visited)
            path.pop()
            visited.remove(nxt)
        if not progressed:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                dead_ends.append(list(path))

    walk(leaf, [leaf], {leaf})
    return valid, dead_ends


def dedupe_paths(paths: Iterable[list[str]]) -> list[list[str]]:
    unique = {tuple(path) for path in paths}
    return [list(path) for path in sorted(unique)][:MAX_CANDIDATE_PATHS]


def chain_resolution_error(
    chain_issues: list[NormalizeChainIssue],
    *,
    ambiguous_message: str,
    no_path_message: str,
) -> NormalizeError:
    """Build the 422 for a no-path / ambiguous-path resolution failure.

    Error-code precedence mirrors every route's historical behavior: if any
    referenced leaf is ambiguous, the whole request is ``ambiguous_chain``;
    otherwise it is ``no_trust_path``.
    """
    ambiguous = any(i.reason == "ambiguous_chain" for i in chain_issues)
    return NormalizeError(
        "ambiguous_chain" if ambiguous else "no_trust_path",
        ambiguous_message if ambiguous else no_path_message,
        chain_issues=chain_issues,
    )


# ---------------------------------------------------------------------------
# The per-request verification context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Verdict:
    """Exact resolution result for one distinct leaf fingerprint."""

    path: Optional[list[str]]
    ambiguous: bool

    @property
    def unique(self) -> bool:
        return self.path is not None and not self.ambiguous


class VerificationContext:
    """Per-request PEM bag + trust roots + issuance graph + chain verdicts.

    Constructed once per request; all derived structures (parsed
    certificates, issuance edges, per-leaf verdicts, diagnostic samples) are
    cached on the instance and never shared between requests.

    The constructor only *records* referential/self-signature problems as
    structured issues; each entry point raises on them in its own historical
    order via the ``*_error`` helpers, preserving field order and error codes.
    """

    def __init__(
        self,
        pem_certificates: list[str],
        trust_root_fingerprints: Iterable[str],
        leaf_refs: Iterable[LeafRef] = (),
    ) -> None:
        # Trust roots/leaves are expected already syntax-validated by the
        # caller (which owns the invalid_fingerprint 422 shape); de-duplicate
        # while preserving inputs for diagnostics.
        self.trust_roots: frozenset[str] = frozenset(trust_root_fingerprints)
        self.leaf_refs: list[LeafRef] = list(leaf_refs)

        self.certs, self.pem_issues = load_pem_bundle(pem_certificates)
        self.edges: dict[str, list[str]] = build_issuer_edges(self.certs)

        # Lazily filled per distinct leaf fingerprint.
        self._verdicts: dict[str, _Verdict] = {}
        self._diagnostics: dict[
            str, tuple[list[list[str]], list[list[str]]]
        ] = {}

    # -- basic accessors -----------------------------------------------------

    @property
    def trust_root_set(self) -> frozenset[str]:
        return self.trust_roots

    # -- referential integrity / trust-anchor checks -------------------------

    def missing_fingerprint_issues(self) -> list[NormalizeFingerprintIssue]:
        """Issues for declared roots/leaves absent from the parsed bundle.

        Order is unchanged from every entry point: roots first (sorted), then
        leaves in request order.
        """
        issues: list[NormalizeFingerprintIssue] = []
        for fp in sorted(self.trust_roots):
            if fp not in self.certs:
                issues.append(NormalizeFingerprintIssue(
                    fingerprint=fp, referred_by="trust_roots"))
        for ref in self.leaf_refs:
            if ref.fingerprint not in self.certs:
                issues.append(NormalizeFingerprintIssue(
                    fingerprint=ref.fingerprint,
                    referred_by=ref.referred_by,
                ))
        return issues

    def non_self_signed_roots(self) -> list[str]:
        """Declared roots that are not self-signed (subject == issuer and a
        valid self-signature), sorted by fingerprint."""
        return sorted(
            fp for fp in self.trust_roots
            if self.certs[fp].subject != self.certs[fp].issuer
            or not verify_signature(self.certs[fp], self.certs[fp])
        )

    def bad_root_issues(self) -> list[NormalizeFingerprintIssue]:
        return [
            NormalizeFingerprintIssue(
                fingerprint=fp, referred_by="trust_roots")
            for fp in self.non_self_signed_roots()
        ]

    # -- exact verdict + bounded diagnostics per leaf ------------------------

    def verdict(self, leaf: str) -> _Verdict:
        """Exact zero/unique/ambiguous verdict for one distinct leaf.

        Cached per fingerprint within this request; diagnostic caps never
        touch it.
        """
        cached = self._verdicts.get(leaf)
        if cached is None:
            first = _find_trust_path(leaf, self.edges, self.trust_root_set)
            ambiguous = first is not None and _has_second_trust_path(
                first, self.edges, self.trust_root_set
            )
            cached = _Verdict(first, ambiguous)
            self._verdicts[leaf] = cached
        return cached

    def _candidate_paths(
        self, leaf: str
    ) -> tuple[list[list[str]], list[list[str]]]:
        cached = self._diagnostics.get(leaf)
        if cached is None:
            cached = _enumerate_paths(
                leaf, self.edges, self.trust_root_set
            )
            self._diagnostics[leaf] = cached
        return cached

    def chain_issue_for(self, ref: LeafRef) -> NormalizeChainIssue:
        """Build the bounded diagnostic chain issue for one failing leaf ref."""
        leaf = ref.fingerprint
        verdict = self.verdict(leaf)
        valid, dead_ends = self._candidate_paths(leaf)
        return NormalizeChainIssue(
            endpoint_id=ref.endpoint_id,
            leaf_role=ref.role,  # type: ignore[arg-type]
            leaf=leaf,
            candidate_issuers=sorted(self.edges.get(leaf, [])),
            candidate_paths=dedupe_paths(valid + dead_ends),
            reason="ambiguous_chain" if verdict.ambiguous else "no_trust_path",
        )

    def resolve_chains(
        self,
        refs: Iterable[LeafRef],
        *,
        dedupe_leaf_failures: bool = False,
    ) -> tuple[dict[str, list[str]], list[NormalizeChainIssue]]:
        """Resolve every referenced leaf to its unique verified chain.

        Returns ``(chosen, chain_issues)`` where ``chosen`` maps each
        resolvable leaf fingerprint to its unique leaf-first path and
        ``chain_issues`` carries one bounded diagnostic issue per *reference*
        that fails (the normalize/precheck/policy-review behavior: a leaf
        referenced as both current and target produces one issue per role).
        When ``dedupe_leaf_failures`` is set, one issue per distinct failing
        fingerprint is emitted instead (revocation-review references target
        leaves only and historically de-duplicated them). The exact verdict
        is evaluated once per distinct fingerprint and is independent of the
        64-path diagnostic sample.
        """
        chosen: dict[str, list[str]] = {}
        chain_issues: list[NormalizeChainIssue] = []
        failed: set[str] = set()
        for ref in refs:
            leaf = ref.fingerprint
            verdict = self.verdict(leaf)
            if verdict.unique:
                chosen[leaf] = verdict.path  # type: ignore[assignment]
                continue
            if dedupe_leaf_failures and leaf in failed:
                continue
            failed.add(leaf)
            chain_issues.append(self.chain_issue_for(ref))
        return chosen, chain_issues
