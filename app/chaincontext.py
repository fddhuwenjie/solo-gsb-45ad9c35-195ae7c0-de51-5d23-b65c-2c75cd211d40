"""Reusable certificate chain verification context.

One :class:`ChainVerificationContext` is built per request and shared by
``/scenarios/normalize``, ``/deployments/precheck``,
``/deployments/policy-review`` and ``/deployments/revocation-review`` (the
OCSP pre-check and variant selection reuse the same parsing/signature
primitives). The context parses the PEM bag exactly once, collapses
duplicate DER copies by fingerprint, establishes issuance edges by
*cryptographically verifying* each child's signature with the candidate
issuer's public key — never by comparing subject/issuer distinguished names
(two unrelated CAs may share a DN, and cross-certified CAs share a key under
several distinct certificates) — and answers, for any leaf, the exact
zero / unique / ambiguous verdict over leaf-first simple paths that
terminate at a declared trust root.

Two properties are deliberate:

* **Request-scoped state.** Everything the context computes derives from the
  constructor's PEM bag; nothing is cached at module scope, so no state can
  leak across requests. Trust roots, validity instants, name constraints and
  revocation rules stay with the calling endpoint.
* **Verdict/diagnostic separation.** The zero / unique / ambiguous verdict
  is computed without enumerating simple paths (one path search plus a
  polynomial uniqueness check) and never reads the bounded diagnostic
  sample. The sample — at most ``_MAX_CANDIDATE_PATHS`` (64) candidate paths
  shown in 422 responses — may be truncated in heavily cross-signed bundles;
  the verdict may not.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable, NamedTuple, Optional

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
_MAX_CANDIDATE_PATHS = 64

FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


class NormalizeError(ValueError):
    """Normalization failure (mapped to HTTP 422 with structured detail)."""

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


class LeafReference(NamedTuple):
    """One endpoint leaf to resolve, in request order."""

    endpoint_id: str
    leaf_role: str  # "current" | "target"
    fingerprint: str  # already normalized (lower-case hex)


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def normalize_fingerprint(raw: str) -> str:
    """Canonical form of a fingerprint request field (case-insensitive hex)."""
    return raw.strip().lower()


def fingerprint_format_issues(
    refs: Iterable[tuple[str, str]],
) -> list[NormalizeFingerprintIssue]:
    """One issue per (raw fingerprint, referred_by) pair that is not 64-hex."""
    return [
        NormalizeFingerprintIssue(fingerprint=raw, referred_by=referred_by)
        for raw, referred_by in refs
        if not FINGERPRINT_RE.match(normalize_fingerprint(raw))
    ]


def require_valid_fingerprints(refs: Iterable[tuple[str, str]]) -> None:
    """Raise ``invalid_fingerprint`` listing every malformed (raw, ref) pair."""
    issues = fingerprint_format_issues(refs)
    if issues:
        raise NormalizeError(
            "invalid_fingerprint",
            "one or more fingerprints are not 64-character hex SHA-256 values",
            fingerprint_issues=issues,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def cert_fingerprint(cert: x509.Certificate) -> str:
    """Lower-case SHA-256 fingerprint of the DER encoding."""
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


def _validity_utc(cert: x509.Certificate, attr: str) -> datetime:
    # cryptography >= 42 exposes timezone-aware *_utc properties; older
    # versions expose naive datetimes that are already expressed in UTC.
    value = getattr(cert, f"{attr}_utc", None)
    if value is not None:
        return value
    return getattr(cert, attr).replace(tzinfo=timezone.utc)


def _dn(name: x509.Name) -> str:
    return name.rfc4514_string()


def _load_pem_bundle(
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


def _verify_signed_data(
    public_key,
    signature: bytes,
    tbs: bytes,
    hash_algo,
    signature_algorithm_oid: str,
    signature_algorithm_parameters,
) -> bool:
    """Verify ``signature`` over ``tbs`` with ``public_key``.

    Shared core of certificate issuance checks and OCSP response
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


def _verify_signature(child: x509.Certificate, issuer: x509.Certificate) -> bool:
    """True iff ``child`` was signed by ``issuer``'s private key."""
    return _verify_signed_data(
        issuer.public_key(),
        child.signature,
        child.tbs_certificate_bytes,
        child.signature_hash_algorithm,
        child.signature_algorithm_oid.dotted_string,
        getattr(child, "signature_algorithm_parameters", None),
    )


def _build_issuer_edges(
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
            if _verify_signature(child, certs[issuer_fp]):
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


# ---------------------------------------------------------------------------
# Bounded candidate-path enumeration (422 diagnostics only)
# ---------------------------------------------------------------------------


def _enumerate_paths(
    leaf: str,
    edges: dict[str, list[str]],
    trust_roots: frozenset[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Up to ``_MAX_CANDIDATE_PATHS`` simple leaf-first paths from ``leaf``.

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
        if len(seen) >= _MAX_CANDIDATE_PATHS:
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


def _dedupe_paths(paths: Iterable[list[str]]) -> list[list[str]]:
    unique = {tuple(path) for path in paths}
    return [list(path) for path in sorted(unique)][:_MAX_CANDIDATE_PATHS]


# ---------------------------------------------------------------------------
# Request-scoped verification context
# ---------------------------------------------------------------------------


class ChainVerificationContext:
    """One request's parsed PEM bag, verified issuance graph and chain verdicts.

    Constructing the context parses and deduplicates the PEM bag (raising
    ``invalid_pem`` on unparseable elements); the signature-verified issuance
    graph is then built at most once, on first use. Every method raises
    :class:`NormalizeError` with the same structured detail the endpoints
    have always returned, so callers keep their existing fields, ordering
    and error codes.
    """

    def __init__(self, pem_certificates: list[str]) -> None:
        certs, pem_issues = _load_pem_bundle(pem_certificates)
        if pem_issues:
            raise NormalizeError(
                "invalid_pem",
                "one or more PEM list elements could not be parsed",
                pem_issues=pem_issues,
            )
        self.certs = certs
        self._edges: Optional[dict[str, list[str]]] = None

    @property
    def edges(self) -> dict[str, list[str]]:
        """Verified issuance graph (child -> signing issuers), built once."""
        if self._edges is None:
            self._edges = _build_issuer_edges(self.certs)
        return self._edges

    def require_known_fingerprints(
        self, refs: Iterable[tuple[str, str]]
    ) -> None:
        """Raise ``unknown_fingerprint`` for every (fp, referred_by) pair
        whose fingerprint matches no certificate in the bag."""
        missing = [
            NormalizeFingerprintIssue(fingerprint=fp, referred_by=referred_by)
            for fp, referred_by in refs
            if fp not in self.certs
        ]
        if missing:
            raise NormalizeError(
                "unknown_fingerprint",
                "some fingerprints do not match any certificate in the PEM "
                "bundle",
                fingerprint_issues=missing,
            )

    def require_self_signed_roots(self, trust_roots: Iterable[str]) -> None:
        """Raise ``trust_root_not_self_signed`` unless every declared trust
        root is a self-signed certificate (subject == issuer with a valid
        self-signature). Roots must already be known to the bag."""
        bad_roots = sorted(
            fp for fp in trust_roots
            if self.certs[fp].subject != self.certs[fp].issuer
            or not _verify_signature(self.certs[fp], self.certs[fp])
        )
        if bad_roots:
            raise NormalizeError(
                "trust_root_not_self_signed",
                "every declared trust root must be a self-signed certificate "
                "(subject == issuer with a valid self-signature)",
                fingerprint_issues=[
                    NormalizeFingerprintIssue(
                        fingerprint=fp, referred_by="trust_roots"
                    )
                    for fp in bad_roots
                ],
            )

    def resolve_unique_chains(
        self,
        leaves: Iterable[LeafReference],
        trust_roots: Iterable[str],
        *,
        no_path_message: str,
    ) -> dict[str, list[str]]:
        """Resolve every referenced leaf to its unique verified chain.

        Returns ``{leaf_fingerprint: leaf-first chain}`` for the leaves that
        have exactly one verified path to a declared trust root. When any
        referenced leaf has zero or several such paths, raises
        ``no_trust_path`` / ``ambiguous_chain`` with one
        :class:`NormalizeChainIssue` per leaf *occurrence* (request order).

        The verdict per leaf is exact and independent of the diagnostic cap:
        the unique verified path to a trust root (or None) plus whether a
        second, distinct verified path exists. The bounded
        ``_enumerate_paths`` sample only fills the 422 diagnostics.
        ``no_path_message`` is the endpoint-specific message used when every
        failing leaf has no path at all.
        """
        leaves = list(leaves)
        edges = self.edges
        roots = frozenset(trust_roots)

        resolved: dict[str, tuple[Optional[list[str]], bool]] = {}
        for ref in leaves:
            if ref.fingerprint not in resolved:
                first = _find_trust_path(ref.fingerprint, edges, roots)
                ambiguous = first is not None and _has_second_trust_path(
                    first, edges, roots
                )
                resolved[ref.fingerprint] = (first, ambiguous)

        chosen: dict[str, list[str]] = {}
        chain_issues: list[NormalizeChainIssue] = []
        diagnostics: dict[str, tuple[list[list[str]], list[list[str]]]] = {}
        for ref in leaves:
            first, ambiguous = resolved[ref.fingerprint]
            if first is not None and not ambiguous:
                chosen[ref.fingerprint] = first
                continue
            if ref.fingerprint not in diagnostics:
                diagnostics[ref.fingerprint] = _enumerate_paths(
                    ref.fingerprint, edges, roots
                )
            valid, dead_ends = diagnostics[ref.fingerprint]
            chain_issues.append(NormalizeChainIssue(
                endpoint_id=ref.endpoint_id,
                leaf_role=ref.leaf_role,  # type: ignore[arg-type]
                leaf=ref.fingerprint,
                candidate_issuers=sorted(edges.get(ref.fingerprint, [])),
                candidate_paths=_dedupe_paths(valid + dead_ends),
                reason="ambiguous_chain" if ambiguous else "no_trust_path",
            ))

        if chain_issues:
            ambiguous = any(i.reason == "ambiguous_chain" for i in chain_issues)
            raise NormalizeError(
                "ambiguous_chain" if ambiguous else "no_trust_path",
                "cross-signing leaves multiple valid paths to a declared "
                "trust root" if ambiguous else no_path_message,
                chain_issues=chain_issues,
            )
        return chosen
