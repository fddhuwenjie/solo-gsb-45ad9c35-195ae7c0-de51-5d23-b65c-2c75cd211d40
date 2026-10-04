"""Normalize a shuffled PEM bundle into planner-ready certificate chains.

The input is an unordered bag of PEM-encoded X.509 certificates plus, for each
endpoint, the SHA-256 fingerprints of its current/target leaf certificates and
the fingerprints of the self-signed roots that are allowed to terminate a chain.

All PEM parsing, DER-fingerprint de-duplication, signature-verified issuance
edges and the exact zero/unique/ambiguous path verdict live in the shared
:mod:`app.chain_context` verification context, together with
``/deployments/precheck``, ``/deployments/policy-review`` and
``/deployments/revocation-review``. This module owns only the normalize
route's request validation, error codes and success DTO.
"""
from __future__ import annotations

from .chain_context import (
    FINGERPRINT_RE,
    LeafRef,
    NormalizeError,
    VerificationContext,
    _find_trust_path,
    _has_second_trust_path,
    cert_fingerprint,
    chain_resolution_error,
    dn_str as _dn,
    load_pem_bundle as _load_pem_bundle,
    normalize_fingerprint,
    validity_utc as _validity_utc,
)
from .models import (
    NormalizeFingerprintIssue,
    NormalizedCertificate,
    NormalizedEndpoint,
    NormalizeRequest,
    NormalizeResult,
)

__all__ = [
    "FINGERPRINT_RE",
    "NormalizeError",
    "cert_fingerprint",
]


def normalize_scenario(req: NormalizeRequest) -> NormalizeResult:
    # -- fingerprint formatting ----------------------------------------------
    fingerprint_issues: list[NormalizeFingerprintIssue] = []
    trust_roots: set[str] = set()
    for raw in req.trust_roots:
        fp = normalize_fingerprint(raw)
        if fp is None:
            fingerprint_issues.append(NormalizeFingerprintIssue(
                fingerprint=raw, referred_by="trust_roots",
            ))
        else:
            trust_roots.add(fp)

    endpoint_ids = [e.endpoint_id for e in req.endpoints]
    # (endpoint, role, reference)
    refs: list[LeafRef] = []
    for e in req.endpoints:
        for role, raw in (
            ("current", e.current_leaf_fingerprint),
            ("target", e.target_leaf_fingerprint),
        ):
            fp = normalize_fingerprint(raw)
            if fp is None:
                fingerprint_issues.append(NormalizeFingerprintIssue(
                    fingerprint=raw,
                    referred_by=f"endpoints['{e.endpoint_id}'].{role}_leaf_fingerprint",
                ))
            refs.append(LeafRef(e.endpoint_id, role, raw))

    if len(set(endpoint_ids)) != len(endpoint_ids):
        raise NormalizeError(
            "invalid_request",
            "endpoints: endpoint_id values must be unique",
        )
    if fingerprint_issues:
        raise NormalizeError(
            "invalid_fingerprint",
            "one or more fingerprints are not 64-character hex SHA-256 values",
            fingerprint_issues=fingerprint_issues,
        )

    # -- shared verification context (parse once, real signature edges) -------
    ctx = VerificationContext(req.pem_certificates, trust_roots, refs)
    if ctx.pem_issues:
        raise NormalizeError(
            "invalid_pem",
            "one or more PEM list elements could not be parsed",
            pem_issues=ctx.pem_issues,
        )

    # -- referential integrity ------------------------------------------------
    missing = ctx.missing_fingerprint_issues()
    if missing:
        raise NormalizeError(
            "unknown_fingerprint",
            "some fingerprints do not match any certificate in the PEM bundle",
            fingerprint_issues=missing,
        )

    # -- trust roots must genuinely be self-signed ----------------------------
    bad_roots = ctx.bad_root_issues()
    if bad_roots:
        raise NormalizeError(
            "trust_root_not_self_signed",
            "every declared trust root must be a self-signed certificate "
            "(subject == issuer with a valid self-signature)",
            fingerprint_issues=bad_roots,
        )

    # -- per-leaf exact reconstruction via the shared context -----------------
    chosen, chain_issues = ctx.resolve_chains(refs)
    if chain_issues:
        raise chain_resolution_error(
            chain_issues,
            ambiguous_message="cross-signing leaves multiple valid paths to a "
                             "declared trust root",
            no_path_message="no verified signature path from a leaf to any "
                            "declared trust root",
        )

    # -- success DTO -----------------------------------------------------------
    certs = ctx.certs
    normalized_certs = [
        NormalizedCertificate(
            cert_id=fp,
            subject=_dn(certs[fp].subject),
            issuer=_dn(certs[fp].issuer),
            not_valid_before=_validity_utc(certs[fp], "not_valid_before"),
            not_valid_after=_validity_utc(certs[fp], "not_valid_after"),
        )
        for fp in sorted(certs)
    ]
    normalized_endpoints = [
        NormalizedEndpoint(
            endpoint_id=e.endpoint_id,
            current_chain=chosen[  # type: ignore[arg-type]
                e.current_leaf_fingerprint.strip().lower()],
            target_chain=chosen[  # type: ignore[arg-type]
                e.target_leaf_fingerprint.strip().lower()],
        )
        for e in req.endpoints
    ]
    return NormalizeResult(
        certificates=normalized_certs,
        endpoints=normalized_endpoints,
    )
