"""Normalize a shuffled PEM bundle into planner-ready certificate chains.

The input is an unordered bag of PEM-encoded X.509 certificates plus, for each
endpoint, the SHA-256 fingerprints of its current/target leaf certificates and
the fingerprints of the self-signed roots that are allowed to terminate a chain.

Issuance edges are established by *cryptographically verifying* the child's
signature with the candidate issuer's public key — never by comparing
subject/issuer distinguished names (two unrelated CAs may share a DN, and
cross-certified CAs share a key under several distinct certificates). For every
leaf we then decide — exactly, and independently of any diagnostic limit —
whether zero, one or several leaf-first simple paths over those signed edges
terminate at a declared trust root; exactly one such path must exist.

All of that machinery lives in :mod:`app.chaincontext`
(:class:`ChainVerificationContext`), shared with the ``/deployments`` chain
endpoints; this module only keeps the normalize-specific request shaping and
the success DTO.
"""
from __future__ import annotations

from .chaincontext import (
    FINGERPRINT_RE,
    ChainVerificationContext,
    LeafReference,
    NormalizeError,
    _dn,
    _validity_utc,
    fingerprint_format_issues,
    normalize_fingerprint,
)
from .models import (
    NormalizeRequest,
    NormalizeResult,
    NormalizedCertificate,
    NormalizedEndpoint,
)

__all__ = ["NormalizeError", "normalize_scenario"]


def normalize_scenario(req: NormalizeRequest) -> NormalizeResult:
    # -- fingerprint formatting ----------------------------------------------
    fp_refs: list[tuple[str, str]] = [
        (raw, "trust_roots") for raw in req.trust_roots
    ]
    leaves: list[LeafReference] = []
    for e in req.endpoints:
        for role, raw in (
            ("current", e.current_leaf_fingerprint),
            ("target", e.target_leaf_fingerprint),
        ):
            fp_refs.append((
                raw,
                f"endpoints['{e.endpoint_id}'].{role}_leaf_fingerprint",
            ))
            leaves.append(LeafReference(
                e.endpoint_id, role, normalize_fingerprint(raw)
            ))
    fingerprint_issues = fingerprint_format_issues(fp_refs)
    trust_roots = {
        fp
        for fp in map(normalize_fingerprint, req.trust_roots)
        if FINGERPRINT_RE.match(fp)
    }

    endpoint_ids = [e.endpoint_id for e in req.endpoints]
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

    # -- shared verification context: parse, dedupe, verify, resolve ----------
    ctx = ChainVerificationContext(req.pem_certificates)

    ctx.require_known_fingerprints(
        [(fp, "trust_roots") for fp in sorted(trust_roots)]
        + [
            (ref.fingerprint,
             f"endpoints['{ref.endpoint_id}'].{ref.leaf_role}_leaf_fingerprint")
            for ref in leaves
        ]
    )
    ctx.require_self_signed_roots(trust_roots)
    chosen = ctx.resolve_unique_chains(
        leaves,
        trust_roots,
        no_path_message="no verified signature path from a leaf to any "
                        "declared trust root",
    )

    # -- success DTO -----------------------------------------------------------
    normalized_certs = [
        NormalizedCertificate(
            cert_id=fp,
            subject=_dn(ctx.certs[fp].subject),
            issuer=_dn(ctx.certs[fp].issuer),
            not_valid_before=_validity_utc(ctx.certs[fp], "not_valid_before"),
            not_valid_after=_validity_utc(ctx.certs[fp], "not_valid_after"),
        )
        for fp in sorted(ctx.certs)
    ]
    normalized_endpoints = [
        NormalizedEndpoint(
            endpoint_id=e.endpoint_id,
            current_chain=chosen[normalize_fingerprint(
                e.current_leaf_fingerprint)],
            target_chain=chosen[normalize_fingerprint(
                e.target_leaf_fingerprint)],
        )
        for e in req.endpoints
    ]
    return NormalizeResult(
        certificates=normalized_certs,
        endpoints=normalized_endpoints,
    )
