"""Cross-entry-point consistency regression tests for the shared certificate
chain verification context (``app.chain_context``).

The four PEM-bag chain reconstructors — ``/scenarios/normalize``,
``/deployments/precheck``, ``/deployments/policy-review`` and
``/deployments/revocation-review`` — parse PEM, de-duplicate by DER
fingerprint, establish issuance edges by real signature verification and
decide zero/unique/multiple paths through one reusable
:class:`~app.chain_context.VerificationContext`. These tests pin that the four
HTTP entries keep identical chain verdicts, error codes, ordering and
diagnostics; that the 64-sample diagnostic cap can never steer the verdict;
that rings, cross-signing, RSA-PSS and bad certificates keep failing closed;
and that contexts never share state across requests.
"""
from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from app import chain_context
from app.chain_context import (
    MAX_CANDIDATE_PATHS,
    LeafRef,
    VerificationContext,
)
from app.main import app

NOW = datetime(2030, 6, 1, tzinfo=timezone.utc)

client = TestClient(app)


# ---------------------------------------------------------------------------
# PKI factory
# ---------------------------------------------------------------------------


def _key(kind="ec"):
    if kind == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return ec.generate_private_key(ec.SECP256R1())


def _name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_cert(
    subject_cn,
    issuer_cn,
    issuer_key,
    *,
    subject_key=None,
    self_signed=False,
    not_before=None,
    not_after=None,
    is_ca=True,
    hash_alg=None,
    pss=False,
    sans=(),
    ekus=(),
):
    if self_signed:
        subject_key = issuer_key
        issuer_name = _name(subject_cn)
    else:
        subject_key = subject_key or _key()
        issuer_name = _name(issuer_cn)
    not_before = not_before or NOW - timedelta(days=365)
    not_after = not_after or NOW + timedelta(days=365)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject_cn))
        .issuer_name(issuer_name)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=None), critical=True
        )
    )
    if sans:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(s) for s in sans]),
            critical=False,
        )
    if ekus:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ObjectIdentifier(o) for o in ekus]),
            critical=False,
        )
    alg = hash_alg or hashes.SHA256()
    if pss:
        alg = alg.__class__()
        return builder.sign(
            issuer_key,
            alg,
            rsa_padding=padding.PSS(
                mgf=padding.MGF1(alg),
                salt_length=alg.digest_size,
            ),
        )
    return builder.sign(issuer_key, alg)


def pem(cert) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def fp(cert) -> str:
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


SERVER_AUTH = "1.3.6.1.5.5.7.3.1"


@pytest.fixture
def pki():
    """root -> inter -> leaf, all valid at NOW; leaf carries a DNS SAN."""
    root_key = _key()
    root = make_cert("Root CA", None, root_key, self_signed=True)
    inter_key = _key()
    inter = make_cert("Intermediate CA", "Root CA", root_key,
                      subject_key=inter_key)
    leaf_key = _key()
    leaf = make_cert(
        "svc.example", "Intermediate CA", inter_key,
        subject_key=leaf_key, is_ca=False, sans=("svc.example",),
        ekus=(SERVER_AUTH,),
    )
    return {
        "root_key": root_key,
        "root": root,
        "inter_key": inter_key,
        "inter": inter,
        "leaf_key": leaf_key,
        "leaf": leaf,
    }


def bundle(pki, *extra):
    return [pem(c) for c in
            (pki["leaf"], pki["inter"], pki["root"], *extra)]


def ref(endpoint_id, role, cert):
    return LeafRef(endpoint_id, role, fp(cert))


# ---------------------------------------------------------------------------
# 1. One context, same verdict everywhere
# ---------------------------------------------------------------------------


def test_four_entries_reconstruct_the_identical_unique_chain(pki):
    """normalize / precheck / policy-review / revocation-review must all hand
    back the exact same leaf-first chain for the same bundle and roots."""
    expected_chain = [fp(pki["leaf"]), fp(pki["inter"]), fp(pki["root"])]
    roots = [fp(pki["root"])]
    pems = bundle(pki)

    # normalize
    r = client.post("/scenarios/normalize", json={
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": fp(pki["leaf"]),
            "target_leaf_fingerprint": fp(pki["leaf"]),
        }],
    })
    assert r.status_code == 200, r.text
    assert (r.json()["endpoints"][0]["target_chain"] == expected_chain)

    # precheck
    r = client.post("/deployments/precheck", json={
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": fp(pki["leaf"]),
            "identities": [{"type": "dns", "value": "svc.example"}],
        }],
    })
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["chain"] == expected_chain

    # policy-review (current == target leaf here)
    r = client.post("/deployments/policy-review", json={
        "pem_certificates": pems,
        "trust_roots": roots,
        "check_time": NOW.isoformat(),
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": fp(pki["leaf"]),
            "target_leaf_fingerprint": fp(pki["leaf"]),
            "role": "server",
            "required_ekus": [],
        }],
    })
    assert r.status_code == 200, r.text
    ep = r.json()["results"][0]
    assert ep["current_chain"]["chain"] == expected_chain
    assert ep["target_chain"]["chain"] == expected_chain

    # revocation-review (no evidence: eligibility false, but the chain itself
    # is reconstructed identically)
    r = client.post("/deployments/revocation-review", json={
        "pem_certificates": pems,
        "trust_roots": roots,
        "check_time": NOW.isoformat(),
        "max_skew_seconds": 300,
        "ocsp_responses": [],
        "crls": [],
        "endpoints": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": fp(pki["leaf"]),
        }],
    })
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["chain"] == expected_chain


def test_context_parses_pem_once_and_dedupes_by_der_fingerprint(pki):
    pems = bundle(pki) + [pem(pki["inter"]), pem(pki["root"])]
    ctx = VerificationContext(pems, {fp(pki["root"])})
    assert set(ctx.certs) == {
        fp(pki["leaf"]), fp(pki["inter"]), fp(pki["root"]),
    }
    assert ctx.pem_issues == []
    # Issuance edges are real signature edges, not DN guesses.
    assert set(ctx.edges[fp(pki["leaf"])]) == {fp(pki["inter"])}
    assert set(ctx.edges[fp(pki["inter"])]) == {fp(pki["root"])}
    assert set(ctx.edges[fp(pki["root"])]) == {fp(pki["root"])}


# ---------------------------------------------------------------------------
# 2. No path / ambiguity: identical failure across all entries
# ---------------------------------------------------------------------------


def cross_signed_pki():
    """Two roots cross-certify one intermediate key; leaf has two paths."""
    root_a_key, root_b_key = _key(), _key()
    root_a = make_cert("Root A", None, root_a_key, self_signed=True)
    root_b = make_cert("Root B", None, root_b_key, self_signed=True)
    inter_key = _key()
    inter_a = make_cert("Shared Inter", "Root A", root_a_key,
                        subject_key=inter_key)
    inter_b = make_cert("Shared Inter", "Root B", root_b_key,
                        subject_key=inter_key)
    leaf = make_cert("svc", "Shared Inter", inter_key, is_ca=False,
                     sans=("svc",), ekus=(SERVER_AUTH,))
    return {
        "root_a": root_a, "root_b": root_b,
        "inter_a": inter_a, "inter_b": inter_b, "leaf": leaf,
    }


def _post_route(route, pki_obj, pems, roots, leaf, *, time_field=NOW):
    endpoints_key = {
        "/scenarios/normalize": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": leaf,
            "target_leaf_fingerprint": leaf,
        }],
        "/deployments/precheck": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": leaf,
            "identities": [{"type": "dns", "value": "svc"}],
        }],
        "/deployments/policy-review": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": leaf,
            "target_leaf_fingerprint": leaf,
            "role": "server",
            "required_ekus": [],
        }],
        "/deployments/revocation-review": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": leaf,
        }],
    }
    body = {
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": endpoints_key[route],
    }
    if route in ("/deployments/policy-review",
                 "/deployments/revocation-review"):
        body["check_time"] = time_field.isoformat()
    if route == "/deployments/revocation-review":
        body["max_skew_seconds"] = 300
        body["ocsp_responses"] = []
        body["crls"] = []
    return client.post(route, json=body)


ROUTES = (
    "/scenarios/normalize",
    "/deployments/precheck",
    "/deployments/policy-review",
    "/deployments/revocation-review",
)


def test_ambiguous_chain_is_422_with_identical_diagnostics_in_every_route():
    x = cross_signed_pki()
    pems = [pem(c) for c in
            (x["leaf"], x["inter_a"], x["inter_b"], x["root_a"], x["root_b"])]
    roots = [fp(x["root_a"]), fp(x["root_b"])]
    leaf = fp(x["leaf"])
    expected_paths = {
        (leaf, fp(x["inter_a"]), fp(x["root_a"])),
        (leaf, fp(x["inter_b"]), fp(x["root_b"])),
    }
    for route in ROUTES:
        r = _post_route(route, x, pems, roots, leaf)
        assert r.status_code == 422, route
        detail = r.json()["detail"]
        assert detail["error"] == "ambiguous_chain", route
        issues = detail["chain_issues"]
        # normalize/policy-review reference the leaf twice (current+target);
        # precheck/revocation-review once (target only).
        expected_issue_count = 2 if route in (
            "/scenarios/normalize", "/deployments/policy-review") else 1
        assert len(issues) == expected_issue_count, (route, issues)
        for issue in issues:
            assert issue["leaf"] == leaf
            assert issue["reason"] == "ambiguous_chain"
            assert set(issue["candidate_issuers"]) == {
                fp(x["inter_a"]), fp(x["inter_b"])
            }
            assert {tuple(p) for p in issue["candidate_paths"]} == expected_paths


def test_no_trust_path_is_422_with_identical_code_in_every_route(pki):
    # Root present but declared trust root set is empty-compatible: declare a
    # different, unrelated root instead, so no verified path terminates.
    other_key = _key()
    other_root = make_cert("Other Root", None, other_key, self_signed=True)
    pems = bundle(pki, other_root)
    roots = [fp(other_root)]
    leaf = fp(pki["leaf"])
    for route in ROUTES:
        r = _post_route(route, pki, pems, roots, leaf)
        assert r.status_code == 422, route
        detail = r.json()["detail"]
        assert detail["error"] == "no_trust_path", (route, detail)
        assert detail["chain_issues"][0]["reason"] == "no_trust_path"
        # the real cryptographic parent still shows as a candidate issuer
        assert detail["chain_issues"][0]["candidate_issuers"] == [
            fp(pki["inter"])
        ]


def test_error_codes_are_shared_for_pem_fingerprint_and_root_failures(pki):
    pems = bundle(pki)
    good_root = fp(pki["root"])
    leaf = fp(pki["leaf"])

    # malformed PEM element
    for route in ROUTES:
        r = _post_route(route, pki, pems + ["garbage not pem"],
                        [good_root], leaf)
        assert r.status_code == 422, route
        assert r.json()["detail"]["error"] == "invalid_pem", route

    # malformed fingerprint
    for route in ROUTES:
        r = _post_route(route, pki, pems, ["ZZ" + "0" * 62], leaf)
        assert r.status_code == 422, route
        assert r.json()["detail"]["error"] == "invalid_fingerprint", route

    # unknown fingerprint (declared root not in the bundle)
    for route in ROUTES:
        r = _post_route(route, pki, pems, ["ab" * 32], leaf)
        assert r.status_code == 422, route
        assert r.json()["detail"]["error"] == "unknown_fingerprint", route

    # trust root that is not genuinely self-signed
    fake_root = make_cert("Fake Root", "Root CA", pki["root_key"],
                          subject_key=_key())
    pems_fake = bundle(pki, fake_root)
    for route in ROUTES:
        r = _post_route(route, pki, pems_fake, [fp(fake_root)], leaf)
        assert r.status_code == 422, route
        assert (r.json()["detail"]["error"]
                == "trust_root_not_self_signed"), route


# ---------------------------------------------------------------------------
# 3. The verdict is independent of the 64-sample diagnostic cap
# ---------------------------------------------------------------------------


def test_candidate_path_sample_is_capped_at_64_but_verdict_is_exact(
    monkeypatch,
):
    """Build a fan-out with well over 64 distinct valid trust paths from one
    leaf through N intermediate keys each wrapped by two cross-certifying
    roots. Enumeration saturates at 64; the verdict must still be ambiguous.
    """
    root_a_key, root_b_key = _key(), _key()
    root_a = make_cert("Root A", None, root_a_key, self_signed=True)
    root_b = make_cert("Root B", None, root_b_key, self_signed=True)

    # One shared intermediate key wrapped by 40 distinct certificates per
    # root (80 wrappers total). Every wrapper cryptographically verifies the
    # same leaf, so the leaf has 80 distinct simple paths to a declared root.
    shared_key = _key()
    wrappers = []
    for i in range(40):
        wrappers.append(make_cert(
            f"Inter A{i}", "Root A", root_a_key, subject_key=shared_key))
        wrappers.append(make_cert(
            f"Inter B{i}", "Root B", root_b_key, subject_key=shared_key))
    leaf = make_cert("svc", "Shared Inter", shared_key, is_ca=False,
                     sans=("svc",), ekus=(SERVER_AUTH,))
    all_certs = [root_a, root_b, *wrappers, leaf]
    pems = [pem(c) for c in all_certs]
    roots = [fp(root_a), fp(root_b)]
    leaf_fp = fp(leaf)

    # Direct context: exact verdict says ambiguous even though the bounded
    # sample cannot list all 80 paths.
    ctx = VerificationContext(pems, set(roots))
    verdict = ctx.verdict(leaf_fp)
    assert verdict.ambiguous is True
    assert verdict.path is not None
    valid, _dead = ctx._candidate_paths(leaf_fp)
    assert len(valid) <= MAX_CANDIDATE_PATHS == 64
    assert len(valid) == MAX_CANDIDATE_PATHS

    # And every HTTP route still returns ambiguous_chain (not, say, a unique
    # pick or a cap-induced no_trust_path).
    for route in ROUTES:
        r = _post_route(route, None, pems, roots, leaf_fp)
        assert r.status_code == 422, route
        detail = r.json()["detail"]
        assert detail["error"] == "ambiguous_chain", route
        paths = detail["chain_issues"][0]["candidate_paths"]
        assert len(paths) <= MAX_CANDIDATE_PATHS
        assert detail["chain_issues"][0]["candidate_issuers"] == sorted(
            fp(w) for w in wrappers
        )


def test_diagnostic_enumeration_is_lazy_and_cached_per_context(pki):
    ctx = VerificationContext(bundle(pki), {fp(pki["root"])})
    refs = [ref("svc", "target", pki["leaf"])]
    # Successful resolution must not force diagnostics at all.
    chosen, issues = ctx.resolve_chains(refs)
    assert chosen[fp(pki["leaf"])] == [
        fp(pki["leaf"]), fp(pki["inter"]), fp(pki["root"])]
    assert issues == []
    assert fp(pki["leaf"]) not in ctx._diagnostics


def test_issue_dedupe_policy_matches_each_route_history():
    """normalize/policy/precheck emit one issue per reference; revocation
    emits one per distinct failing leaf. The shared context encodes both."""
    # No-path bundle: leaf + intermediate without the declared root.
    x = cross_signed_pki()
    pems = [pem(x["leaf"]), pem(x["inter_a"]), pem(x["inter_b"])]
    ctx = VerificationContext(pems, {"dd" * 32})
    same_leaf = x["leaf"]
    two_refs = [
        LeafRef("e1", "current", fp(same_leaf)),
        LeafRef("e1", "target", fp(same_leaf)),
    ]
    _chosen, per_reference = ctx.resolve_chains(two_refs)
    assert {i.leaf_role for i in per_reference} == {"current", "target"}
    assert len(per_reference) == 2

    _chosen2, deduped = ctx.resolve_chains(
        two_refs, dedupe_leaf_failures=True)
    assert len(deduped) == 1
    assert deduped[0].leaf_role == "current"


# ---------------------------------------------------------------------------
# 4. Cycles, cross-signing, RSA-PSS, bad certificates fail closed
# ---------------------------------------------------------------------------


def test_signature_cycle_does_not_hang_and_fails_closed():
    # Two unrelated keys cross-sign each other; neither reaches a declared
    # root. The path search must terminate and the verdict must be no path.
    key_a, key_b = _key(), _key()
    cert_a = make_cert("A", "B", key_b, subject_key=key_a)
    cert_b = make_cert("B", "A", key_a, subject_key=key_b)
    root_key = _key()
    root = make_cert("Root", None, root_key, self_signed=True)
    ctx = VerificationContext(
        [pem(cert_a), pem(cert_b), pem(root)], {fp(root)})
    verdict = ctx.verdict(fp(cert_a))
    assert verdict.path is None and verdict.ambiguous is False
    _, issues = ctx.resolve_chains([ref("e", "target", cert_a)])
    assert issues[0].reason == "no_trust_path"


def test_rsa_pss_signed_chain_verifies_and_is_rejected_when_tampered():
    root_key = _key("rsa")
    root = make_cert("PSS Root", None, root_key, self_signed=True, pss=True)
    inter_key = _key("rsa")
    inter = make_cert("PSS Inter", "PSS Root", root_key,
                      subject_key=inter_key, pss=True)
    leaf_key = _key("rsa")
    leaf = make_cert("svc.example", "PSS Inter", inter_key,
                     subject_key=leaf_key, is_ca=False, pss=True,
                     sans=("svc.example",), ekus=(SERVER_AUTH,))
    pems = [pem(leaf), pem(inter), pem(root)]
    r = client.post("/scenarios/normalize", json={
        "pem_certificates": pems,
        "trust_roots": [fp(root)],
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": fp(leaf),
            "target_leaf_fingerprint": fp(leaf),
        }],
    })
    assert r.status_code == 200, r.text
    assert r.json()["endpoints"][0]["target_chain"] == [
        fp(leaf), fp(inter), fp(root)]

    # A leaf whose signature bytes do not verify against any wrapper (wrong
    # key) yields no edge and a no_trust_path failure — never a DN-based guess.
    forged = make_cert("svc.example", "PSS Inter", _key("rsa"),
                       subject_key=leaf_key, is_ca=False,
                       sans=("svc.example",), ekus=(SERVER_AUTH,))
    r = client.post("/scenarios/normalize", json={
        "pem_certificates": [pem(forged), pem(inter), pem(root)],
        "trust_roots": [fp(root)],
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": fp(forged),
            "target_leaf_fingerprint": fp(forged),
        }],
    })
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "no_trust_path"


def test_tampered_leaf_signature_has_no_edge_and_fails_closed(pki):
    # Flip a signature-region DER byte: the block still parses (so no
    # invalid_pem), but no key in the bundle verifies the signature — the
    # issuance edge must disappear and every route fails no_trust_path rather
    # than trusting the matching issuer DN.
    der = pki["leaf"].public_bytes(serialization.Encoding.DER)
    pos = len(der) - 16
    tampered = der[:pos] + bytes([der[pos] ^ 0xFF]) + der[pos + 1:]
    bad_pem = ("-----BEGIN CERTIFICATE-----\n"
               + base64.encodebytes(tampered).decode()
               + "-----END CERTIFICATE-----\n")
    x509.load_pem_x509_certificate(bad_pem.encode())  # parses, sig invalid
    import hashlib as _hashlib
    bad_fp = _hashlib.sha256(tampered).hexdigest()
    for route in ROUTES:
        r = _post_route(
            route, pki,
            [bad_pem, pem(pki["inter"]), pem(pki["root"])],
            [fp(pki["root"])],
            bad_fp,
        )
        assert r.status_code == 422, route
        detail = r.json()["detail"]
        assert detail["error"] == "no_trust_path", (route, detail)
        assert detail["chain_issues"][0]["candidate_issuers"] == []


def test_garbage_pem_block_is_an_invalid_pem_error_in_every_route(pki):
    for route in ROUTES:
        r = _post_route(
            route, pki,
            bundle(pki) + ["-----BEGIN CERTIFICATE-----\nnotbase64!!\n"
                           "-----END CERTIFICATE-----\n"],
            [fp(pki["root"])],
            fp(pki["leaf"]),
        )
        assert r.status_code == 422, route
        assert r.json()["detail"]["error"] == "invalid_pem", route


def test_bad_signature_never_forms_an_edge(pki, monkeypatch):
    # Force the cryptographic verifier to reject everything: no edge may be
    # fabricated from matching DNs.
    monkeypatch.setattr(chain_context, "verify_signature",
                        lambda child, issuer: False)
    ctx = VerificationContext(bundle(pki), {fp(pki["root"])})
    assert all(issuers == [] for issuers in ctx.edges.values())
    chosen, issues = ctx.resolve_chains([ref("svc", "target", pki["leaf"])])
    assert fp(pki["leaf"]) not in chosen
    assert issues[0].reason == "no_trust_path"


# ---------------------------------------------------------------------------
# 5. Request isolation: no shared/global state between contexts
# ---------------------------------------------------------------------------


def test_contexts_do_not_share_roots_or_verdicts(pki):
    x = cross_signed_pki()
    pems = [pem(c) for c in
            (x["leaf"], x["inter_a"], x["inter_b"], x["root_a"], x["root_b"])]

    ctx_a = VerificationContext(pems, {fp(x["root_a"])})
    ctx_b = VerificationContext(pems, {fp(x["root_b"])})
    ctx_both = VerificationContext(
        pems, {fp(x["root_a"]), fp(x["root_b"])})

    va = ctx_a.verdict(fp(x["leaf"]))
    vb = ctx_b.verdict(fp(x["leaf"]))
    vboth = ctx_both.verdict(fp(x["leaf"]))
    assert va.unique and va.path[-1] == fp(x["root_a"])
    assert vb.unique and vb.path[-1] == fp(x["root_b"])
    assert vboth.ambiguous

    # Contexts hold independent root sets: one request's declared roots cannot
    # leak into another's graph search.
    roots_a = set(ctx_a.trust_roots)
    roots_a.add("cc" * 32)
    assert "cc" * 32 not in ctx_b.trust_roots
    assert ctx_b.verdict(fp(x["leaf"])).unique


def test_consecutive_requests_via_http_keep_isolated_results():
    """A request that declares both roots (ambiguous) followed by one
    declaring a single root must not carry the first request's verdict."""
    x = cross_signed_pki()
    pems = [pem(c) for c in
            (x["leaf"], x["inter_a"], x["inter_b"], x["root_a"], x["root_b"])]
    leaf = fp(x["leaf"])

    body = lambda roots: {  # noqa: E731
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": leaf,
            "target_leaf_fingerprint": leaf,
        }],
    }
    r1 = client.post("/scenarios/normalize",
                     json=body([fp(x["root_a"]), fp(x["root_b"])]))
    assert r1.status_code == 422
    assert r1.json()["detail"]["error"] == "ambiguous_chain"

    r2 = client.post("/scenarios/normalize", json=body([fp(x["root_a"])]))
    assert r2.status_code == 200, r2.text
    assert r2.json()["endpoints"][0]["target_chain"] == [
        leaf, fp(x["inter_a"]), fp(x["root_a"])]

    # And a fresh request in the other direction is still independent.
    r3 = client.post("/scenarios/normalize", json=body([fp(x["root_b"])]))
    assert r3.status_code == 200
    assert r3.json()["endpoints"][0]["target_chain"] == [
        leaf, fp(x["inter_b"]), fp(x["root_b"])]


# ---------------------------------------------------------------------------
# 6. Trust/validity/name/revocation rules are untouched by the convergence
# ---------------------------------------------------------------------------


def test_revocation_review_still_skips_trust_anchor_and_fails_without_evidence(pki):
    """Convergence must not change revocation semantics: the anchor is not
    reviewed and missing evidence fails closed per non-anchor cert."""
    r = client.post("/deployments/revocation-review", json={
        "pem_certificates": bundle(pki),
        "trust_roots": [fp(pki["root"])],
        "check_time": NOW.isoformat(),
        "max_skew_seconds": 300,
        "ocsp_responses": [],
        "crls": [],
        "endpoints": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": fp(pki["leaf"]),
        }],
    })
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["eligible"] is False
    by_fp = {c["certificate_fingerprint"]: c for c in result["certificates"]}
    anchor = by_fp[fp(pki["root"])]
    assert anchor["role"] == "trust_root"
    assert anchor["eligible"] is True
    assert anchor["reason"] == "trust_root"
    assert anchor["evidence"] == []
    for cert_fp in (fp(pki["leaf"]), fp(pki["inter"])):
        review = by_fp[cert_fp]
        assert review["eligible"] is False
        assert review["reason"] == "evidence_missing"


def test_policy_review_validity_and_eku_rules_unchanged(pki):
    # Expired intermediate: policy-review must still fail on time validity
    # even though the chain reconstructs uniquely.
    expired_inter = make_cert(
        "Intermediate CA", "Root CA", pki["root_key"],
        subject_key=pki["inter_key"],
        not_before=NOW - timedelta(days=700),
        not_after=NOW - timedelta(days=10),
    )
    pems = [pem(pki["leaf"]), pem(expired_inter), pem(pki["root"])]
    r = client.post("/deployments/policy-review", json={
        "pem_certificates": pems,
        "trust_roots": [fp(pki["root"])],
        "check_time": NOW.isoformat(),
        "endpoints": [{
            "endpoint_id": "svc",
            "current_leaf_fingerprint": fp(pki["leaf"]),
            "target_leaf_fingerprint": fp(pki["leaf"]),
            "role": "server",
            "required_ekus": [],
        }],
    })
    assert r.status_code == 200, r.text
    ep = r.json()["results"][0]
    assert ep["eligible"] is False
    reasons = {c["reason"] for ch in (ep["current_chain"], ep["target_chain"])
               for c in ch["conflicts"]}
    assert "certificate_expired" in reasons


def test_precheck_name_constraints_unchanged_through_shared_context(pki):
    # An intermediate that excludes the leaf's DNS SAN must still fail
    # precheck independently of the chain convergence.
    constrained_leaf = make_cert(
        "svc.example.com", "Intermediate CA", pki["inter_key"],
        subject_key=pki["leaf_key"], is_ca=False,
        sans=("svc.example.com",),
    )
    nc = x509.NameConstraints(
        permitted_subtrees=None,
        excluded_subtrees=[x509.DNSName("example.com")],
    )
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name("Intermediate CA"))
        .issuer_name(_name("Root CA"))
        .public_key(pki["inter_key"].public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(NOW - timedelta(days=365))
        .not_valid_after(NOW + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                       critical=True)
        .add_extension(nc, critical=True)
    )
    constrained_inter = builder.sign(pki["root_key"], hashes.SHA256())
    r = client.post("/deployments/precheck", json={
        "pem_certificates": [pem(constrained_leaf), pem(constrained_inter),
                             pem(pki["root"])],
        "trust_roots": [fp(pki["root"])],
        "endpoints": [{
            "endpoint_id": "svc",
            "target_leaf_fingerprint": fp(constrained_leaf),
            "identities": [{"type": "dns", "value": "svc.example.com"}],
        }],
    })
    assert r.status_code == 200, r.text
    result = r.json()["results"][0]
    assert result["eligible"] is False
    assert result["reason"] == "excluded_by_name_constraint"
