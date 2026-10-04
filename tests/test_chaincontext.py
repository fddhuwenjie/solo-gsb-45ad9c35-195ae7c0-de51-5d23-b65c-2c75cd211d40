"""Cross-entry consistency regression tests for ChainVerificationContext.

``/scenarios/normalize``, ``/deployments/precheck``,
``/deployments/policy-review`` and ``/deployments/revocation-review`` all
resolve certificate chains through the shared
``app.chaincontext.ChainVerificationContext``: one PEM parse, DER-fingerprint
dedupe, real signature edges and the exact zero/unique/ambiguous verdict —
computed independently of the bounded (<=64) diagnostic path sample carried
by 422 responses.

These tests pin that the four entries agree on every verdict class over the
*same* bundle (success chains, no_trust_path, ambiguous_chain, cycles,
cross-signing, RSA-PSS, tampered certificates), that each entry keeps its
own output shape and error codes, and that no state leaks between requests.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from app.chaincontext import ChainVerificationContext, LeafReference
from app.main import app

NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)
CHECK_TIME = "2030-06-01T00:00:00Z"
SERVER_AUTH = "1.3.6.1.5.5.7.3.1"

client = TestClient(app)


# ---------------------------------------------------------------------------
# PKI factory
# ---------------------------------------------------------------------------


def _key(kind="ec"):
    if kind == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return ec.generate_private_key(ec.SECP256R1())


def _dn(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_cert(
    subject_cn,
    issuer_cn,
    issuer_key,
    *,
    subject_key=None,
    self_signed=False,
    is_ca=True,
    dns_names=(),
    ekus=(),
    rsa_pss=False,
    not_before=None,
    not_after=None,
):
    """Build a certificate; when self_signed, subject_key == issuer_key."""
    if self_signed:
        subject_key = issuer_key
        issuer_name = _dn(subject_cn)
    else:
        subject_key = subject_key or _key()
        issuer_name = _dn(issuer_cn)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_dn(subject_cn))
        .issuer_name(issuer_name)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before or datetime(2024, 1, 1))
        .not_valid_after(not_after or datetime(2035, 1, 1))
        .add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=not is_ca,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=is_ca,
                crl_sign=is_ca,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    )
    if dns_names:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in dns_names]),
            critical=False,
        )
    if ekus:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ObjectIdentifier(o) for o in ekus]),
            critical=False,
        )
    if rsa_pss:
        return builder.sign(
            issuer_key,
            hashes.SHA256(),
            rsa_padding=padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=32
            ),
        )
    return builder.sign(issuer_key, hashes.SHA256())


def pem(cert) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def fp(cert) -> str:
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


def make_leaf(cn, issuer_cn, issuer_key, **kwargs):
    """A server leaf whose SAN/EKU also satisfy precheck/policy-review."""
    return make_cert(
        cn, issuer_cn, issuer_key,
        is_ca=False, dns_names=[cn], ekus=[SERVER_AUTH], **kwargs,
    )


@pytest.fixture
def pki():
    """root_old -> leaf_old and root_new -> inter_new -> leaf_new."""
    root_old_key = _key()
    root_old = make_cert("Old Root CA", None, root_old_key, self_signed=True)
    leaf_old = make_leaf("svc-old.example.com", "Old Root CA", root_old_key)

    root_new_key = _key()
    root_new = make_cert("New Root CA", None, root_new_key, self_signed=True)
    inter_new_key = _key()
    inter_new = make_cert("New Inter CA", "New Root CA", root_new_key,
                          subject_key=inter_new_key)
    leaf_new = make_leaf("svc.example.com", "New Inter CA", inter_new_key)
    return {
        "root_old": root_old,
        "leaf_old": leaf_old,
        "root_new": root_new,
        "inter_new": inter_new,
        "leaf_new": leaf_new,
    }


def _bundle(pki, *names):
    return [pem(pki[n]) for n in names]


# ---------------------------------------------------------------------------
# One request per chain-consuming entry over the same bundle
# ---------------------------------------------------------------------------

# eps: list of dicts {endpoint_id, current, target, dns} (fingerprints + the
# DNS identity precheck should certify for the target leaf).


def _post_all(pems, roots, eps):
    normalize_body = {
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": [
            {"endpoint_id": e["endpoint_id"],
             "current_leaf_fingerprint": e["current"],
             "target_leaf_fingerprint": e["target"]}
            for e in eps
        ],
    }
    precheck_body = {
        "pem_certificates": pems,
        "trust_roots": roots,
        "endpoints": [
            {"endpoint_id": e["endpoint_id"],
             "target_leaf_fingerprint": e["target"],
             "identities": [{"type": "dns", "value": e["dns"]}]}
            for e in eps
        ],
    }
    policy_body = {
        "pem_certificates": pems,
        "trust_roots": roots,
        "check_time": CHECK_TIME,
        "endpoints": [
            {"endpoint_id": e["endpoint_id"],
             "current_leaf_fingerprint": e["current"],
             "target_leaf_fingerprint": e["target"],
             "role": "server",
             "required_ekus": []}
            for e in eps
        ],
    }
    revocation_body = {
        "pem_certificates": pems,
        "trust_roots": roots,
        "check_time": CHECK_TIME,
        "endpoints": [
            {"endpoint_id": e["endpoint_id"],
             "target_leaf_fingerprint": e["target"]}
            for e in eps
        ],
    }
    return (
        client.post("/scenarios/normalize", json=normalize_body),
        client.post("/deployments/precheck", json=precheck_body),
        client.post("/deployments/policy-review", json=policy_body),
        client.post("/deployments/revocation-review", json=revocation_body),
    )


def _issue_keys(detail, leaf_role=None):
    """Chain issues as comparable tuples (order-preserving)."""
    return [
        (
            i["endpoint_id"],
            i["leaf_role"],
            i["leaf"],
            tuple(i["candidate_issuers"]),
            tuple(tuple(p) for p in i["candidate_paths"]),
            i["reason"],
        )
        for i in detail["chain_issues"]
        if leaf_role is None or i["leaf_role"] == leaf_role
    ]


def _assert_same_chain_failure(norm, pre, pol, rev, error):
    """All four entries rejected with the same error and identical issues.

    normalize/policy-review resolve current+target leaves, precheck/
    revocation-review only the target leaf; the shared target-leaf issues
    must be identical across all four, and normalize must agree with
    policy-review on the full list.
    """
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
    details = [r.json()["detail"] for r in (norm, pre, pol, rev)]
    assert [d["error"] for d in details] == [error] * 4
    assert _issue_keys(details[0]) == _issue_keys(details[2])
    target = _issue_keys(details[0], leaf_role="target")
    assert target == _issue_keys(details[1]) == _issue_keys(details[3])
    return details


# ---------------------------------------------------------------------------
# Success: identical chains from every entry
# ---------------------------------------------------------------------------


def test_success_chains_agree_across_entries(pki):
    # shuffled bundle with a duplicate DER copy in its own element
    pems = _bundle(pki, "leaf_new", "root_old", "inter_new", "leaf_old",
                   "root_new") + [pem(pki["inter_new"])]
    roots = [fp(pki["root_old"]), fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": fp(pki["leaf_new"]),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text

    current = [fp(pki["leaf_old"]), fp(pki["root_old"])]
    target = [fp(pki["leaf_new"]), fp(pki["inter_new"]), fp(pki["root_new"])]

    norm_ep = norm.json()["endpoints"][0]
    assert norm_ep["current_chain"] == current
    assert norm_ep["target_chain"] == target

    pre_ep = pre.json()["results"][0]
    assert pre_ep["eligible"] is True
    assert pre_ep["chain"] == target

    pol_ep = pol.json()["results"][0]
    assert pol_ep["eligible"] is True
    assert pol_ep["current_chain"]["chain"] == current
    assert pol_ep["target_chain"]["chain"] == target

    rev_ep = rev.json()["results"][0]
    # no evidence submitted: ineligible, but the reconstructed chain agrees
    assert rev_ep["reason"] == "evidence_missing"
    assert rev_ep["chain"] == target


def test_dedupe_and_concatenated_pem_agree_across_entries(pki):
    blob = "".join(_bundle(pki, "leaf_new", "inter_new", "root_new"))
    pems = [blob, pem(pki["leaf_old"]) + pem(pki["root_old"]),
            pem(pki["leaf_new"])]  # duplicate DER copy of the target leaf
    roots = [fp(pki["root_old"]).upper(), fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": "  " + fp(pki["leaf_old"]) + " ",
        "target": fp(pki["leaf_new"]).upper(),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text
    target = [fp(pki["leaf_new"]), fp(pki["inter_new"]), fp(pki["root_new"])]
    assert norm.json()["endpoints"][0]["target_chain"] == target
    assert pre.json()["results"][0]["chain"] == target
    assert pol.json()["results"][0]["target_chain"]["chain"] == target
    assert rev.json()["results"][0]["chain"] == target


# ---------------------------------------------------------------------------
# no_trust_path: identical verdict and diagnostics from every entry
# ---------------------------------------------------------------------------


def test_no_trust_path_agrees_across_entries(pki):
    # root_old is absent: the current leaf resolves, the target cannot
    pems = _bundle(pki, "leaf_old", "leaf_new", "inter_new", "root_new")
    roots = [fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_new"]),
        "target": fp(pki["leaf_old"]),
        "dns": "svc-old.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    details = _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")

    issue = details[0]["chain_issues"][0]
    assert issue["leaf"] == fp(pki["leaf_old"])
    assert issue["leaf_role"] == "target"
    # nothing in the bundle verifies the orphan leaf's signature
    assert issue["candidate_issuers"] == []
    assert issue["candidate_paths"] == [[fp(pki["leaf_old"])]]
    # each entry keeps its own historical message wording
    assert "from a leaf" in details[0]["message"]
    assert "from a target leaf" in details[1]["message"]
    assert "from a leaf" in details[2]["message"]
    assert "from a target leaf" in details[3]["message"]


def test_imposter_ca_same_dn_fails_closed_across_entries():
    """A same-DN/different-key 'CA' must not become an issuance edge."""
    root_key = _key()
    imposter = make_cert("Root CA", None, _key(), self_signed=True)
    leaf = make_leaf("svc.example.com", "Root CA", root_key)
    # the real root key's certificate is not in the bundle at all
    pems = [pem(leaf), pem(imposter)]
    roots = [fp(imposter)]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    details = _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")
    # normalize reports both leaf roles; precheck/revocation only 'target' —
    # every entry agrees on the shared target-leaf issue
    target_issues = [
        i for i in details[0]["chain_issues"] if i["leaf_role"] == "target"
    ]
    assert len(target_issues) == 1
    assert target_issues[0]["candidate_issuers"] == []


# ---------------------------------------------------------------------------
# Cross-signing: ambiguous_chain from every entry
# ---------------------------------------------------------------------------


def _cross_signed_bundle():
    """One intermediate key wrapped by two distinct trusted roots."""
    root_a_key, root_b_key = _key(), _key()
    root_a = make_cert("Root A", None, root_a_key, self_signed=True)
    root_b = make_cert("Root B", None, root_b_key, self_signed=True)
    inter_key = _key()
    inter_a = make_cert("Shared Inter", "Root A", root_a_key,
                        subject_key=inter_key)
    inter_b = make_cert("Shared Inter", "Root B", root_b_key,
                        subject_key=inter_key)
    leaf = make_leaf("svc.example.com", "Shared Inter", inter_key)
    return leaf, inter_a, inter_b, root_a, root_b


def test_ambiguous_chain_agrees_across_entries(pki):
    leaf, inter_a, inter_b, root_a, root_b = _cross_signed_bundle()
    pems = [pem(c) for c in (leaf, inter_a, inter_b, root_a, root_b)]
    roots = [fp(root_a), fp(root_b)]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    details = _assert_same_chain_failure(norm, pre, pol, rev,
                                         "ambiguous_chain")
    issue = [i for i in details[0]["chain_issues"]
             if i["leaf_role"] == "target"][0]
    assert set(issue["candidate_issuers"]) == {fp(inter_a), fp(inter_b)}
    path_tuples = {tuple(p) for p in issue["candidate_paths"]}
    assert (fp(leaf), fp(inter_a), fp(root_a)) in path_tuples
    assert (fp(leaf), fp(inter_b), fp(root_b)) in path_tuples


def test_single_trusted_root_disambiguates_across_entries(pki):
    leaf, inter_a, inter_b, root_a, root_b = _cross_signed_bundle()
    pems = [pem(c) for c in (leaf, inter_a, inter_b, root_a, root_b)]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, [fp(root_a)], eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text
    chain = [fp(leaf), fp(inter_a), fp(root_a)]
    assert norm.json()["endpoints"][0]["target_chain"] == chain
    assert pre.json()["results"][0]["chain"] == chain
    assert pol.json()["results"][0]["target_chain"]["chain"] == chain
    assert rev.json()["results"][0]["chain"] == chain


# ---------------------------------------------------------------------------
# Cycles in the issuance graph
# ---------------------------------------------------------------------------


def _cycle_bundle(with_exit):
    leaf_key = _key()
    leaf = make_cert("svc", "A", leaf_key, is_ca=False)
    key_b = _key()
    cert_a = make_cert("A", "B", key_b, subject_key=leaf_key)
    cert_b = make_cert("B", "A", leaf_key, subject_key=key_b)  # A <-> B cycle
    root_key = _key()
    root = make_cert("Root", None, root_key, self_signed=True)
    certs = [leaf, cert_a, cert_b, root]
    chain = None
    if with_exit:
        cert_b2 = make_cert("B", "Root", root_key, subject_key=key_b)
        certs.append(cert_b2)
        chain = [fp(leaf), fp(cert_a), fp(cert_b2), fp(root)]
    return leaf, root, certs, chain


def test_cycle_with_single_exit_is_unique_across_entries():
    leaf, root, certs, chain = _cycle_bundle(with_exit=True)
    eps = [{
        "endpoint_id": "svc",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc",
    }]
    norm, pre, pol, rev = _post_all([pem(c) for c in certs], [fp(root)], eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text
    assert norm.json()["endpoints"][0]["target_chain"] == chain
    assert pre.json()["results"][0]["chain"] == chain
    assert pol.json()["results"][0]["target_chain"]["chain"] == chain
    assert rev.json()["results"][0]["chain"] == chain


def test_cycle_without_exit_has_no_trust_path_across_entries():
    leaf, root, certs, _chain = _cycle_bundle(with_exit=False)
    eps = [{
        "endpoint_id": "svc",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc",
    }]
    norm, pre, pol, rev = _post_all([pem(c) for c in certs], [fp(root)], eps)
    _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")


# ---------------------------------------------------------------------------
# RSA-PSS signed chains
# ---------------------------------------------------------------------------


def test_rsa_pss_chain_resolves_across_entries():
    root_key = _key("rsa")
    root = make_cert("PSS Root", None, root_key, self_signed=True,
                     rsa_pss=True)
    inter_key = _key("rsa")
    inter = make_cert("PSS Inter", "PSS Root", root_key,
                      subject_key=inter_key, rsa_pss=True)
    leaf = make_leaf("svc.example.com", "PSS Inter", inter_key)
    leaf_key_pem = [pem(leaf), pem(inter), pem(root)]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(leaf_key_pem, [fp(root)], eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text
    chain = [fp(leaf), fp(inter), fp(root)]
    assert norm.json()["endpoints"][0]["target_chain"] == chain
    assert pre.json()["results"][0]["chain"] == chain
    assert pol.json()["results"][0]["target_chain"]["chain"] == chain
    assert rev.json()["results"][0]["chain"] == chain


def test_rsa_pss_tampered_leaf_fails_closed_across_entries():
    root_key = _key("rsa")
    root = make_cert("PSS Root", None, root_key, self_signed=True,
                     rsa_pss=True)
    leaf = make_leaf("svc.example.com", "PSS Root", root_key, rsa_pss=True)
    der = bytearray(leaf.public_bytes(serialization.Encoding.DER))
    der[-1] ^= 0x01  # flip one signature bit
    tampered = x509.load_der_x509_certificate(bytes(der))
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(tampered),
        "target": fp(tampered),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(
        [pem(tampered), pem(root)], [fp(root)], eps)
    _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")


# ---------------------------------------------------------------------------
# Tampered certificates fail closed everywhere
# ---------------------------------------------------------------------------


def test_tampered_certificate_fails_closed_across_entries(pki):
    leaf = pki["leaf_new"]
    der = bytearray(leaf.public_bytes(serialization.Encoding.DER))
    der[-1] ^= 0x01
    tampered = x509.load_der_x509_certificate(bytes(der))
    pems = _bundle(pki, "leaf_old", "root_old", "inter_new", "root_new")
    pems.append(pem(tampered))
    roots = [fp(pki["root_old"]), fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": fp(tampered),
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")


# ---------------------------------------------------------------------------
# The 64-entry diagnostic cap never drives the verdict — in any entry
# ---------------------------------------------------------------------------


def _regenerate_until(make, predicate):
    """Rebuild a cert (random serial => random fingerprint) until predicate
    holds for its fingerprint."""
    while True:
        cert = make()
        if predicate(fp(cert)):
            return cert


def _dead_end_fanout(leaf_key, count=70):
    """One issuer of the leaf whose own issuer key is wrapped by ``count``
    cross-certified certs that lead nowhere (more than the 64-path cap)."""
    k0 = _key()
    dead = make_cert("Dead Branch", "K0 Wrappers", k0, subject_key=leaf_key)
    wrappers = [
        make_cert(f"K0 Wrapper {i}", f"Throwaway {i}", _key(), subject_key=k0)
        for i in range(count)
    ]
    return dead, wrappers


def _fanout_eps(leaf):
    return [{
        "endpoint_id": "svc",
        "current": fp(leaf),
        "target": fp(leaf),
        "dns": "svc",
    }]


def test_valid_path_beyond_diagnostic_cap_all_entries():
    leaf_key = _key()
    leaf = make_cert("svc", "Leaf Issuer", leaf_key, is_ca=False)
    dead, wrappers = _dead_end_fanout(leaf_key)
    root_key = _key()
    root = make_cert("Root", None, root_key, self_signed=True)
    valid = _regenerate_until(
        lambda: make_cert("Valid Issuer", "Root", root_key,
                          subject_key=leaf_key),
        lambda f: f > fp(dead),
    )
    pems = [pem(c) for c in [leaf, dead, *wrappers, valid, root]]
    norm, pre, pol, rev = _post_all(pems, [fp(root)], _fanout_eps(leaf))
    for r in (norm, pre, pol, rev):
        assert r.status_code == 200, r.text
    chain = [fp(leaf), fp(valid), fp(root)]
    assert norm.json()["endpoints"][0]["target_chain"] == chain
    assert pre.json()["results"][0]["chain"] == chain
    assert pol.json()["results"][0]["target_chain"]["chain"] == chain
    assert rev.json()["results"][0]["chain"] == chain


def test_ambiguity_beyond_diagnostic_cap_all_entries():
    leaf_key = _key()
    leaf = make_cert("svc", "Leaf Issuer", leaf_key, is_ca=False)
    dead, wrappers = _dead_end_fanout(leaf_key)
    root_a_key, root_b_key = _key(), _key()
    root_a = make_cert("Root A", None, root_a_key, self_signed=True)
    root_b = make_cert("Root B", None, root_b_key, self_signed=True)
    valid_a = _regenerate_until(
        lambda: make_cert("Valid A", "Root A", root_a_key,
                          subject_key=leaf_key),
        lambda f: f < fp(dead),
    )
    valid_b = _regenerate_until(
        lambda: make_cert("Valid B", "Root B", root_b_key,
                          subject_key=leaf_key),
        lambda f: f > fp(dead),
    )
    pems = [pem(c) for c in [leaf, dead, *wrappers,
                             valid_a, valid_b, root_a, root_b]]
    norm, pre, pol, rev = _post_all(
        pems, [fp(root_a), fp(root_b)], _fanout_eps(leaf))
    details = _assert_same_chain_failure(norm, pre, pol, rev,
                                         "ambiguous_chain")
    issue = [i for i in details[0]["chain_issues"]
             if i["leaf_role"] == "target"][0]
    assert set(issue["candidate_issuers"]) == {
        fp(dead), fp(valid_a), fp(valid_b)
    }


def test_no_trust_path_diagnostic_sample_capped_all_entries():
    leaf_key = _key()
    leaf = make_cert("svc", "Leaf Issuer", leaf_key, is_ca=False)
    dead, wrappers = _dead_end_fanout(leaf_key)
    root_key = _key()
    root = make_cert("Root", None, root_key, self_signed=True)
    pems = [pem(c) for c in [leaf, dead, *wrappers, root]]
    norm, pre, pol, rev = _post_all(pems, [fp(root)], _fanout_eps(leaf))
    details = _assert_same_chain_failure(norm, pre, pol, rev, "no_trust_path")
    for detail in details:
        for issue in detail["chain_issues"]:
            assert 0 < len(issue["candidate_paths"]) <= 64


# ---------------------------------------------------------------------------
# Request-level errors agree across entries
# ---------------------------------------------------------------------------


def test_trust_root_not_self_signed_agrees_across_entries(pki):
    pems = _bundle(pki, "leaf_old", "root_old", "leaf_new",
                   "inter_new", "root_new")
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": fp(pki["leaf_new"]),
        "dns": "svc.example.com",
    }]
    # an intermediate is declared as a trust root
    norm, pre, pol, rev = _post_all(pems, [fp(pki["inter_new"])], eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["error"] == "trust_root_not_self_signed"
        assert [i["fingerprint"] for i in detail["fingerprint_issues"]] == [
            fp(pki["inter_new"])
        ]


def test_invalid_pem_agrees_across_entries(pki):
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": "a" * 64,
        "target": "b" * 64,
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(
        ["not a certificate at all"], ["c" * 64], eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["error"] == "invalid_pem"
        assert detail["pem_issues"][0]["index"] == 0


def test_unknown_fingerprint_agrees_across_entries(pki):
    missing = "11" * 32
    pems = _bundle(pki, "leaf_old", "root_old", "leaf_new",
                   "inter_new", "root_new")
    roots = [fp(pki["root_old"]), fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": missing,
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
        detail = r.json()["detail"]
        assert detail["error"] == "unknown_fingerprint"
        assert missing in [i["fingerprint"]
                           for i in detail["fingerprint_issues"]]


def test_invalid_fingerprint_agrees_across_entries(pki):
    pems = _bundle(pki, "leaf_old", "root_old", "leaf_new",
                   "inter_new", "root_new")
    roots = [fp(pki["root_old"]), fp(pki["root_new"])]
    eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": "ZZ" + "0" * 62,
        "dns": "svc.example.com",
    }]
    norm, pre, pol, rev = _post_all(pems, roots, eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["error"] == "invalid_fingerprint"


# ---------------------------------------------------------------------------
# No shared state between requests
# ---------------------------------------------------------------------------


def test_requests_do_not_leak_state(pki):
    good_pems = _bundle(pki, "leaf_old", "root_old", "leaf_new",
                        "inter_new", "root_new")
    good_roots = [fp(pki["root_old"]), fp(pki["root_new"])]
    good_eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": fp(pki["leaf_new"]),
        "dns": "svc.example.com",
    }]
    # broken bundle: the target leaf's issuer is missing
    bad_pems = _bundle(pki, "leaf_old", "root_old", "leaf_new")
    bad_roots = [fp(pki["root_old"])]
    bad_eps = [{
        "endpoint_id": "svc.example.com",
        "current": fp(pki["leaf_old"]),
        "target": fp(pki["leaf_new"]),
        "dns": "svc.example.com",
    }]

    first = _post_all(good_pems, good_roots, good_eps)
    for r in first:
        assert r.status_code == 200, r.text

    # interleave broken and good requests across the four entries
    norm, pre, pol, rev = _post_all(bad_pems, bad_roots, bad_eps)
    for r in (norm, pre, pol, rev):
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["error"] == "no_trust_path"

    # the exact same good request again: byte-identical responses
    second = _post_all(good_pems, good_roots, good_eps)
    for a, b in zip(first, second):
        assert b.status_code == 200, b.text
        assert a.json() == b.json()


def test_context_instances_are_independent(pki):
    """Two contexts over the same bag share nothing mutable."""
    pems = _bundle(pki, "leaf_new", "inter_new", "root_new")
    ctx1 = ChainVerificationContext(pems)
    ctx2 = ChainVerificationContext(pems)
    assert ctx1.edges is ctx1.edges  # cached per instance
    assert ctx1.edges is not ctx2.edges
    ctx1.certs.pop(fp(pki["root_new"]))
    assert fp(pki["root_new"]) in ctx2.certs

    leaves = [LeafReference("svc", "target", fp(pki["leaf_new"]))]
    roots = [fp(pki["root_new"])]
    chain = ctx2.resolve_unique_chains(
        leaves, roots, no_path_message="no path"
    )
    assert chain[fp(pki["leaf_new"])] == [
        fp(pki["leaf_new"]), fp(pki["inter_new"]), fp(pki["root_new"])
    ]
    # resolving again on the same context is deterministic
    assert ctx2.resolve_unique_chains(
        leaves, roots, no_path_message="no path"
    ) == chain
