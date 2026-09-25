"""Principal store, key handling, and fail-closed startup."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from morpheus.api.app import ConfigurationError, create_app
from morpheus.auth.principals import (
    KEY_PREFIX,
    Principal,
    PrincipalStore,
    hash_key,
    mint_key,
)
from morpheus.config import Settings
from morpheus.errors import InvalidCredentialError

from .conftest import ALICE_KEY, DISABLED_KEY, SONNET_ONLY_KEY

# --- key handling ---------------------------------------------------------


def test_minted_keys_are_prefixed_unique_and_high_entropy():
    keys = {mint_key() for _ in range(200)}
    assert len(keys) == 200
    for key in keys:
        assert key.startswith(KEY_PREFIX)
        assert len(key) > 40


def test_hashing_is_stable_and_does_not_embed_the_key():
    digest = hash_key("mh_example")
    assert digest == hash_key("mh_example")
    assert len(digest) == 64
    assert "mh_example" not in digest


def test_a_principal_does_not_leak_its_hash_in_its_repr():
    """Principals are logged and put in error context; the digest stays out."""
    principal = Principal(id="alice", key_sha256=hash_key("mh_secret"))
    assert hash_key("mh_secret") not in repr(principal)


# --- store ----------------------------------------------------------------


def test_authenticate_resolves_a_known_key(principal_store):
    assert principal_store.authenticate(ALICE_KEY).id == "alice"
    assert principal_store.authenticate(SONNET_ONLY_KEY).id == "sonnet-only"


def test_authenticate_rejects_an_unknown_key(principal_store):
    with pytest.raises(InvalidCredentialError):
        principal_store.authenticate("mh_not_a_real_key")


def test_a_disabled_principal_is_rejected_indistinguishably(principal_store):
    """Same status, code and message as an unknown key — no enumeration oracle."""
    with pytest.raises(InvalidCredentialError) as disabled:
        principal_store.authenticate(DISABLED_KEY)
    with pytest.raises(InvalidCredentialError) as unknown:
        principal_store.authenticate("mh_nope")

    assert (
        disabled.value.to_payload()["error"]["message"]
        == (unknown.value.to_payload()["error"]["message"])
    )


def test_duplicate_ids_and_shared_hashes_are_rejected():
    with pytest.raises(ValueError, match="Duplicate principal id"):
        PrincipalStore(
            [
                Principal(id="a", key_sha256=hash_key("k1")),
                Principal(id="a", key_sha256=hash_key("k2")),
            ]
        )
    with pytest.raises(ValueError, match="share a key hash"):
        PrincipalStore(
            [
                Principal(id="a", key_sha256=hash_key("same")),
                Principal(id="b", key_sha256=hash_key("same")),
            ]
        )


def test_a_principal_without_a_key_hash_is_rejected():
    with pytest.raises(ValueError, match="no key_sha256"):
        PrincipalStore([Principal(id="a", key_sha256="")])


# --- file loading ---------------------------------------------------------


def write_principals(tmp_path, payload):
    path = tmp_path / "principals.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_a_well_formed_file(tmp_path):
    path = write_principals(
        tmp_path,
        {
            "principals": [
                {
                    "id": "team-a",
                    "key_sha256": hash_key("mh_a"),
                    "allowed_models": ["gpt-5"],
                    "max_tokens_per_turn": 1000,
                },
                {"id": "team-b", "key_sha256": hash_key("mh_b")},
            ]
        },
    )
    store = PrincipalStore.from_file(path)

    assert store.ids == ["team-a", "team-b"]
    team_a = store.authenticate("mh_a")
    assert team_a.allowed_models == frozenset({"gpt-5"})
    assert team_a.max_tokens_per_turn == 1000
    # No allowlist means every model.
    assert store.authenticate("mh_b").allowed_models is None


def test_a_bare_list_is_accepted_too(tmp_path):
    path = write_principals(tmp_path, [{"id": "solo", "key_sha256": hash_key("mh_s")}])
    assert PrincipalStore.from_file(path).ids == ["solo"]


def test_a_raw_key_in_the_file_is_refused(tmp_path):
    """Guards the likeliest operator mistake: pasting the secret itself."""
    path = write_principals(
        tmp_path,
        [{"id": "oops", "key_sha256": hash_key("mh_x"), "key": "mh_x"}],
    )
    with pytest.raises(ValueError, match="raw 'key' field"):
        PrincipalStore.from_file(path)


def test_malformed_files_are_reported_clearly(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="not found"):
        PrincipalStore.from_file(missing)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        PrincipalStore.from_file(bad_json)

    wrong_shape = write_principals(tmp_path, {"principals": "nope"})
    with pytest.raises(ValueError, match="must hold a list"):
        PrincipalStore.from_file(wrong_shape)

    incomplete = write_principals(tmp_path, [{"id": "x"}])
    with pytest.raises(ValueError, match="missing key_sha256"):
        PrincipalStore.from_file(incomplete)


# --- fail-closed startup --------------------------------------------------


def base_settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_startup_refuses_with_no_auth_configuration():
    """The central safety property: forgetting to configure auth does not
    silently produce an open proxy for the provider credentials."""
    with pytest.raises(ConfigurationError, match="Inbound authentication is not configured"):
        create_app(base_settings())


def test_startup_allows_anonymous_only_when_asked(tmp_path):
    app = create_app(base_settings(HARNESS_ALLOW_ANONYMOUS=True))
    assert app.state.principals is None


def test_startup_refuses_both_auth_modes_at_once(tmp_path):
    path = write_principals(tmp_path, [{"id": "a", "key_sha256": hash_key("mh_a")}])
    with pytest.raises(ConfigurationError, match="Pick one"):
        create_app(base_settings(HARNESS_PRINCIPALS_FILE=path, HARNESS_ALLOW_ANONYMOUS=True))


def test_startup_refuses_an_empty_principals_file(tmp_path):
    path = write_principals(tmp_path, {"principals": []})
    with pytest.raises(ConfigurationError, match="defines no principals"):
        create_app(base_settings(HARNESS_PRINCIPALS_FILE=path))


def test_startup_reports_a_malformed_principals_file_as_configuration_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        create_app(base_settings(HARNESS_PRINCIPALS_FILE=path))


def test_an_authenticated_app_serves_health_without_a_key(tmp_path):
    path = write_principals(tmp_path, [{"id": "a", "key_sha256": hash_key("mh_a")}])
    app = create_app(base_settings(HARNESS_PRINCIPALS_FILE=path, HARNESS_ALLOW_ANONYMOUS=False))
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
