"""Outbound credential detection.

Environment is injected rather than read from the process, so these assert the
documented precedence rather than whatever the developer happens to have
exported.
"""

from __future__ import annotations

from pathlib import Path

from morpheus.providers.credentials import (
    CredentialSource,
    detect_anthropic,
    detect_openai,
)

EMPTY: dict[str, str] = {}

WIF_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "rule",
    "ANTHROPIC_ORGANIZATION_ID": "org",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa",
    "ANTHROPIC_IDENTITY_TOKEN_FILE": "/var/run/token",
}


def anthropic(env=None, *, api_key=None, profile=None, config_dir=None):
    return detect_anthropic(
        configured_api_key=api_key,
        configured_profile=profile,
        env=env if env is not None else EMPTY,
        # A path that cannot exist, so the default-profile probe never picks up
        # the developer's real ~/.config/anthropic.
        config_dir=config_dir if config_dir is not None else Path("/nonexistent"),
    )


# --- Anthropic precedence -------------------------------------------------


def test_an_explicit_key_wins():
    info = anthropic({"ANTHROPIC_API_KEY": "env"}, api_key="explicit")
    assert info.source is CredentialSource.API_KEY
    assert info.note == "explicitly configured"


def test_env_key_beats_auth_token():
    env = {"ANTHROPIC_API_KEY": "k", "ANTHROPIC_AUTH_TOKEN": "t"}
    assert anthropic(env).source is CredentialSource.API_KEY


def test_auth_token_beats_a_profile():
    env = {"ANTHROPIC_AUTH_TOKEN": "t", "ANTHROPIC_PROFILE": "work"}
    assert anthropic(env).source is CredentialSource.AUTH_TOKEN


def test_a_profile_is_detected_from_env_or_config():
    from_env = anthropic({"ANTHROPIC_PROFILE": "work"})
    assert from_env.source is CredentialSource.OAUTH_PROFILE
    assert "work" in from_env.note

    configured = anthropic(EMPTY, profile="explicit-profile")
    assert configured.source is CredentialSource.OAUTH_PROFILE
    assert "explicit-profile" in configured.note


def test_a_profile_beats_workload_identity():
    assert anthropic({**WIF_ENV, "ANTHROPIC_PROFILE": "work"}).source is (
        CredentialSource.OAUTH_PROFILE
    )


def test_workload_identity_needs_the_full_set_plus_a_token():
    assert anthropic(WIF_ENV).source is CredentialSource.WORKLOAD_IDENTITY

    # An inline token substitutes for the token file.
    inline = dict(WIF_ENV)
    del inline["ANTHROPIC_IDENTITY_TOKEN_FILE"]
    inline["ANTHROPIC_IDENTITY_TOKEN"] = "jwt"
    assert anthropic(inline).source is CredentialSource.WORKLOAD_IDENTITY

    # A partial set is not a credential.
    partial = dict(WIF_ENV)
    del partial["ANTHROPIC_SERVICE_ACCOUNT_ID"]
    assert anthropic(partial).source is CredentialSource.UNDETECTED

    no_token = dict(WIF_ENV)
    del no_token["ANTHROPIC_IDENTITY_TOKEN_FILE"]
    assert anthropic(no_token).source is CredentialSource.UNDETECTED


def test_a_default_profile_directory_counts(tmp_path):
    (tmp_path / "profile.json").write_text("{}", encoding="utf-8")
    info = anthropic(EMPTY, config_dir=tmp_path)
    assert info.source is CredentialSource.OAUTH_PROFILE
    assert str(tmp_path) in info.note


def test_an_empty_profile_directory_does_not_count(tmp_path):
    assert anthropic(EMPTY, config_dir=tmp_path).source is CredentialSource.UNDETECTED


def test_nothing_configured_is_undetected_with_actionable_advice():
    info = anthropic(EMPTY)
    assert info.source is CredentialSource.UNDETECTED
    assert info.detected is False
    assert "ant auth login" in info.note


def test_an_empty_string_key_is_not_treated_as_a_credential():
    """An exported-but-empty variable is the classic footgun: it must not be
    reported as a credential, and must not shadow the rest of the chain."""
    info = anthropic({"ANTHROPIC_API_KEY": "", "ANTHROPIC_PROFILE": "work"})
    assert info.source is CredentialSource.OAUTH_PROFILE


# --- OpenAI ---------------------------------------------------------------


def test_openai_prefers_an_explicit_key():
    info = detect_openai(configured_api_key="explicit", env={"OPENAI_API_KEY": "env"})
    assert info.source is CredentialSource.API_KEY
    assert info.note == "explicitly configured"


def test_openai_reads_its_env_key():
    assert detect_openai(configured_api_key=None, env={"OPENAI_API_KEY": "k"}).source is (
        CredentialSource.API_KEY
    )


def test_openai_detects_workload_identity():
    env = {"OPENAI_WORKLOAD_IDENTITY_TOKEN_FILE": "/var/run/token"}
    assert detect_openai(configured_api_key=None, env=env).source is (
        CredentialSource.WORKLOAD_IDENTITY
    )


def test_openai_undetected_is_actionable():
    info = detect_openai(configured_api_key=None, env=EMPTY)
    assert info.source is CredentialSource.UNDETECTED
    assert "OPENAI_API_KEY" in info.note
