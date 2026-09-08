"""Layer 3: model dispatch, delegated to Strands.

Our hand-written Anthropic and OpenAI adapters are replaced by
``strands.models``, which ships providers for both plus a ``ModelRouter`` that
does the cross-format failover we had planned to build ourselves.

What stays ours is the *catalog*: which model ids we accept, which aliases map
where, and what a principal is allowed to ask for. Strands will happily talk to
any model string; the allowlist and the alias table are policy, and policy
belongs on our side of the boundary.

Credential resolution also stays ours, because it feeds ``GET /v1/health``.
Strands takes a provider client's kwargs verbatim, so an explicitly configured
key is passed through and anything omitted still falls through to the SDK's own
chain — the behaviour we fixed earlier survives the migration.
"""

from __future__ import annotations

from typing import Any

from strands.models import Model, ModelRouter
from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel

from ..config import Settings
from ..core.registry import ModelSpec, Provider
from ..errors import ProviderUnavailableError
from ..providers.credentials import CredentialInfo, detect_anthropic, detect_openai


class ModelFactory:
    """Builds a Strands ``Model`` for one of our catalog entries."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.credentials: dict[str, CredentialInfo] = {
            Provider.ANTHROPIC.value: detect_anthropic(
                configured_api_key=settings.anthropic_api_key,
                configured_profile=settings.anthropic_profile,
            ),
            Provider.OPENAI.value: detect_openai(configured_api_key=settings.openai_api_key),
        }

    # --- credential reporting (unchanged contract for /v1/health) ---------

    def status(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                # Anthropic resolves lazily, so an undetected credential is not
                # proof of failure; OpenAI's SDK raises at construction, so for
                # it detection is authoritative. Same asymmetry as before.
                "available": (True if name == Provider.ANTHROPIC.value else info.detected),
                "credential_source": info.source.value,
                "credential_detected": info.detected,
                "note": info.note,
            }
            for name, info in self.credentials.items()
        }

    # --- construction -----------------------------------------------------

    def build(self, spec: ModelSpec) -> Model:
        if spec.provider is Provider.ANTHROPIC:
            return self._anthropic(spec)
        if spec.provider is Provider.OPENAI:
            return self._openai(spec)
        raise ProviderUnavailableError(
            f"No Strands provider wired for '{spec.provider}'",
            provider=spec.provider.value,
        )

    def _anthropic(self, spec: ModelSpec) -> Model:
        client_args: dict[str, Any] = {"timeout": self._settings.request_timeout_seconds}
        # Only explicitly configured values are passed, so the SDK's credential
        # chain (env key, auth token, `ant auth login` profile, workload
        # identity) still applies. An empty string would shadow all of it.
        if self._settings.anthropic_api_key:
            client_args["api_key"] = self._settings.anthropic_api_key
        if self._settings.anthropic_base_url:
            client_args["base_url"] = self._settings.anthropic_base_url

        return AnthropicModel(
            client_args=client_args,
            model_config={
                "model_id": spec.native_id,
                "max_tokens": min(spec.max_output_tokens, 16_000),
            },
        )

    def _openai(self, spec: ModelSpec) -> Model:
        if not self.credentials[Provider.OPENAI.value].detected:
            raise ProviderUnavailableError(
                "No OpenAI credential is configured, so OpenAI models cannot be "
                "served. Set OPENAI_API_KEY or configure workload identity.",
                provider=Provider.OPENAI.value,
            )

        client_args: dict[str, Any] = {"timeout": self._settings.request_timeout_seconds}
        if self._settings.openai_api_key:
            client_args["api_key"] = self._settings.openai_api_key
        if self._settings.openai_organization:
            client_args["organization"] = self._settings.openai_organization
        if self._settings.openai_project:
            client_args["project"] = self._settings.openai_project
        if self._settings.openai_base_url:
            client_args["base_url"] = self._settings.openai_base_url

        return OpenAIModel(
            client_args=client_args,
            model_config={"model_id": spec.native_id},
        )

    def build_with_fallbacks(
        self, spec: ModelSpec, fallbacks: list[ModelSpec]
    ) -> tuple[Model, list[str]]:
        """Wrap a primary model in a router that fails over to others.

        This is the capability the research called the one thing SDKs
        structurally lack, and the reason a gateway earns its place in the
        request path. Strands' router re-enters model dispatch per attempt, so
        a fallback may sit on a *different provider* with a different wire
        format — an Anthropic overload falling through to OpenAI.
        """
        usable = [f for f in fallbacks if self._usable(f)]
        if not usable:
            return self.build(spec), []

        models = [self.build(spec), *(self.build(f) for f in usable)]
        notes = ["failover chain: " + " -> ".join([spec.id, *(f.id for f in usable)])]
        return ModelRouter(models=models), notes

    def _usable(self, spec: ModelSpec) -> bool:
        status = self.status().get(spec.provider.value, {})
        return bool(status.get("available"))
