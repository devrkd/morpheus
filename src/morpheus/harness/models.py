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
from ..core.registry import ModelSpec, Provider, ThinkingStyle
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

    def credential_for(self, spec: ModelSpec) -> CredentialInfo:
        """What was detected for this model's provider.

        Used by error translation, which reinterprets a bare TypeError as a
        configuration problem only when nothing was found.
        """
        return self.credentials[spec.provider.value]

    # --- construction -----------------------------------------------------

    def build(
        self,
        spec: ModelSpec,
        *,
        max_tokens: int | None = None,
        sampling: dict[str, Any] | None = None,
        stop_sequences: list[str] | None = None,
    ) -> Model:
        """Build a Strands model for one catalog entry.

        Provider config is passed as **flat keyword arguments** — the
        constructors take ``**model_config: Unpack[...Config]``, so nesting
        them in a ``model_config=`` dict silently lands in an unknown key and
        the model then raises ``KeyError: 'model_id'`` on first use. That was a
        real 500 in this service; the tests below construct every catalog
        entry for exactly that reason.
        """
        if spec.provider is Provider.ANTHROPIC:
            return self._anthropic(spec, max_tokens, sampling, stop_sequences)
        if spec.provider is Provider.OPENAI:
            return self._openai(spec, max_tokens, sampling, stop_sequences)
        raise ProviderUnavailableError(
            f"No Strands provider wired for '{spec.provider}'",
            provider=spec.provider.value,
        )

    def _client_args(self, provider: Provider) -> dict[str, Any]:
        """Only explicitly configured values, so each SDK's own credential
        chain still applies. An empty string would shadow all of it."""
        args: dict[str, Any] = {"timeout": self._settings.request_timeout_seconds}
        if provider is Provider.ANTHROPIC:
            if self._settings.anthropic_api_key:
                args["api_key"] = self._settings.anthropic_api_key
            if self._settings.anthropic_base_url:
                args["base_url"] = self._settings.anthropic_base_url
        else:
            if self._settings.openai_api_key:
                args["api_key"] = self._settings.openai_api_key
            if self._settings.openai_organization:
                args["organization"] = self._settings.openai_organization
            if self._settings.openai_project:
                args["project"] = self._settings.openai_project
            if self._settings.openai_base_url:
                args["base_url"] = self._settings.openai_base_url
        return args

    def _anthropic(
        self,
        spec: ModelSpec,
        max_tokens: int | None,
        sampling: dict[str, Any] | None,
        stop_sequences: list[str] | None,
    ) -> Model:
        params: dict[str, Any] = dict(sampling or {})
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        config: dict[str, Any] = {
            "model_id": spec.native_id,
            "max_tokens": min(
                max_tokens or self._settings.default_max_tokens, spec.max_output_tokens
            ),
        }
        if params:
            config["params"] = params

        return AnthropicModel(client_args=self._client_args(Provider.ANTHROPIC), **config)

    def _openai(
        self,
        spec: ModelSpec,
        max_tokens: int | None,
        sampling: dict[str, Any] | None,
        stop_sequences: list[str] | None,
    ) -> Model:
        if not self.credentials[Provider.OPENAI.value].detected:
            raise ProviderUnavailableError(
                "No OpenAI credential is configured, so OpenAI models cannot be "
                "served. Set OPENAI_API_KEY or configure workload identity.",
                provider=Provider.OPENAI.value,
            )

        params: dict[str, Any] = dict(sampling or {})
        if stop_sequences:
            params["stop"] = stop_sequences
        if max_tokens:
            # Reasoning models count reasoning against the output ceiling and
            # take `max_completion_tokens`; the older models take `max_tokens`.
            key = (
                "max_completion_tokens"
                if spec.thinking is ThinkingStyle.REASONING_EFFORT
                else "max_tokens"
            )
            params[key] = min(max_tokens, spec.max_output_tokens)

        config: dict[str, Any] = {"model_id": spec.native_id}
        if params:
            config["params"] = params

        return OpenAIModel(client_args=self._client_args(Provider.OPENAI), **config)

    def build_with_fallbacks(
        self, spec: ModelSpec, fallbacks: list[ModelSpec], **kwargs: Any
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
            return self.build(spec, **kwargs), []

        models = [self.build(spec, **kwargs), *(self.build(f, **kwargs) for f in usable)]
        notes = ["failover chain: " + " -> ".join([spec.id, *(f.id for f in usable)])]
        return ModelRouter(models=models), notes

    def _usable(self, spec: ModelSpec) -> bool:
        status = self.status().get(spec.provider.value, {})
        return bool(status.get("available"))
