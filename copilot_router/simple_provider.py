"""SimpleFailoverProvider — default model with ordered provider fallback."""

import time
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


class SimpleFailoverProvider(LLMProvider):
    """Tries the primary provider, then each fallback in order.

    Supports ``/use`` overrides via ``session_metadata["force_provider"]`` and
    ``session_metadata["force_model"]``.  No circuit breakers, no routing plans,
    no escalation, no preference learning.
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: dict[str, LLMProvider],
        cost_logger=None,
        *,
        primary_name: str = "primary",
        provider_models: dict[str, str] | None = None,
    ):
        super().__init__()
        self.primary = primary
        self.fallbacks = fallbacks          # name → LiteLLMProvider
        self.cost_logger = cost_logger
        self.primary_name = primary_name
        self.provider_models = provider_models or {}  # name → configured default model
        self.last_decision: str | None = None   # which provider served last call
        self.last_model_used: str | None = None  # actual model that served last call

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        session_metadata: dict | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Try primary, then each fallback in order.  Respects /use overrides."""
        meta = session_metadata or {}
        force_provider = meta.get("force_provider")
        force_model = meta.get("force_model")

        # Build ordered attempt list
        attempts: list[tuple[str, LLMProvider, str | None]] = []

        if force_provider and force_provider in self.fallbacks:
            # /use override: try forced provider first, then normal chain
            attempts.append((force_provider, self.fallbacks[force_provider], force_model))
        elif force_provider == self.primary_name:
            attempts.append((self.primary_name, self.primary, force_model))

        # Always include full chain as fallback
        attempts.append((self.primary_name, self.primary, model))
        for name, provider in self.fallbacks.items():
            attempts.append((name, provider, model))

        errors: list[tuple[str, str]] = []  # (provider_name, error_message)
        t0 = time.monotonic()
        for provider_name, provider, use_model in attempts:
            try:
                response = await provider.chat(
                    messages=messages,
                    tools=tools,
                    model=use_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                # LiteLLMProvider swallows exceptions → finish_reason="error".
                # Treat as failure and try next provider.
                if response.finish_reason == "error":
                    err_msg = (response.content or "Unknown error")[:200]
                    errors.append((provider_name, err_msg))
                    elapsed_ms = int((time.monotonic() - t0) * 1000)
                    logger.warning(f"Provider {provider_name} returned error ({elapsed_ms}ms): {err_msg}")
                    if self.cost_logger:
                        try:
                            await self.cost_logger.log_route(
                                input_length=sum(len(str(m.get("content", ""))) for m in messages),
                                has_images=False,
                                routed_to=provider_name, provider=provider_name,
                                model_used=use_model or provider.get_default_model(),
                                route_reason="failover_attempt", success=False,
                                latency_ms=elapsed_ms, failure_reason=err_msg,
                            )
                        except Exception:
                            pass
                    continue  # try next provider

                elapsed_ms = int((time.monotonic() - t0) * 1000)
                self.last_decision = provider_name
                self.last_model_used = response.model_used or use_model or provider.get_default_model()

                if self.cost_logger:
                    try:
                        await self.cost_logger.log_route(
                            input_length=sum(len(str(m.get("content", ""))) for m in messages),
                            has_images=False,
                            routed_to=provider_name, provider=provider_name,
                            model_used=self.last_model_used,
                            route_reason="failover" if errors else "primary",
                            success=True, latency_ms=elapsed_ms,
                        )
                    except Exception:
                        pass

                return response

            except Exception as e:
                err_msg = str(e)[:200]
                errors.append((provider_name, err_msg))
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                logger.warning(f"Provider {provider_name} failed ({elapsed_ms}ms): {e}")
                if self.cost_logger:
                    try:
                        await self.cost_logger.log_route(
                            input_length=sum(len(str(m.get("content", ""))) for m in messages),
                             has_images=False,
                            routed_to=provider_name, provider=provider_name,
                            model_used=use_model or provider.get_default_model(),
                            route_reason="failover_attempt", success=False,
                            latency_ms=elapsed_ms, failure_reason=err_msg,
                        )
                    except Exception:
                        pass

        # All providers failed — return diagnostic response (don't raise,
        # so the agent loop can display the message to the user).
        diagnostics = [f"  {name}: {err}" for name, err in errors]
        return LLMResponse(
            content=f"All {len(errors)} providers failed:\n" + "\n".join(diagnostics)
                    + "\n\nCheck /status for provider health.",
            finish_reason="error",
            model_used="none",
        )

    def check_use_override_timeout(
        self, metadata: dict, timeout_s: int
    ) -> str | None:
        """Check if /use override has expired due to inactivity."""
        since = metadata.get("force_provider_since", 0)
        last_activity = metadata.get("last_user_message_at", 0)
        if not since:
            return None
        idle = time.time() - last_activity
        if idle >= timeout_s:
            return "expired"
        if idle >= timeout_s - 120:
            return "warning"
        return None

    def get_default_model(self) -> str:
        return self.primary.get_default_model()
