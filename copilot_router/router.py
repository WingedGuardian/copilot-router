"""RouterProvider V2 — plan-based routing with mandatory safety net."""

import asyncio
import time
from typing import Any

from loguru import logger

from nanobot.copilot.cost.logger import CostLogger
from nanobot.copilot.routing.failover import FailoverChain, ProviderTier
from nanobot.copilot.routing.heuristics import RouteDecision
from nanobot.providers.base import LLMProvider, LLMResponse

# Instruction injected into the system prompt for models that can self-escalate.
_ESCALATION_INSTRUCTION = (
    "\n\n---\n\n## Self-Escalation\n"
    "You can escalate to a more powerful model when needed. "
    "Begin your response with exactly `[ESCALATE]` followed by a brief reason.\n\n"
    "**Escalate when:**\n"
    "- A tool call failed and you cannot determine why\n"
    "- The task requires chaining 3+ tool calls with conditional logic\n"
    "- You are about to present 'options' back to the user because you are stuck\n"
    "- The user asks for debugging, code analysis, or multi-step technical reasoning\n"
    "- You are not confident your response is correct\n\n"
    "**Do NOT escalate for:** simple questions, greetings, status checks, "
    "single-tool tasks, or anything you can handle confidently.\n"
    "Escalation is free — the user prefers a correct answer from a stronger model "
    "over a wrong answer from you."
)


class RouterProvider(LLMProvider):
    """Routes each LLM call based on a user-approved routing plan.

    V2: Heuristic classification is removed.  Routing is determined by:
      1. Private mode → local only
      2. /use override → forced provider/model
      3. Routing plan → LLM-generated, user-approved provider order
      4. Default → default_model on all cloud providers

    The mandatory safety net is ALWAYS appended:
      - Last known working provider/model
      - LM Studio local (if not already primary)
      - Emergency model on all providers
    """

    def __init__(
        self,
        local_provider: LLMProvider,
        cloud_providers: dict[str, LLMProvider],
        cost_logger: CostLogger,
        *,
        local_model: str = "huihui-qwen3-30b-a3b-instruct-2507-abliterated-i1@q4_k_m",
        fast_model: str = "anthropic/claude-haiku-4-5",
        big_model: str = "anthropic/claude-sonnet-4-6",
        default_model: str = "MiniMax-M2.5",
        escalation_model: str = "anthropic/claude-sonnet-4-6",
        strongest_model: str = "",
        emergency_cloud_model: str = "openai/gpt-4o-mini",
        escalation_enabled: bool = True,
        escalation_marker: str = "[ESCALATE]",
        provider_models: dict[str, str] | None = None,
        routing_plan: list[dict] | None = None,
        notify_on_failover: bool = True,
    ):
        super().__init__(api_key=None, api_base=None)

        self._local = local_provider
        self._cloud = cloud_providers
        self._cost_logger = cost_logger
        self._failover = FailoverChain()
        self._provider_models = provider_models or {}

        self._local_model = local_model
        self._fast_model = fast_model          # kept for backward compat (model_override)
        self._big_model = big_model            # kept for backward compat (model_override)
        self._default_model = default_model
        self._escalation_model = escalation_model
        self._strongest_model = strongest_model
        self._emergency_cloud_model = emergency_cloud_model
        self._escalation_enabled = escalation_enabled
        self._escalation_marker = escalation_marker
        self._routing_plan = routing_plan or []
        self._notify_on_failover = notify_on_failover

        self._private_mode_timeout = 1800
        self._use_override_timeout = 1800
        self._last_decision: RouteDecision | None = None
        self._last_winning_provider: str = ""
        self._last_known_working: tuple[str, str] | None = None  # (provider_name, model)

        # Recovery probing state
        self._recovery_task: asyncio.Task | None = None
        self._in_failover = False

    def get_default_model(self) -> str:
        return self._default_model

    @property
    def last_decision(self) -> RouteDecision | None:
        return self._last_decision

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        session_metadata: dict[str, Any] | None = None,
        force_route: str | None = None,
    ) -> LLMResponse:
        # --- Forced re-route (e.g. after web search consent denial) ---
        if force_route:
            model_map = {
                "local": self._local_model,
                "fast": self._default_model,
                "big": self._escalation_model,
            }
            forced_model = model_map.get(force_route, self._default_model)
            decision = RouteDecision(force_route, "consent_reroute", forced_model)
            self._last_decision = decision
            logger.info(f"Route: {force_route} (consent_reroute) → {forced_model}")

            chain = self._build_chain(decision)
            try:
                response, tier, latency_ms = await self._failover.try_providers(
                    chain=chain, messages=messages, tools=tools,
                    max_tokens=max_tokens, temperature=temperature,
                )
            except RuntimeError as e:
                logger.error(f"Forced re-route failed: {e}")
                return LLMResponse(
                    content="I'm having trouble connecting right now. Please try again.",
                    finish_reason="error", model_used="none",
                )

            await self._log_success(decision, tier, response, latency_ms, 0)
            return response

        # Extract last user message for logging
        message_text = ""
        has_images = False
        token_estimate = 0

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    message_text = content
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                has_images = True
                    message_text = " ".join(parts)
                break

        for msg in messages:
            c = msg.get("content", "")
            if isinstance(c, str):
                token_estimate += len(c) // 4
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        token_estimate += len(part.get("text", "")) // 4

        # --- Routing decision ---
        meta = session_metadata or {}
        is_private = meta.get("private_mode", False)
        force_provider = meta.get("force_provider")

        if force_provider and force_provider in self._cloud:
            force_model = meta.get("force_model")
            if force_model:
                model_used = force_model
            elif force_provider in self._provider_models:
                model_used = self._provider_models[force_provider]
            else:
                force_tier = meta.get("force_tier", "big")
                model_used = self._default_model if force_tier == "fast" else self._escalation_model
            decision = RouteDecision("cloud", f"manual:{force_provider}", model_used)
            logger.info(f"Route: {force_provider} (manual) → {model_used}")
        elif is_private:
            decision = RouteDecision("local", "private_mode", self._local_model)
            logger.info(f"Route: local (private_mode) → {self._local_model}")
        elif self._routing_plan:
            first = self._routing_plan[0]
            decision = RouteDecision("plan", "routing_plan", first.get("model", self._default_model))
            logger.info(f"Route: plan ({first.get('provider', '?')}) → {decision.model}")
        else:
            decision = RouteDecision("default", "default", self._default_model)
            logger.info(f"Route: default → {self._default_model}")

        self._last_decision = decision

        chain = self._build_chain(decision)
        logger.info(
            f"Route: {decision.target} ({decision.reason}) → "
            f"{decision.model} | tokens≈{token_estimate} images={has_images}"
        )

        # Inject escalation instruction for any non-strongest model
        call_messages = messages
        escalation_active = self._escalation_enabled and not is_private
        if decision.target in ("local", "default", "plan") and escalation_active:
            call_messages = self._inject_escalation(messages)

        # Execute with failover
        try:
            response, tier, latency_ms = await self._failover.try_providers(
                chain=chain, messages=call_messages, tools=tools,
                max_tokens=max_tokens, temperature=temperature,
            )
        except RuntimeError as e:
            logger.error(f"All providers failed: {e}")
            await self._cost_logger.log_route(
                input_length=len(message_text), has_images=has_images,
                routed_to=decision.target, provider="none",
                model_used=decision.model, route_reason=decision.reason,
                success=False, failure_reason=str(e),
            )
            return LLMResponse(
                content="I'm having trouble connecting to my language models right now. "
                "Please try again in a moment.",
                finish_reason="error", model_used="none",
            )

        # --- Self-escalation check (two-tier) ---
        if (
            escalation_active
            and decision.target in ("local", "default", "plan")
            and response.content
            and response.content.strip().startswith(self._escalation_marker)
        ):
            reason_text = response.content.strip()[len(self._escalation_marker):].strip()
            logger.info(f"Self-escalation triggered: {reason_text[:120]} → retrying with escalation model")

            # Tier-1: retry with escalation model (inject escalation for tier-2 if strongest is configured)
            esc_messages = messages
            if self._strongest_model:
                esc_messages = self._inject_escalation(messages)

            esc_chain = self._build_chain(
                RouteDecision("escalation", "escalation", self._escalation_model)
            )
            try:
                response, tier, latency_ms = await self._failover.try_providers(
                    chain=esc_chain, messages=esc_messages, tools=tools,
                    max_tokens=max_tokens, temperature=temperature,
                )
                decision = RouteDecision("escalation", "escalation", self._escalation_model)
                self._last_decision = decision

                # Tier-2: if escalation model also escalates and strongest is configured
                if (
                    self._strongest_model
                    and response.content
                    and response.content.strip().startswith(self._escalation_marker)
                ):
                    tier2_reason = response.content.strip()[len(self._escalation_marker):].strip()
                    logger.info(f"Tier-2 escalation: {tier2_reason[:120]} → retrying with strongest model")

                    strongest_chain = self._build_chain(
                        RouteDecision("strongest", "strongest", self._strongest_model)
                    )
                    try:
                        response, tier, latency_ms = await self._failover.try_providers(
                            chain=strongest_chain, messages=messages, tools=tools,
                            max_tokens=max_tokens, temperature=temperature,
                        )
                        decision = RouteDecision("strongest", "strongest", self._strongest_model)
                        self._last_decision = decision
                    except RuntimeError as e:
                        logger.error(f"Tier-2 escalation failed: {e}")
                        # Fall through with the escalation model's response minus the marker
                        response.content = tier2_reason or response.content

            except RuntimeError as e:
                logger.error(f"Escalation retry failed: {e}")
                from nanobot.copilot.alerting.bus import get_alert_bus
                await get_alert_bus().alert(
                    "routing", "high",
                    f"Escalation failed — both default and escalation models down: {e}",
                    "escalation_failed",
                )
                response.content = reason_text or response.content

        # Track last-known-working (only for plan/default tiers, not safety/emergency)
        self._last_winning_provider = tier.name
        if not tier.name.startswith(("safety:", "emergency:")):
            self._last_known_working = (tier.name, tier.model)
            if self._in_failover:
                self._in_failover = False
                self._stop_recovery_probe()
        else:
            # We're in failover mode — start recovery probe if not already running
            if not self._in_failover:
                self._in_failover = True
                self._start_recovery_probe()

        # Failover notification
        if tier.name.startswith(("safety:", "emergency:")):
            from nanobot.copilot.alerting.bus import get_alert_bus
            await get_alert_bus().alert(
                "routing", "medium",
                f"Failover: routed via {tier.name} ({tier.model})",
                "routing_failover",
            )
            if self._notify_on_failover:
                response.content = (
                    (response.content or "")
                    + f"\n\n_(Routed via {tier.name}: {tier.model} — primary providers unavailable)_"
                )

        # Log
        await self._log_success(decision, tier, response, latency_ms, len(message_text), has_images)

        return response

    def _build_chain(self, decision: RouteDecision) -> list[ProviderTier]:
        """Build the ordered failover chain for a routing decision."""
        chain: list[ProviderTier] = []

        # Manual provider override — put chosen provider first, others as fallback
        if decision.reason.startswith("manual:"):
            forced = decision.reason.split(":", 1)[1]
            if forced in self._cloud:
                chain.append(ProviderTier(forced, self._cloud[forced], decision.model))
            for name, provider in self._cloud.items():
                if name != forced:
                    chain.append(ProviderTier(name, provider, decision.model))
            return chain

        # Escalation / strongest — dedicated chain, native provider first
        # Falls through to safety net (no early return)
        if decision.target in ("escalation", "strongest"):
            target_model = (
                self._strongest_model if decision.target == "strongest"
                else self._escalation_model
            )
            from nanobot.providers.registry import find_by_model as _find_by_model
            native_spec = _find_by_model(target_model)
            native_name = native_spec.name if native_spec else None
            if not native_spec:
                logger.warning(f"Native provider lookup failed for model '{target_model}' — falling back to all cloud providers")
            if native_name and native_name in self._cloud:
                chain.append(ProviderTier(native_name, self._cloud[native_name], target_model))
            for name, provider in self._cloud.items():
                if name != native_name:
                    chain.append(ProviderTier(name, provider, target_model))
            # Fall through to safety net below (not returning early)
        elif decision.target == "local":
            # Local (private mode)
            chain.append(
                ProviderTier("lm_studio", self._local, self._local_model, is_local=True)
            )
        else:
            # Plan entries — LLM-generated, user-approved order
            if self._routing_plan:
                for entry in self._routing_plan:
                    provider = self._cloud.get(entry.get("provider", ""))
                    if provider:
                        chain.append(ProviderTier(
                            f"plan:{entry['provider']}", provider, entry.get("model", self._default_model),
                        ))
            else:
                # No plan — try native provider first, then others as fallback
                from nanobot.providers.registry import find_by_model as _find_by_model
                native_spec = _find_by_model(self._default_model)
                native_name = native_spec.name if native_spec else None
                if not native_spec:
                    logger.warning(f"Native provider lookup failed for model '{self._default_model}' — falling back to all cloud providers")
                if native_name and native_name in self._cloud:
                    chain.append(ProviderTier(native_name, self._cloud[native_name], self._default_model))
                for name, provider in self._cloud.items():
                    if name != native_name:
                        chain.append(ProviderTier(name, provider, self._default_model))

        # ── Mandatory safety net (always appended by code) ──────────
        # 1. Last known working provider/model
        if self._last_known_working:
            lkw_name, lkw_model = self._last_known_working
            # Strip plan: prefix to find the actual provider
            clean_name = lkw_name.removeprefix("plan:")
            if clean_name in self._cloud:
                chain.append(ProviderTier(
                    f"safety:{clean_name}", self._cloud[clean_name], lkw_model,
                ))

        # 2. LM Studio local (if not already primary)
        if decision.target != "local":
            chain.append(ProviderTier(
                "safety:lm_studio", self._local, self._local_model, is_local=True,
            ))

        # 3. Emergency free model on all providers
        if self._emergency_cloud_model:
            for name, provider in self._cloud.items():
                chain.append(ProviderTier(
                    f"emergency:{name}", provider, self._emergency_cloud_model,
                ))

        return chain

    async def _log_success(
        self, decision: RouteDecision, tier: ProviderTier, response: LLMResponse,
        latency_ms: int, input_length: int = 0, has_images: bool = False,
    ) -> None:
        """Log routing and cost data for a successful call."""
        tokens_in = response.usage.get("prompt_tokens", 0)
        tokens_out = response.usage.get("completion_tokens", 0)
        cost = self._cost_logger.calculate_cost(tier.model, tokens_in, tokens_out)

        self._last_winning_provider = tier.name
        await self._cost_logger.log_route(
            input_length=input_length, has_images=has_images,
            routed_to=decision.target, provider=tier.name,
            model_used=tier.model, route_reason=decision.reason,
            success=True, latency_ms=latency_ms, cost_usd=cost,
        )
        if tokens_in or tokens_out:
            await self._cost_logger.log_call(
                model=tier.model, tokens_in=tokens_in,
                tokens_out=tokens_out, cost_usd=cost,
            )

    # --- Recovery probing ---

    def _start_recovery_probe(self) -> None:
        """Start background probe loop when in failover mode."""
        if self._recovery_task and not self._recovery_task.done():
            return
        self._recovery_task = asyncio.create_task(self._recovery_probe_loop())
        logger.info("Router: started recovery probe")

    def _stop_recovery_probe(self) -> None:
        """Stop recovery probe when recovered."""
        if self._recovery_task and not self._recovery_task.done():
            self._recovery_task.cancel()
            self._recovery_task = None
            logger.info("Router: stopped recovery probe (recovered)")

    async def _recovery_probe_loop(self) -> None:
        """Probe plan entries and LM Studio every 30s until recovery."""
        import urllib.request
        while self._in_failover:
            await asyncio.sleep(30)
            if not self._in_failover:
                break

            # Probe LM Studio
            try:
                lm_base = getattr(self._local, "api_base", "http://192.168.50.100:1234")
                url = f"{lm_base.rstrip('/').rstrip('/v1')}/v1/models"
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: urllib.request.urlopen(url, timeout=5),
                    ),
                    timeout=10,
                )
                # LM Studio is back — half-open its circuit
                self._failover._breaker.record_success("lm_studio")
                self._failover._breaker.record_success("safety:lm_studio")
                logger.info("Recovery probe: LM Studio is back online")
            except Exception:
                pass

            # Probe plan entries with open circuits
            for entry in self._routing_plan:
                name = f"plan:{entry.get('provider', '')}"
                if self._failover._breaker.is_open(name):
                    # Allow half-open probe on next actual call
                    state = self._failover._breaker._get(name)
                    if state["state"] == "open":
                        state["opened_at"] = 0  # Force cooldown expiry
                        logger.info(f"Recovery probe: forcing half-open for {name}")

    # --- Timeout checks ---

    def check_private_mode_timeout(
        self, session_metadata: dict[str, Any], timeout_seconds: int = 1800,
    ) -> str | None:
        if not session_metadata.get("private_mode"):
            return None
        last_activity = session_metadata.get("last_user_message_at", 0)
        if not last_activity:
            return None
        elapsed = time.time() - last_activity
        if elapsed > timeout_seconds:
            return "expired"
        if elapsed > timeout_seconds - 120:
            return "warning"
        return None

    def check_use_override_timeout(
        self, session_metadata: dict[str, Any], timeout_seconds: int | None = None,
    ) -> str | None:
        if not session_metadata.get("force_provider"):
            return None
        last_activity = session_metadata.get("last_user_message_at", 0)
        if not last_activity:
            return None
        timeout = timeout_seconds or self._use_override_timeout
        elapsed = time.time() - last_activity
        if elapsed > timeout:
            return "expired"
        if elapsed > timeout - 120:
            return "warning"
        return None

    async def check_routing_preference(
        self, message_text: str, session_key: str, db_path: str,
    ) -> dict | None:
        if not db_path or not message_text:
            return None
        try:
            import aiosqlite
            words = set(message_text.lower().split())
            async with aiosqlite.connect(db_path) as db:
                cur = await db.execute(
                    """SELECT id, provider, tier, model, keywords, confidence
                       FROM routing_preferences
                       WHERE session_key = ? AND confidence >= 0.3
                       ORDER BY last_matched DESC LIMIT 20""",
                    (session_key,),
                )
                rows = await cur.fetchall()
                best = None
                best_score = 0.0
                for row_id, provider, tier, model, kw_json, conf in rows:
                    kw = set(kw_json.split(",")) if kw_json else set()
                    overlap = len(words & kw)
                    if overlap >= 2:
                        score = overlap * conf
                        if score > best_score:
                            best_score = score
                            best = {"provider": provider, "tier": tier, "model": model, "id": row_id}
                if best:
                    await db.execute(
                        "UPDATE routing_preferences SET last_matched = CURRENT_TIMESTAMP WHERE id = ?",
                        (best["id"],),
                    )
                    await db.commit()
                    logger.info(f"Routing preference matched: {best['provider']} (score={best_score:.1f})")
            return best
        except Exception as e:
            logger.warning(f"Routing preference check failed: {e}")
            return None

    def set_model(self, tier: str, model: str) -> None:
        """Hot-swap a model tier at runtime."""
        mapping = {
            "fast": "_fast_model",
            "default": "_default_model",
            "big": "_big_model",
            "escalation": "_escalation_model",
            "strongest": "_strongest_model",
            "local": "_local_model",
        }
        attr = mapping.get(tier)
        if attr:
            setattr(self, attr, model)
            logger.info(f"Model updated: {tier} → {model}")

    def set_routing_plan(self, plan: list[dict]) -> None:
        """Set a new routing plan (validated externally by PlanRoutingTool)."""
        self._routing_plan = plan
        logger.info(f"Routing plan updated: {len(plan)} entries")

    @staticmethod
    def _inject_escalation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a shallow copy of messages with escalation instruction."""
        messages = [msg.copy() for msg in messages]
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = msg["content"] + _ESCALATION_INSTRUCTION
                break
        return messages
