from __future__ import annotations

from typing import Any
import logging

from aiohttp import ClientError, ClientTimeout

from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationEntityFeature,
    ConversationInput,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

try:
    from homeassistant.components.conversation import ConversationResult
except ImportError:  # pragma: no cover
    from homeassistant.components.conversation.agent import ConversationResult

from .const import (
    CONF_ORCHESTRATOR_URL,
    CONF_REQUEST_TIMEOUT,
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_REQUEST_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Deterministic conversation entity from config entry."""
    async_add_entities([DeterministicConversationEntity(hass, entry)], True)


class DeterministicConversationEntity(ConversationEntity):
    """Conversation agent that delegates planning to the Deterministic orchestrator."""

    _attr_name = "Deterministic Agent"
    _attr_supported_features = ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._session = async_get_clientsession(hass)
        self._attr_unique_id = f"deterministic_agent_{entry.entry_id}"
        self._pending_actions: dict[str, list[dict[str, Any]]] = {}

    @property
    def supported_languages(self) -> list[str] | str:
        return ["*"]

    async def _async_handle_message(self, user_input: ConversationInput, chat_log: list[dict[str, Any]] | None = None) -> ConversationResult:
        """Handle modern ConversationEntity API."""
        return await self._async_process_input(user_input)

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Handle legacy ConversationEntity API."""
        return await self._async_process_input(user_input)

    async def _async_process_input(self, user_input: ConversationInput) -> ConversationResult:
        orchestrator_url = (
            self._entry.options.get(CONF_ORCHESTRATOR_URL)
            or self._entry.data.get(CONF_ORCHESTRATOR_URL)
            or DEFAULT_ORCHESTRATOR_URL
        )
        timeout_seconds = (
            self._entry.options.get(CONF_REQUEST_TIMEOUT)
            or self._entry.data.get(CONF_REQUEST_TIMEOUT)
            or DEFAULT_REQUEST_TIMEOUT
        )

        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
        }

        # Check for pending confirmations
        conv_id = user_input.conversation_id
        if conv_id and conv_id in self._pending_actions:
            pending = self._pending_actions.pop(conv_id)
            user_text = user_input.text.lower().strip()
            if user_text in ["yes", "yeah", "yep", "sure", "do it", "confirm", "ok", "okay"]:
                error = await self._async_execute_actions(pending, user_input)
                if error:
                    return self._build_result(user_input, error)
                return self._build_result(user_input, "Done.")
            else:
                return self._build_result(user_input, "Action cancelled.")

        try:
            async with self._session.post(
                f"{orchestrator_url}/v1/ha/process",
                json=payload,
                timeout=ClientTimeout(total=timeout_seconds),
            ) as resp:
                if resp.status != 200:
                    raw = await resp.text()
                    _LOGGER.warning("Deterministic orchestrator returned status %s: %s", resp.status, raw)
                    return self._build_result(
                        user_input,
                        "Voice control is temporarily unavailable. Please try again.",
                    )
                body = await resp.json(content_type=None)
        except (ClientError, TimeoutError, ValueError) as err:
            _LOGGER.warning("Failed calling Deterministic orchestrator: %s", err)
            return self._build_result(
                user_input,
                "I could not reach the Deterministic voice service.",
            )

        speech = str(body.get("speech") or body.get("non_ha_response") or "Done.")
        needs_clarification = bool(body.get("needs_clarification"))
        needs_confirmation = bool(body.get("needs_confirmation"))
        actions = body.get("actions") or []

        if needs_confirmation and actions and conv_id:
            self._pending_actions[conv_id] = actions

        if not needs_clarification and not needs_confirmation and actions:
            error = await self._async_execute_actions(actions, user_input)
            if error is not None:
                return self._build_result(user_input, error)

        return self._build_result(
            user_input,
            speech,
            continue_conversation=needs_clarification or needs_confirmation,
        )

    async def _async_execute_actions(
        self, actions: list[dict[str, Any]], user_input: ConversationInput
    ) -> str | None:
        for action in actions[:5]:
            entity_id = action.get("entity_id")
            domain = action.get("domain")
            service = action.get("service")
            service_data = dict(action.get("service_data") or {})

            if not entity_id or not domain or not service:
                _LOGGER.warning("Skipping malformed action: %s", action)
                continue

            service_data.setdefault("entity_id", entity_id)

            try:
                await self.hass.services.async_call(
                    domain,
                    service,
                    service_data,
                    blocking=True,
                    context=user_input.context,
                )
            except Exception as err:  # noqa: BLE001
                _LOGGER.exception("Failed executing %s.%s for %s: %s", domain, service, entity_id, err)
                return f"I planned an action but failed to execute {entity_id}."

        return None

    def _build_result(
        self,
        user_input: ConversationInput,
        speech: str,
        continue_conversation: bool = False,
    ) -> ConversationResult:
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(speech)
        return ConversationResult(
            response=intent_response,
            conversation_id=user_input.conversation_id,
            continue_conversation=continue_conversation,
        )
