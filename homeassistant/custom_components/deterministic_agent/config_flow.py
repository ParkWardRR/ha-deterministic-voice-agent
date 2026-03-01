from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    CONF_ORCHESTRATOR_URL,
    CONF_REQUEST_TIMEOUT,
    DEFAULT_ORCHESTRATOR_URL,
    DEFAULT_REQUEST_TIMEOUT,
    DOMAIN,
)


def _user_schema(url: str, timeout: int) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_ORCHESTRATOR_URL, default=url): str,
            vol.Required(CONF_REQUEST_TIMEOUT, default=timeout): vol.All(
                vol.Coerce(int), vol.Range(min=3, max=60)
            ),
        }
    )


class DeterministicAgentConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle config flow for Deterministic Agent."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            orchestrator_url = user_input[CONF_ORCHESTRATOR_URL].strip()
            if not orchestrator_url.startswith(("http://", "https://")):
                errors["base"] = "invalid_url"
            else:
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Deterministic Agent",
                    data={
                        CONF_ORCHESTRATOR_URL: orchestrator_url.rstrip("/"),
                        CONF_REQUEST_TIMEOUT: user_input[CONF_REQUEST_TIMEOUT],
                    },
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_user_schema(
                DEFAULT_ORCHESTRATOR_URL, DEFAULT_REQUEST_TIMEOUT
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return DeterministicAgentOptionsFlow(config_entry)


class DeterministicAgentOptionsFlow(config_entries.OptionsFlow):
    """Handle Deterministic Agent options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        errors: dict[str, str] = {}

        current_url = self._config_entry.options.get(
            CONF_ORCHESTRATOR_URL,
            self._config_entry.data.get(CONF_ORCHESTRATOR_URL, DEFAULT_ORCHESTRATOR_URL),
        )
        current_timeout = self._config_entry.options.get(
            CONF_REQUEST_TIMEOUT,
            self._config_entry.data.get(CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT),
        )

        if user_input is not None:
            orchestrator_url = user_input[CONF_ORCHESTRATOR_URL].strip()
            if not orchestrator_url.startswith(("http://", "https://")):
                errors["base"] = "invalid_url"
            else:
                return self.async_create_entry(
                    title="",
                    data={
                        CONF_ORCHESTRATOR_URL: orchestrator_url.rstrip("/"),
                        CONF_REQUEST_TIMEOUT: user_input[CONF_REQUEST_TIMEOUT],
                    },
                )

        return self.async_show_form(
            step_id="init",
            data_schema=_user_schema(current_url, current_timeout),
            errors=errors,
        )
