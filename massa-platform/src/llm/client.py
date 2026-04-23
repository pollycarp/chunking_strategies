"""
Claude API client — thin wrapper around the Anthropic AsyncAnthropic SDK.

Why a wrapper rather than using the SDK directly?
1. Centralises the model name and default parameters in one place
2. Makes the agent testable — tests mock this class, not the SDK internals
3. Converts FastMCP tool definitions (MCP format) to Anthropic tool format here,
   keeping the conversion logic out of the agent loop
"""

from __future__ import annotations

import anthropic


class ClaudeClient:
    """
    Async Claude API client for the financial agent.

    Wraps anthropic.AsyncAnthropic and handles:
    - Default model and token settings
    - MCP → Anthropic tool format conversion
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
    ) -> anthropic.types.Message:
        """
        Sends messages to Claude and returns the raw Anthropic Message response.

        The caller (FinancialAgent) is responsible for inspecting stop_reason
        and looping if a tool call is requested.

        tools: list of dicts in Anthropic format:
            [{"name": ..., "description": ..., "input_schema": {...}}, ...]
        """
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        return await self._client.messages.create(**kwargs)

    @staticmethod
    def convert_mcp_tools(mcp_tools: list) -> list[dict]:
        """
        Converts FastMCP tool objects to Anthropic tool format.

        MCP Tool object:      name, description, inputSchema (camelCase)
        Anthropic tool dict:  name, description, input_schema (snake_case)

        Also ensures the input_schema has "type": "object" as required by
        the Anthropic API — MCP tools occasionally omit this top-level field.
        """
        result = []
        for tool in mcp_tools:
            schema = dict(getattr(tool, "inputSchema", {}) or {})
            schema.setdefault("type", "object")
            schema.setdefault("properties", {})

            result.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": schema,
            })
        return result
