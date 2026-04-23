"""
Context window manager — keeps conversation history within the model's token limit.

Why this matters:
Each API call sends the entire conversation history plus tool results. A long
session with many tool calls can accumulate thousands of tokens. Exceeding
the context limit causes an API error; approaching it increases latency and cost.

Trimming strategy: drop from the front (oldest first).
- The most recent exchange is always most relevant
- We never trim the last user message (would leave the agent without a question)
- Tool call pairs (assistant: tool_use + user: tool_result) are removed atomically
  because Claude will reject a tool_result block with no matching tool_use block

Token counting:
We use tiktoken with the cl100k_base encoding (same as GPT-4). Claude uses a
different tokenizer, but cl100k_base is a close approximation and errs on the
side of over-counting (safer than under-counting).
"""

from __future__ import annotations

import tiktoken


class ContextManager:
    """
    Trims conversation history to fit within a token budget.

    Usage:
        cm = ContextManager(max_tokens=180_000)
        trimmed = cm.trim(messages)
        token_count = cm.count_tokens(messages)
    """

    def __init__(self, max_tokens: int = 180_000) -> None:
        """
        max_tokens: the budget for conversation history tokens.
        Leave headroom for the system prompt and the next response
        (e.g. model limit 200K → set max_tokens=160K to leave ~40K free).
        """
        self._max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Counts tokens across all messages.

        For string content: encode directly.
        For list content (tool use / tool result blocks): encode each text block.
        Adds 4 tokens per message as a conservative overhead estimate
        (role label + formatting).
        """
        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._enc.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        # text content blocks
                        text = block.get("text") or block.get("content") or ""
                        if isinstance(text, str):
                            total += len(self._enc.encode(text))
                    else:
                        # SDK objects (ToolUseBlock, TextBlock, etc.)
                        text = getattr(block, "text", None) or getattr(block, "content", None) or ""
                        if isinstance(text, str):
                            total += len(self._enc.encode(text))
        return total

    def trim(self, messages: list[dict]) -> list[dict]:
        """
        Returns a copy of messages trimmed to fit within max_tokens.

        Removes from the front (oldest messages first).
        Always keeps at least 2 messages (the most recent user + any assistant response).
        Removes tool call pairs atomically: if the oldest message is a tool_use,
        its corresponding tool_result is removed with it.

        Returns the original list unchanged if it already fits.
        """
        if self.count_tokens(messages) <= self._max_tokens:
            return messages

        trimmed = list(messages)

        while len(trimmed) > 2 and self.count_tokens(trimmed) > self._max_tokens:
            # Always remove in pairs: assistant message + following user (tool_result)
            # This keeps the history structurally valid for the API.
            if (
                len(trimmed) >= 2
                and trimmed[0].get("role") == "assistant"
                and trimmed[1].get("role") == "user"
            ):
                trimmed = trimmed[2:]
            else:
                # Single message at front (e.g. initial user turn) — remove it
                trimmed = trimmed[1:]

        return trimmed
