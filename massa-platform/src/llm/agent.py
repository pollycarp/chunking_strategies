"""
Financial agent — the agentic loop that connects Claude to the MCP tools.

Agentic loop pattern:
┌─────────────────────────────────────────────┐
│ 1. Build messages = history + [user message] │
│ 2. Trim messages to fit context window       │
│ 3. Call Claude with messages + tool defs     │
│ 4a. stop_reason = "end_turn"                 │
│     → extract text, return AgentResponse     │
│ 4b. stop_reason = "tool_use"                 │
│     → execute each tool via FastMCP Client   │
│     → append assistant + tool_result msgs    │
│     → go to step 2                           │
└─────────────────────────────────────────────┘

Why this structure?
Claude does not call tools directly — it returns a tool_use block saying
"please call X with args Y". The agent executes the call and feeds the result
back as a tool_result message. Only then does Claude see the answer and
decide what to do next (call another tool, or answer the user).

Multi-tool turns: Claude can request multiple tools in one response.
We execute all of them in parallel and return all results in one user message.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from fastmcp import Client as MCPClient
from fastmcp import FastMCP

from src.llm.client import ClaudeClient
from src.llm.context_manager import ContextManager
from src.llm.system_prompt import build_system_prompt


@dataclass
class ToolCall:
    """Records one tool invocation during an agent turn."""
    name: str
    input: dict
    result: str


@dataclass
class AgentResponse:
    """The agent's answer plus everything that happened to produce it."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    updated_history: list[dict] = field(default_factory=list)


class FinancialAgent:
    """
    An agentic Claude loop connected to the MASSA MCP server.

    Usage:
        agent = FinancialAgent(
            client=ClaudeClient(api_key=settings.anthropic_api_key),
            server=create_server(pool, embedder),
        )
        response = await agent.chat("What was Agri Co's EBITDA margin in Q3 2024?")
        print(response.text)
    """

    def __init__(
        self,
        client: ClaudeClient,
        server: FastMCP,
        system_prompt: str = "",
        max_iterations: int = 10,
        context_manager: ContextManager | None = None,
    ) -> None:
        self._client = client
        self._server = server
        self._system_prompt = system_prompt or build_system_prompt()
        self._max_iterations = max_iterations
        self._context_manager = context_manager or ContextManager()

    async def chat(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> AgentResponse:
        """
        Runs the agentic loop for one user turn.

        Parameters:
            message : the user's question
            history : prior conversation messages (list of {role, content} dicts)
                      Pass the updated_history from the previous AgentResponse
                      to maintain a multi-turn conversation.

        Returns AgentResponse with:
            text           : Claude's final text answer
            tool_calls     : list of all tool calls made during this turn
            updated_history: full conversation history including this turn
        """
        messages = list(history or []) + [{"role": "user", "content": message}]
        all_tool_calls: list[ToolCall] = []

        async with MCPClient(self._server) as mcp:
            # Fetch tool definitions once per turn (they don't change mid-turn)
            mcp_tools = await mcp.list_tools()
            anthropic_tools = ClaudeClient.convert_mcp_tools(mcp_tools)

            for _ in range(self._max_iterations):
                trimmed = self._context_manager.trim(messages)

                response = await self._client.complete(
                    messages=trimmed,
                    system=self._system_prompt,
                    tools=anthropic_tools if anthropic_tools else None,
                )

                if response.stop_reason == "end_turn":
                    # Claude has finished — extract the text response
                    text = _extract_text(response)
                    messages.append({"role": "assistant", "content": text})
                    return AgentResponse(
                        text=text,
                        tool_calls=all_tool_calls,
                        updated_history=messages,
                    )

                elif response.stop_reason == "tool_use":
                    # Append assistant's tool-use message (includes the tool_use blocks)
                    messages.append({"role": "assistant", "content": response.content})

                    # Execute all requested tool calls (possibly multiple in one turn)
                    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                    results = await asyncio.gather(*[
                        _execute_tool(mcp, block.name, block.input)
                        for block in tool_use_blocks
                    ])

                    # Record what happened
                    tool_result_content = []
                    for block, result_text in zip(tool_use_blocks, results):
                        all_tool_calls.append(ToolCall(
                            name=block.name,
                            input=dict(block.input),
                            result=result_text,
                        ))
                        tool_result_content.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                    messages.append({"role": "user", "content": tool_result_content})

                else:
                    # Unexpected stop reason (e.g. max_tokens) — return what we have
                    text = _extract_text(response) or "[Response truncated]"
                    messages.append({"role": "assistant", "content": text})
                    return AgentResponse(
                        text=text,
                        tool_calls=all_tool_calls,
                        updated_history=messages,
                    )

        raise RuntimeError(
            f"Agent reached max_iterations ({self._max_iterations}) without a final answer. "
            "This may indicate the model is stuck in a tool-call loop."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    """Extracts the text from an Anthropic Message response."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


async def _execute_tool(mcp: MCPClient, name: str, tool_input: dict) -> str:
    """
    Executes a tool via the FastMCP client and returns the result as a string.

    FastMCP's call_tool returns a list of content blocks.
    We join all text blocks into a single string.
    """
    result = await mcp.call_tool(name, tool_input)

    if isinstance(result, list):
        parts = []
        for block in result:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)

    return str(result)
