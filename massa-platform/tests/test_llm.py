"""
Phase 7 Tests: LLM Integration

Test categories:
- system_prompt tests  : prompt contains required elements
- context_manager tests: token counting and trimming logic
- client tests         : MCP → Anthropic tool format conversion
- agent tests          : agentic loop with mocked Claude client
- api tests            : HTTP endpoint request/response contract
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from src.api.chat import create_app
from src.llm.agent import AgentResponse, FinancialAgent, ToolCall
from src.llm.client import ClaudeClient
from src.llm.context_manager import ContextManager
from src.llm.system_prompt import build_system_prompt


# ---------------------------------------------------------------------------
# Helpers — mock Anthropic response objects
# ---------------------------------------------------------------------------

def _text_response(text: str):
    """Mock Anthropic Message with stop_reason=end_turn and a TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _tool_use_response(name: str, tool_input: dict, use_id: str = "use_abc123"):
    """Mock Anthropic Message with stop_reason=tool_use and a ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = use_id

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [block]
    return response


def _mock_mcp_tool(name: str, description: str, schema: dict | None = None):
    """Creates a mock FastMCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = schema or {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    return tool


# ---------------------------------------------------------------------------
# Test 1: System prompt
# ---------------------------------------------------------------------------

def test_system_prompt_contains_role_definition():
    """Prompt must identify the assistant's role and domain."""
    prompt = build_system_prompt()
    assert "financial analyst" in prompt.lower()
    assert "MASSA Advisors" in prompt or "massa" in prompt.lower()


def test_system_prompt_mentions_both_tools():
    """Prompt must reference both retrieve_docs and query_metrics."""
    prompt = build_system_prompt()
    assert "retrieve_docs" in prompt
    assert "query_metrics" in prompt


def test_system_prompt_requires_citations():
    """Prompt must mandate source citation."""
    prompt = build_system_prompt()
    assert "cite" in prompt.lower() or "citation" in prompt.lower() or "source" in prompt.lower()


def test_system_prompt_forbids_fabrication():
    """Prompt must explicitly prohibit making up figures."""
    prompt = build_system_prompt()
    # Variations: "never fabricate", "do not fabricate", "NEVER"
    prompt_lower = prompt.lower()
    assert "fabricat" in prompt_lower or "never" in prompt_lower


def test_system_prompt_with_schema_includes_schema():
    """When schema_description is provided it should appear in the prompt."""
    schema = "TABLE fact_financials\n  revenue (numeric)"
    prompt = build_system_prompt(schema_description=schema)
    assert "fact_financials" in prompt
    assert "revenue" in prompt


# ---------------------------------------------------------------------------
# Test 2: ContextManager
# ---------------------------------------------------------------------------

def test_context_manager_counts_tokens_string_content():
    """count_tokens on a simple string message returns a positive integer."""
    cm = ContextManager(max_tokens=100_000)
    messages = [{"role": "user", "content": "What is the EBITDA margin?"}]
    count = cm.count_tokens(messages)
    assert count > 0
    assert count < 50  # short message should be well under 50 tokens


def test_context_manager_counts_tokens_list_content():
    """count_tokens handles list-style content (tool results)."""
    cm = ContextManager(max_tokens=100_000)
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "text": "Revenue was 100000"}],
        }
    ]
    count = cm.count_tokens(messages)
    assert count > 0


def test_context_manager_no_trim_when_under_limit():
    """Messages under the token limit are returned unchanged."""
    cm = ContextManager(max_tokens=100_000)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = cm.trim(messages)
    assert result == messages


def test_context_manager_trims_oldest_first():
    """
    When over the token limit, oldest messages are dropped first.
    The most recent user message must always be present in the result.
    """
    cm = ContextManager(max_tokens=20)  # very small budget

    # Create messages where the combined content exceeds 20 tokens
    messages = [
        {"role": "user", "content": "First question about revenue growth and EBITDA margins"},
        {"role": "assistant", "content": "Here is the detailed answer about financials"},
        {"role": "user", "content": "Follow up question about debt ratios"},
        {"role": "assistant", "content": "The debt to EBITDA ratio is shown below"},
        {"role": "user", "content": "Latest question"},
    ]

    result = cm.trim(messages)

    # Most recent message must always be kept
    assert result[-1]["content"] == "Latest question"
    # Result should be smaller than input
    assert len(result) < len(messages)


def test_context_manager_never_trims_below_two_messages():
    """trim() must always return at least 2 messages."""
    cm = ContextManager(max_tokens=1)  # absurdly small — forces trimming

    messages = [
        {"role": "user", "content": "A" * 1000},
        {"role": "assistant", "content": "B" * 1000},
        {"role": "user", "content": "C" * 1000},
    ]

    result = cm.trim(messages)
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# Test 3: ClaudeClient tool conversion
# ---------------------------------------------------------------------------

def test_convert_mcp_tools_renames_input_schema():
    """
    MCP tools use camelCase 'inputSchema'; Anthropic requires snake_case 'input_schema'.
    Conversion must rename the key.
    """
    mcp_tools = [_mock_mcp_tool("retrieve_docs", "Search docs")]
    result = ClaudeClient.convert_mcp_tools(mcp_tools)

    assert len(result) == 1
    assert "input_schema" in result[0]
    assert "inputSchema" not in result[0]


def test_convert_mcp_tools_preserves_name_and_description():
    """Conversion must not alter name or description."""
    mcp_tools = [_mock_mcp_tool("query_metrics", "Query financial metrics")]
    result = ClaudeClient.convert_mcp_tools(mcp_tools)

    assert result[0]["name"] == "query_metrics"
    assert result[0]["description"] == "Query financial metrics"


def test_convert_mcp_tools_adds_type_object_if_missing():
    """
    Anthropic requires input_schema to have "type": "object" at the top level.
    If the MCP tool schema omits it, conversion must add it.
    """
    tool = MagicMock()
    tool.name = "list_sources"
    tool.description = "List sources"
    tool.inputSchema = {}  # missing "type"

    result = ClaudeClient.convert_mcp_tools([tool])
    assert result[0]["input_schema"]["type"] == "object"


def test_convert_multiple_mcp_tools():
    """All tools in the list are converted."""
    mcp_tools = [
        _mock_mcp_tool("retrieve_docs", "Search docs"),
        _mock_mcp_tool("query_metrics", "Query metrics"),
        _mock_mcp_tool("list_sources", "List sources"),
    ]
    result = ClaudeClient.convert_mcp_tools(mcp_tools)
    assert len(result) == 3
    assert {t["name"] for t in result} == {"retrieve_docs", "query_metrics", "list_sources"}


# ---------------------------------------------------------------------------
# Test 4: FinancialAgent — agentic loop (mock Claude client)
# ---------------------------------------------------------------------------

def _make_agent(claude_responses: list, server=None) -> FinancialAgent:
    """
    Creates a FinancialAgent with a mocked Claude client that returns
    the given responses in sequence.

    For the server, we use a real (but minimal) FastMCP server so that
    list_tools() and call_tool() work without a real DB.
    """
    from fastmcp import FastMCP

    if server is None:
        # Minimal server with one tool that returns a fixed string
        server = FastMCP("test")

        @server.tool()
        async def query_metrics(metric_name: str) -> str:
            """Query a metric."""
            return f"Revenue for Q3 2024: 100000.00"

        @server.tool()
        async def retrieve_docs(query: str) -> str:
            """Retrieve documents."""
            return "[1] report.pdf — page 1\nEBITDA margin improved to 25%."

        @server.tool()
        async def list_sources() -> str:
            """List sources."""
            return "Available: dim_company, financial_metrics"

    mock_client = MagicMock(spec=ClaudeClient)
    mock_client.complete = AsyncMock(side_effect=claude_responses)
    mock_client.convert_mcp_tools = ClaudeClient.convert_mcp_tools  # use real conversion

    return FinancialAgent(
        client=mock_client,
        server=server,
        system_prompt="You are a financial assistant.",
    )


async def test_agent_returns_direct_response_without_tools():
    """
    When Claude responds with end_turn immediately (no tools needed),
    the agent returns that text in AgentResponse.
    """
    agent = _make_agent([_text_response("The EBITDA margin was 25%.")])

    response = await agent.chat("What was the EBITDA margin?")

    assert response.text == "The EBITDA margin was 25%."
    assert response.tool_calls == []


async def test_agent_calls_query_metrics_for_numbers():
    """
    Given a quantitative question, Claude requests query_metrics.
    The agent executes the tool and feeds the result back.
    Claude then provides a final answer.
    """
    agent = _make_agent([
        _tool_use_response("query_metrics", {"metric_name": "revenue"}),
        _text_response("Revenue for Q3 2024 was 100,000."),
    ])

    response = await agent.chat("What was the revenue in Q3 2024?")

    assert response.text == "Revenue for Q3 2024 was 100,000."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "query_metrics"
    assert "100000" in response.tool_calls[0].result


async def test_agent_calls_retrieve_docs_for_narrative():
    """
    Given a qualitative question, Claude requests retrieve_docs.
    The agent executes it and the tool result appears in tool_calls.
    """
    agent = _make_agent([
        _tool_use_response("retrieve_docs", {"query": "EBITDA margin performance"}),
        _text_response("According to report.pdf, EBITDA margin improved to 25%."),
    ])

    response = await agent.chat("What did the board report say about EBITDA margins?")

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "retrieve_docs"
    assert "EBITDA margin" in response.tool_calls[0].result


async def test_agent_handles_multi_step_tool_calls():
    """
    The agent can handle a two-step sequence: Claude first calls list_sources,
    then calls query_metrics, then gives a final answer.
    """
    agent = _make_agent([
        _tool_use_response("list_sources", {}, use_id="use_1"),
        _tool_use_response("query_metrics", {"metric_name": "revenue"}, use_id="use_2"),
        _text_response("Based on the data, revenue was 100,000."),
    ])

    response = await agent.chat("What companies do you have data for and what was their revenue?")

    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].name == "list_sources"
    assert response.tool_calls[1].name == "query_metrics"
    assert response.text == "Based on the data, revenue was 100,000."


async def test_agent_updated_history_includes_full_conversation():
    """
    updated_history must contain all turns: user, assistant tool-use,
    user tool-result, and final assistant text.
    """
    agent = _make_agent([
        _tool_use_response("query_metrics", {"metric_name": "ebitda"}),
        _text_response("EBITDA was 25,000."),
    ])

    response = await agent.chat("What was EBITDA?")

    roles = [msg["role"] for msg in response.updated_history]
    # Sequence: user → assistant (tool_use) → user (tool_result) → assistant (text)
    assert roles == ["user", "assistant", "user", "assistant"]


async def test_agent_raises_on_max_iterations():
    """If Claude never stops calling tools, the agent raises RuntimeError."""
    # Infinite tool-call loop: always returns tool_use, never end_turn
    infinite_responses = [
        _tool_use_response("list_sources", {}, use_id=f"use_{i}")
        for i in range(15)
    ]

    agent = _make_agent(infinite_responses)
    agent._max_iterations = 3  # low limit to trigger quickly

    with pytest.raises(RuntimeError, match="max_iterations"):
        await agent.chat("What is the data?")


# ---------------------------------------------------------------------------
# Test 5: HTTP API endpoint
# ---------------------------------------------------------------------------

def _make_mock_agent(response_text: str = "The answer is 42.") -> FinancialAgent:
    """Creates a FinancialAgent mock for HTTP endpoint tests."""
    mock_agent = MagicMock(spec=FinancialAgent)
    mock_agent.chat = AsyncMock(return_value=AgentResponse(
        text=response_text,
        tool_calls=[ToolCall(name="query_metrics", input={"metric_name": "revenue"}, result="100000")],
        updated_history=[
            {"role": "user", "content": "What was revenue?"},
            {"role": "assistant", "content": response_text},
        ],
    ))
    return mock_agent


def test_chat_endpoint_returns_200_with_response():
    """POST /chat returns 200 with response, tool_calls, and history fields."""
    app = create_app(_make_mock_agent("Revenue was 100,000."))
    client = TestClient(app)

    resp = client.post("/chat", json={"message": "What was revenue in Q3 2024?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"] == "Revenue was 100,000."
    assert "tool_calls" in body
    assert "history" in body


def test_chat_endpoint_passes_history_to_agent():
    """History from the request body is forwarded to agent.chat()."""
    mock_agent = _make_mock_agent()
    app = create_app(mock_agent)
    client = TestClient(app)

    prior_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    client.post("/chat", json={"message": "Follow-up", "history": prior_history})

    # Verify agent.chat was called with the prior history.
    # chat(message, history) is called positionally, so args[1] is history.
    call = mock_agent.chat.call_args
    history_passed = call.args[1] if len(call.args) > 1 else call.kwargs.get("history")
    assert history_passed == prior_history


def test_chat_endpoint_400_on_missing_message():
    """POST /chat without a 'message' field returns 400."""
    app = create_app(_make_mock_agent())
    client = TestClient(app)

    resp = client.post("/chat", json={"history": []})
    assert resp.status_code == 400
    assert "message" in resp.json().get("error", "").lower()


def test_chat_endpoint_400_on_invalid_json():
    """POST /chat with non-JSON body returns 400."""
    app = create_app(_make_mock_agent())
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/chat", content=b"not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400
