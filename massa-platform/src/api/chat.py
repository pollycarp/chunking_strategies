"""
HTTP API — POST /chat endpoint.

Wraps the FinancialAgent in a simple Starlette HTTP interface so any client
(browser, Slack bot, CLI script) can send questions and receive answers.

Why Starlette directly instead of FastAPI?
Starlette is already installed as a FastMCP dependency — no extra package needed.
For a single endpoint, Starlette is sufficient. If the API grows, migrating
to FastAPI (which wraps Starlette) is a one-line change.

Request:  POST /chat
          {"message": "What was EBITDA margin in Q3 2024?", "history": [...]}

Response: {"response": "...", "tool_calls": [...], "history": [...]}
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.llm.agent import FinancialAgent


async def chat_endpoint(request: Request) -> JSONResponse:
    """
    Handles POST /chat requests.

    Reads message + optional history from the JSON body, runs the agent,
    and returns the response with updated history for multi-turn use.

    The agent is stored on app.state.agent — set during application startup.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    message: str = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "'message' field is required"}, status_code=400)

    history: list[dict] = body.get("history", [])

    agent: FinancialAgent = request.app.state.agent
    agent_response = await agent.chat(message, history)

    return JSONResponse({
        "response": agent_response.text,
        "tool_calls": [
            {"name": tc.name, "input": tc.input, "result": tc.result}
            for tc in agent_response.tool_calls
        ],
        "history": agent_response.updated_history,
    })


def create_app(agent: FinancialAgent) -> Starlette:
    """
    Creates and returns the Starlette ASGI application.

    agent: a configured FinancialAgent — injected so the app is testable
           (pass a mock agent in tests, a real one in production).
    """
    app = Starlette(routes=[Route("/chat", chat_endpoint, methods=["POST"])])
    app.state.agent = agent
    return app
