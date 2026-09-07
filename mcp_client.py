"""MCP client helpers for TripMate AI.

The AviationStack integration is local instead of launching the third-party
``aviationstack-mcp`` package through uvx. This avoids the stdio/session failure
seen when multiple specialist branches request tools concurrently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY") or os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
AVIATION_SERVER_PATH = BASE_DIR / "custom_aviation_mcp_server.py"


def _require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to the project .env file.")
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


client = MultiServerMCPClient({
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY or ''}",
    },
    "aviationstack": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(AVIATION_SERVER_PATH)],
        "env": _subprocess_env(AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY),
    },
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(WEATHER_SERVER_PATH)],
        "env": _subprocess_env(OPENWEATHER_API_KEY=OPENWEATHER_API_KEY),
    },
})


async def _get_server_tool(server_name: str, tool_name: str):
    if server_name == "tavily":
        _require_env("TAVILY_API_KEY", TAVILY_API_KEY)
    elif server_name == "aviationstack":
        _require_env("AVIATION_STACK_API_KEY", AVIATION_STACK_API_KEY)
        if not AVIATION_SERVER_PATH.is_file():
            raise FileNotFoundError(f"Aviation MCP server not found: {AVIATION_SERVER_PATH}")
    elif server_name == "weather":
        _require_env("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY)
        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(f"Weather MCP server not found: {WEATHER_SERVER_PATH}")
    else:
        raise ValueError(f"Unknown MCP server: {server_name}")

    tools = await client.get_tools(server_name=server_name)
    tool = next((item for item in tools if item.name == tool_name), None)
    if tool is None:
        names = ", ".join(sorted(item.name for item in tools)) or "none"
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found on server '{server_name}'. Available tools: {names}"
        )
    return tool


async def get_all_tools() -> None:
    for server_name in ("tavily", "aviationstack", "weather"):
        try:
            tools = await client.get_tools(server_name=server_name)
            names = ", ".join(tool.name for tool in tools) or "no tools"
            print(f"{server_name}: OK -> {names}")
        except Exception as exc:
            print(f"{server_name}: FAILED -> {type(exc).__name__}: {exc}")


async def tavily_mcp_search(query: str):
    tool = await _get_server_tool("tavily", "tavily_search")
    return await tool.ainvoke({"query": query.strip()})


async def aviation_mcp_call(tool_name: str, tool_args: dict[str, Any] | None = None):
    tool = await _get_server_tool("aviationstack", tool_name)
    return await tool.ainvoke(tool_args or {})


async def weather_mcp_search(city: str):
    tool = await _get_server_tool("weather", "get_weather_bundle")
    return await tool.ainvoke({"city": city.strip()})


_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        key = _require_env("GROQ_API_KEY", GROQ_API_KEY)
        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=key,
            temperature=0,
            max_retries=0,
        )
    return _llm


def extract_destination(query: str) -> str:
    prompt = f"""Extract the primary destination city or country.
Return only the destination name. If none is identifiable, return UNKNOWN.

Travel request:
{query}"""
    response = _get_llm().invoke([HumanMessage(content=prompt)])
    destination = str(response.content).strip().strip('"').strip()
    if not destination or destination.upper() == "UNKNOWN":
        raise ValueError("The destination could not be extracted from the request.")
    return destination
