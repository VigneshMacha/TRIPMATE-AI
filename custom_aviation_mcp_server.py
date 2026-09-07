"""Local AviationStack MCP server used by TripMate AI."""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("AviationStack MCP Server")
API_KEY = os.getenv("AVIATION_STACK_API_KEY") or os.getenv("AVIATIONSTACK_API_KEY")
BASE_URL = "https://api.aviationstack.com/v1"
TIMEOUT = 20


def _request(endpoint: str) -> dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("AVIATION_STACK_API_KEY is missing from .env")
    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params={"access_key": API_KEY},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data
    except requests.RequestException as exc:
        raise RuntimeError(f"AviationStack request failed: {exc}") from exc


@mcp.tool()
def get_reference_data() -> dict[str, Any]:
    """Return compact airport and airline reference data for flight planning."""
    airports = _request("airports").get("data", [])
    airlines = _request("airlines").get("data", [])
    return {
        "airports": [
            {
                "name": item.get("airport_name"),
                "iata": item.get("iata_code"),
                "city": item.get("city_iata") or item.get("city_name"),
                "country": item.get("country_name"),
            }
            for item in airports[:100]
        ],
        "airlines": [
            {
                "name": item.get("airline_name"),
                "iata": item.get("iata_code"),
                "icao": item.get("icao_code"),
                "country": item.get("country_name"),
            }
            for item in airlines[:100]
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
