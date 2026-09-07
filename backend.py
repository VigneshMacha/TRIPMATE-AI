"""TripMate AI orchestration backend.

The graph is deliberately split into independent specialist branches so flight,
hotel, weather, and budget work can run in parallel. The itinerary is produced
only after all selected specialists finish, then a real LangGraph interrupt
pauses the run for human approval.
"""

from __future__ import annotations

import asyncio
import json
import operator
import os
import threading
import uuid
from typing import Annotated, Any, TypedDict

import certifi
import psycopg
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from mcp_client import (
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    tavily_mcp_search,
    weather_mcp_search,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is missing. Add it to the project .env file.")
    return value


def get_database_url() -> str:
    url = _required_env("DATABASE_URL")
    if "sslmode=" not in url.lower():
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


_llm: ChatGroq | None = None
_llm_lock = threading.Lock()


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = ChatGroq(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    api_key=_required_env("GROQ_API_KEY"),
                )
    return _llm


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str
    budget_results: str
    itinerary: str

    approval_request: str
    approved: bool
    human_feedback: str
    approval_round: int
    final_response: str

    # Reducer is required because specialist branches execute concurrently.
    llm_calls: Annotated[int, operator.add]


KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

SPECIALIST_AGENTS = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
]

AGENT_ORDER = SPECIALIST_AGENTS + ["itinerary_agent"]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content).strip()


def _json_from_llm(text: str) -> dict[str, Any]:
    """Parse a JSON object even when the model wraps it in markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("The model did not return a JSON object.")
        value, _ = decoder.raw_decode(cleaned[start:])

    if not isinstance(value, dict):
        raise ValueError("The model returned JSON, but not an object.")
    return value


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _normalize_constraints(value: Any) -> dict[str, Any]:
    result = _empty_constraints()
    if not isinstance(value, dict):
        return result

    for key in result:
        raw = value.get(key)
        if key == "special_preferences":
            if isinstance(raw, list):
                result[key] = [str(item).strip() for item in raw if str(item).strip()]
        elif raw is not None:
            result[key] = str(raw).strip()
    return result


def _safe_text(value: Any, limit: int = 8000) -> str:
    text = str(value if value is not None else "")
    return text[:limit]


async def _parallel(*coroutines):
    return await asyncio.gather(*coroutines)


def _run_async(coro):
    """Run an async MCP operation from synchronous LangGraph nodes."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Defensive fallback for callers that invoke a node from an already-running
    # event loop. The FastAPI layer normally runs graph work in a worker thread.
    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - defensive path
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None


# -----------------------------------------------------------------------------
# Guardrail + supervisor
# -----------------------------------------------------------------------------
def supervisor_agent(state: TravelState) -> dict[str, Any]:
    query = state["user_query"]

    guardrail_prompt = f"""
Classify this user request for a travel-planning application.

Allowed: travel planning/information such as destinations, flights, hotels,
weather, budgets, visas, transportation, sightseeing, food, packing, and
itineraries. Missing trip details are allowed.
Block clearly unrelated requests and requests asking for harmful or illegal
instructions.

Return JSON only:
{{"allowed": true, "reason": "short explanation"}}

User request:
{query}
"""

    try:
        guardrail = _json_from_llm(
            _llm_text(
                "You are a strict but helpful travel input guardrail. Return JSON only.",
                guardrail_prompt,
            )
        )
        allowed = bool(guardrail.get("allowed", True))
        reason = _safe_text(guardrail.get("reason", "")).strip()
        calls = 1
    except Exception as exc:
        # A parser failure should not turn a valid travel request into a hard 500.
        print(f"Guardrail fallback: {type(exc).__name__}: {exc}", flush=True)
        allowed = True
        reason = "Guardrail validation fallback was used."
        calls = 1

    if not allowed:
        message = reason or (
            "TripMate AI is focused on travel planning. Ask about a destination, "
            "flight, hotel, weather, budget, transport, or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": message,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": message,
            "final_response": message,
            "messages": [AIMessage(content=message)],
            "llm_calls": calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Select only the specialist agents needed for the request. The itinerary_agent
is always required and should be selected last.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare
- hotel_agent: hotels, accommodation, neighborhoods
- weather_agent: current weather, forecast, climate, season, packing
- budget_agent: trip cost, affordability, budget feasibility
- itinerary_agent: integrates the selected research into a practical plan

Return JSON only:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": "short routing explanation"
}}

User request:
{query}
"""

    try:
        parsed = _json_from_llm(
            _llm_text(
                "You are the routing supervisor for TripMate. Return JSON only.",
                supervisor_prompt,
            )
        )
        requested = parsed.get("selected_agents", [])
        if not isinstance(requested, list):
            requested = []
        selected = [agent for agent in AGENT_ORDER if agent in requested and agent in KNOWN_AGENTS]
        if "itinerary_agent" not in selected:
            selected.append("itinerary_agent")
        constraints = _normalize_constraints(parsed.get("trip_constraints", {}))
        reasoning = _safe_text(parsed.get("reasoning", "")).strip()
        calls += 1
    except Exception as exc:
        print(f"Supervisor fallback: {type(exc).__name__}: {exc}", flush=True)
        selected = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = "Supervisor parsing failed; the full travel workflow was selected."
        calls += 1

    return {
        "guardrail_allowed": True,
        "guardrail_reason": reason,
        "selected_agents": selected,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor routing completed.")],
        "llm_calls": calls,
    }


def guardrail_blocked_agent(state: TravelState) -> dict[str, Any]:
    message = state.get("final_response") or state.get("guardrail_reason") or "Request blocked."
    return {"final_response": message, "messages": [AIMessage(content=message)]}


# -----------------------------------------------------------------------------
# Specialist agents
# -----------------------------------------------------------------------------
FLIGHT_AGENT_PROMPT = """
You are a professional flight-planning specialist.

User request:
{query}

Trip constraints:
{constraints}

Airport data:
{airport_data}

Airline data:
{airline_data}

Provide concise guidance covering:
- likely departure and arrival airports
- airlines serving the route
- typical duration
- estimated airfare range, clearly labeled as approximate
- peak-season warning
- booking advice
Never claim a live fare unless the supplied data proves it.
"""


def flight_agent(state: TravelState) -> dict[str, Any]:
    try:
        airports, airlines = _run_async(
            _parallel(
                aviation_mcp_call("list_airports"),
                aviation_mcp_call("list_airlines"),
            )
        )
        prompt = FLIGHT_AGENT_PROMPT.format(
            query=state["user_query"],
            constraints=state.get("trip_constraints", {}),
            airport_data=_safe_text(airports, 3500),
            airline_data=_safe_text(airlines, 3500),
        )
        result = _llm_text("You are an expert travel flight planner.", prompt)
    except Exception as exc:
        result = f"Live flight research is unavailable right now: {type(exc).__name__}: {exc}"
    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight research completed.")],
        "llm_calls": 1,
    }


def hotel_agent(state: TravelState) -> dict[str, Any]:
    query = f"Best hotels and neighborhoods for: {state['user_query']}"
    try:
        research = _run_async(tavily_mcp_search(query))
        result = _llm_text(
            "You are a hotel research specialist. Summarize evidence and label estimates clearly.",
            f"User request:\n{state['user_query']}\n\nSearch results:\n{_safe_text(research, 9000)}\n\nGive 3-5 practical accommodation options or areas and why they fit.",
        )
    except Exception as exc:
        result = f"Live hotel research is unavailable right now: {type(exc).__name__}: {exc}"
    return {
        "hotel_results": result,
        "messages": [AIMessage(content="Hotel research completed.")],
        "llm_calls": 1,
    }


def _destination_for(state: TravelState) -> str:
    constraints = state.get("trip_constraints", {})
    destination = str(constraints.get("destination", "")).strip()
    if destination:
        return destination
    return extract_destination(state["user_query"])


def weather_agent(state: TravelState) -> dict[str, Any]:
    try:
        city = _destination_for(state)
        current, forecast = _run_async(
            _parallel(
                weather_mcp_search(city),
                forecast_mcp_search(city),
            )
        )
        result = f"Current weather for {city}:\n{current}\n\nForecast:\n{forecast}"
    except Exception as exc:
        destination = state.get("trip_constraints", {}).get("destination") or "the destination"
        result = (
            f"Live weather information for {destination} is temporarily unavailable. "
            f"Verify the forecast before departure. ({type(exc).__name__}: {exc})"
        )
    return {
        "weather_results": result,
        "messages": [AIMessage(content="Weather research completed.")],
        "llm_calls": 1,
    }


def budget_agent(state: TravelState) -> dict[str, Any]:
    prompt = f"""
Assess whether this trip is realistic for the user's budget.

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight research:
{state.get('flight_results', '')}

Hotel research:
{state.get('hotel_results', '')}

Weather research:
{state.get('weather_results', '')}

Return:
1. estimated cost categories
2. budget risks
3. money-saving opportunities
4. overall feasibility
Clearly label non-live prices as estimates.
"""
    try:
        result = _llm_text("You are a practical travel budget analyst.", prompt)
    except Exception as exc:
        result = f"Budget analysis is unavailable right now: {type(exc).__name__}: {exc}"
    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget analysis completed.")],
        "llm_calls": 1,
    }


# -----------------------------------------------------------------------------
# Itinerary + HITL
# -----------------------------------------------------------------------------
def itinerary_agent(state: TravelState) -> dict[str, Any]:
    revision = state.get("human_feedback", "").strip()
    revision_context = (
        f"\nHuman revision request from the previous review:\n{revision}\n"
        if revision
        else ""
    )
    prompt = f"""
Create a practical travel itinerary draft for human review.
{revision_context}

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight research:
{state.get('flight_results', '')}

Hotel research:
{state.get('hotel_results', '')}

Weather research:
{state.get('weather_results', '')}

Budget analysis:
{state.get('budget_results', '')}

Structure the draft with:
- trip summary
- transport
- accommodation
- day-by-day plan
- estimated budget
- weather/packing notes
- assumptions and items the traveler should verify
Do not invent live availability or guaranteed prices.
"""
    try:
        draft = _llm_text("You are an expert travel itinerary designer.", prompt)
    except Exception as exc:
        draft = f"Itinerary generation failed: {type(exc).__name__}: {exc}"

    return {
        "itinerary": draft,
        "approval_request": "Review the draft. Approve it, or request a specific revision.",
        "messages": [AIMessage(content="Itinerary draft prepared for human review.")],
        "llm_calls": 1,
    }


def human_approval_agent(state: TravelState) -> dict[str, Any]:
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "approval_round": state.get("approval_round", 0) + 1,
        }
    )

    if not isinstance(review, dict):
        review = {}

    approved = bool(review.get("approved", False))
    feedback = _safe_text(review.get("feedback", ""), 4000).strip()
    return {
        "approved": approved,
        "human_feedback": feedback,
        "approval_round": state.get("approval_round", 0) + 1,
        "messages": [AIMessage(content="Human review decision received.")],
    }


def final_agent(state: TravelState) -> dict[str, Any]:
    prompt = f"""
Prepare the final travel response after human approval.

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight research:
{state.get('flight_results', '')}

Hotel research:
{state.get('hotel_results', '')}

Weather research:
{state.get('weather_results', '')}

Budget analysis:
{state.get('budget_results', '')}

Approved itinerary draft:
{state.get('itinerary', '')}

Write a polished response using these sections:
1. Trip Summary
2. Transport & Flights
3. Accommodation
4. Day-by-Day Itinerary
5. Estimated Budget
6. Weather & Packing
7. Final Recommendations

Be transparent about estimates and anything that must be verified live.
"""
    try:
        final = _llm_text("You are a professional travel planning assistant.", prompt)
    except Exception as exc:
        # A successful approval should still return the draft if final polishing fails.
        final = state.get("itinerary", "") + f"\n\nNote: final polishing failed ({type(exc).__name__}: {exc})."
    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "llm_calls": 1,
    }


# -----------------------------------------------------------------------------
# Graph routing
# -----------------------------------------------------------------------------
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def route_from_supervisor(state: TravelState) -> str | list[str]:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = [agent for agent in SPECIALIST_AGENTS if agent in state.get("selected_agents", [])]
    return selected or "itinerary_agent"


def route_after_review(state: TravelState) -> str:
    # Rejected drafts go back through itinerary_agent with the human feedback.
    return "final_agent" if state.get("approved", False) else "itinerary_agent"


builder = StateGraph(TravelState)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("guardrail_blocked", guardrail_blocked_agent)
builder.add_node("flight_agent", flight_agent)
builder.add_node("hotel_agent", hotel_agent)
builder.add_node("weather_agent", weather_agent)
builder.add_node("budget_agent", budget_agent)
builder.add_node("itinerary_agent", itinerary_agent)
builder.add_node("human_approval", human_approval_agent)
builder.add_node("final_agent", final_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

# Fan-out: all selected specialists execute in the same graph superstep.
for agent in SPECIALIST_AGENTS:
    builder.add_edge(agent, "itinerary_agent")

builder.add_edge("itinerary_agent", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    route_after_review,
    {"itinerary_agent": "itinerary_agent", "final_agent": "final_agent"},
)
builder.add_edge("final_agent", END)
builder.add_edge("guardrail_blocked", END)


# -----------------------------------------------------------------------------
# PostgreSQL checkpointer
# -----------------------------------------------------------------------------
_schema_ready = False
_schema_lock = threading.Lock()


def _ensure_checkpoint_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        # Use a short-lived setup connection. Each graph invocation receives its
        # own connection, avoiding cross-request contention on one psycopg connection.
        with psycopg.connect(get_database_url(), autocommit=True) as conn:
            PostgresSaver(conn).setup()
        _schema_ready = True


def get_travel_graph():
    _ensure_checkpoint_schema()
    conn = psycopg.connect(get_database_url(), autocommit=True)
    checkpointer = PostgresSaver(conn)
    return builder.compile(checkpointer=checkpointer), conn


# -----------------------------------------------------------------------------
# API-facing helpers
# -----------------------------------------------------------------------------
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return None
    first = interrupts[0]
    payload = getattr(first, "value", first)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    payload = _interrupt_payload(result)
    messages = result.get("messages") or []
    last_message = getattr(messages[-1], "content", "") if messages else ""

    if payload:
        answer = payload.get("draft_itinerary") or result.get("itinerary", "")
    else:
        answer = result.get("final_response") or last_message

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": payload is not None,
        "approval_request": payload.get("approval_request", "") if payload else result.get("approval_request", ""),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": payload.get("draft_itinerary", "") if payload else result.get("itinerary", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", _empty_constraints()),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "approval_round": result.get("approval_round", 0),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict[str, Any]:
    user_input = (user_input or "").strip()
    if not user_input:
        raise ValueError("Travel request cannot be empty.")
    if len(user_input) > 8000:
        raise ValueError("Travel request is too long. Keep it under 8,000 characters.")

    thread_id = thread_id.strip() if thread_id else f"trip_{uuid.uuid4().hex}"
    if len(thread_id) > 200:
        raise ValueError("thread_id is invalid.")

    graph, conn = get_travel_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "approval_round": 0,
            "final_response": "",
            "llm_calls": 0,
        },
            config=config,
        )
        return _serialize_result(result, thread_id)
    finally:
        conn.close()


def resume_travel_agent(thread_id: str, approved: bool, feedback: str = "") -> dict[str, Any]:
    thread_id = (thread_id or "").strip()
    if not thread_id or len(thread_id) > 200:
        raise ValueError("A valid thread_id is required.")

    feedback = (feedback or "").strip()
    if len(feedback) > 4000:
        raise ValueError("Feedback is too long. Keep it under 4,000 characters.")
    if not approved and not feedback:
        raise ValueError("Revision feedback is required when rejecting a draft.")

    graph, conn = get_travel_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = graph.invoke(
            Command(resume={"approved": bool(approved), "feedback": feedback}),
            config=config,
        )
        return _serialize_result(result, thread_id)
    finally:
        conn.close()
