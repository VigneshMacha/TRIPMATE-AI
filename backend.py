"""TripMate AI orchestration backend.

The workflow uses LangGraph fan-out/fan-in: selected research specialists run
in parallel, the itinerary is synthesized once, and LangGraph pauses for HITL.
LLM/MCP failures are isolated so one optional integration does not crash the
whole request.
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
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from mcp_client import (
    aviation_mcp_call,
    extract_destination,
    tavily_mcp_search,
    weather_mcp_search,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

MAX_LLM_RETRIES = 0
MISTRAL_MAX_RETRIES = 0
GROQ_MAX_OUTPUT_TOKENS = int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1200"))
MISTRAL_MAX_OUTPUT_TOKENS = int(os.getenv("MISTRAL_MAX_OUTPUT_TOKENS", "1000"))


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


_llm_cache: dict[str, ChatGroq] = {}
_mistral_cache: dict[str, ChatMistralAI] = {}
_llm_lock = threading.Lock()


def _model_name() -> str:
    # The smaller model is a safer default for Groq free/on-demand TPM limits.
    return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_llm(model: str | None = None) -> ChatGroq:
    model = model or _model_name()
    with _llm_lock:
        llm = _llm_cache.get(model)
        if llm is None:
            llm = ChatGroq(
                model=model,
                api_key=_required_env("GROQ_API_KEY"),
                temperature=0,
                max_retries=0,
                max_tokens=GROQ_MAX_OUTPUT_TOKENS,
            )
            _llm_cache[model] = llm
        return llm


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
    research_summary: str
    itinerary: str
    approval_request: str
    approved: bool
    human_feedback: str
    approval_round: int
    final_response: str
    llm_calls: Annotated[int, operator.add]


KNOWN_AGENTS = {
    "flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"
}
SPECIALIST_AGENTS = ["flight_agent", "hotel_agent", "weather_agent", "budget_agent"]
AGENT_ORDER = SPECIALIST_AGENTS + ["itinerary_agent"]


def _is_rate_limit(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text or "tokens per minute" in text




def get_mistral_llm(model: str | None = None) -> ChatMistralAI:
    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    with _llm_lock:
        llm = _mistral_cache.get(model)
        if llm is None:
            llm = ChatMistralAI(
                model=model,
                api_key=_required_env("MISTRAL_API_KEY"),
                temperature=0,
                max_retries=MISTRAL_MAX_RETRIES,
                max_tokens=MISTRAL_MAX_OUTPUT_TOKENS,
            )
            _mistral_cache[model] = llm
        return llm


def _mistral_text(system_prompt: str, user_prompt: str) -> str:
    """Call Mistral once; quota/rate-limit failures are surfaced for provider fallback."""
    try:
        response = get_mistral_llm().invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return str(response.content).strip()
    except Exception as exc:
        if _is_rate_limit(exc):
            raise RuntimeError(
                "Mistral is rate-limited or quota-limited. Falling back to Groq for research synthesis."
            ) from exc
        raise

def _llm_text(system_prompt: str, user_prompt: str) -> str:
    """Make one Groq request. Do not retry a 429 or silently switch models.

    Groq TPM/quota limits are account-scoped, so an immediate second request
    usually makes the situation worse. Keeping this function single-shot also
    prevents a 429 -> retry -> 404 cascade.
    """
    model = _model_name()
    try:
        response = get_llm(model).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return str(response.content).strip()
    except Exception as exc:
        if _is_rate_limit(exc):
            raise RuntimeError(
                f"Groq is rate-limited or quota-limited for model '{model}'."
            ) from exc
        raise RuntimeError(f"Groq request failed for model '{model}': {exc}") from exc

def _json_from_llm(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "", 1).replace("```", "", 1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("The model did not return a JSON object.")
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("The model returned JSON, but not an object.")
    return value


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "", "origin": "", "duration": "", "budget": "",
        "travel_style": "", "special_preferences": [],
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


def _safe_text(value: Any, limit: int = 6000) -> str:
    text = str(value if value is not None else "")
    return text[:limit]


async def _parallel(*coroutines):
    return await asyncio.gather(*coroutines, return_exceptions=True)


def _run_async(coro):
    """Run an async MCP operation from a synchronous LangGraph node."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    errors: list[BaseException] = []

    def runner():
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0] if result else None


def supervisor_agent(state: TravelState) -> dict[str, Any]:
    """Single LLM routing call: guardrail + agent selection + constraint extraction.

    Keeping these decisions in one request is important on Groq TPM-limited tiers.
    """
    query = state["user_query"]
    prompt = f"""
You are TripMate AI's routing supervisor.
Analyze the user's request and return JSON ONLY.

Allowed requests: travel planning/information, destinations, flights, hotels,
weather, budgets, transport, sightseeing, food, packing, visas and itineraries.
Block clearly unrelated or harmful/illegal requests.

Choose only the specialists that are useful. itinerary_agent is always implied
and must NOT be included in selected_agents.

Agents:
- flight_agent: flights, airports, airlines, routes, airfare
- hotel_agent: hotels, accommodation, neighborhoods
- weather_agent: current weather, forecast, climate, season, packing
- budget_agent: cost, affordability, budget feasibility

Return exactly:
{{
  "allowed": true,
  "reason": "short explanation",
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent"],
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
        parsed = _json_from_llm(_llm_text(
            "You are a strict but helpful travel routing supervisor. Return JSON only.",
            prompt,
        ))
        allowed = bool(parsed.get("allowed", True))
        reason = _safe_text(parsed.get("reason", "")).strip()
        requested = parsed.get("selected_agents", [])
        if not isinstance(requested, list):
            requested = []
        selected = [a for a in SPECIALIST_AGENTS if a in requested]
        constraints = _normalize_constraints(parsed.get("trip_constraints", {}))
        reasoning = _safe_text(parsed.get("reasoning", "")).strip()
    except Exception as exc:
        print(f"Supervisor fallback: {type(exc).__name__}: {exc}", flush=True)
        allowed = True
        reason = "Supervisor fallback used."
        selected = SPECIALIST_AGENTS.copy()
        constraints = _empty_constraints()
        reasoning = "Supervisor fallback selected the complete workflow."

    if not allowed:
        message = reason or "TripMate AI is focused on travel planning."
        return {
            "guardrail_allowed": False, "guardrail_reason": message,
            "selected_agents": [], "trip_constraints": constraints,
            "supervisor_reasoning": message, "final_response": message,
            "messages": [AIMessage(content=message)], "llm_calls": 1,
        }

    return {
        "guardrail_allowed": True, "guardrail_reason": reason,
        "selected_agents": selected, "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor routing completed.")],
        "llm_calls": 1,
    }

def guardrail_blocked_agent(state: TravelState) -> dict[str, Any]:
    message = state.get("final_response") or state.get("guardrail_reason") or "Request blocked."
    return {"final_response": message, "messages": [AIMessage(content=message)]}


def flight_agent(state: TravelState) -> dict[str, Any]:
    try:
        data = _run_async(aviation_mcp_call("get_reference_data"))
        result = f"""AviationStack reference evidence (NOT live fares):
{_safe_text(data, 5000)}

Use this evidence only to identify likely airports/airlines. Do not claim live
availability, schedules or guaranteed prices."""
    except Exception as exc:
        result = f"Flight research is unavailable right now. Verify flights live before booking. ({type(exc).__name__}: {exc})"
    return {"flight_results": result, "messages": [AIMessage(content="Flight research completed.")], "llm_calls": 0}

def hotel_agent(state: TravelState) -> dict[str, Any]:
    try:
        research = _run_async(tavily_mcp_search(
            f"Best hotels, accommodation areas, neighborhoods and typical price ranges for: {state['user_query']}"
        ))
        result = f"Live web research from Tavily (verify prices/availability before booking):\n{_safe_text(research, 7000)}"
    except Exception as exc:
        result = f"Live hotel research is unavailable right now. ({type(exc).__name__}: {exc})"
    return {"hotel_results": result, "messages": [AIMessage(content="Hotel research completed.")], "llm_calls": 0}

def _destination_for(state: TravelState) -> str:
    destination = str(state.get("trip_constraints", {}).get("destination", "")).strip()
    if destination:
        return destination
    # Avoid a separate LLM call here; itinerary synthesis can still proceed.
    return ""

def weather_agent(state: TravelState) -> dict[str, Any]:
    try:
        city = _destination_for(state)
        if not city:
            raise ValueError("Destination was not identified by the supervisor.")
        data = _run_async(weather_mcp_search(city))
        result = f"Weather for {city}:\n{_safe_text(data, 3500)}"
    except Exception as exc:
        destination = state.get("trip_constraints", {}).get("destination") or "the destination"
        result = f"Live weather for {destination} is temporarily unavailable. Verify before departure. ({type(exc).__name__}: {exc})"
    return {"weather_results": result, "messages": [AIMessage(content="Weather research completed.")], "llm_calls": 0}


def budget_agent(state: TravelState) -> dict[str, Any]:
    # Budget analysis is intentionally deterministic-free of another external
    # request: it consumes the specialist evidence and is folded into synthesis.
    result = f"""Budget inputs for itinerary synthesis:
Requested budget: {state.get('trip_constraints', {}).get('budget') or 'not specified'}
Flight evidence: {_safe_text(state.get('flight_results', ''), 1800)}
Hotel evidence: {_safe_text(state.get('hotel_results', ''), 1800)}
Provide a realistic estimate and clearly mark all non-live prices as estimates."""
    return {"budget_results": result, "messages": [AIMessage(content="Budget analysis prepared.")], "llm_calls": 0}


def research_synthesis_agent(state: TravelState) -> dict[str, Any]:
    """Use Mistral as a dedicated evidence/research reasoning layer.

    This is deliberately separate from itinerary generation: Mistral turns the
    parallel MCP results into compact, structured evidence, reducing the amount
    of noisy tool output sent to the final itinerary model.
    """
    prompt = f"""You are TripMate AI's research analyst.
Synthesize the following research gathered in parallel. Do not invent facts.
Separate verified/live-looking evidence from estimates or assumptions.
Highlight conflicts, missing information, and booking verification requirements.
Return concise structured text with these headings:
- Confirmed Evidence
- Best Options / Recommendations
- Conflicts or Uncertainty
- Budget Implications
- Final Verification Checklist

User request:
{state['user_query']}

Constraints:
{state.get('trip_constraints', {})}

Flight evidence:
{_safe_text(state.get('flight_results', ''), 1800)}

Hotel evidence:
{_safe_text(state.get('hotel_results', ''), 2200)}

Weather evidence:
{_safe_text(state.get('weather_results', ''), 1800)}

Budget inputs:
{_safe_text(state.get('budget_results', ''), 1200)}
"""
    try:
        summary = _mistral_text(
            "You are a careful travel research analyst. Never fabricate live travel data.",
            prompt,
        )
        provider = "Mistral"
    except Exception as exc:
        # Keep the 3-stage architecture even when Mistral's account is rate-limited:
        # the same research role is executed by Groq rather than discarding synthesis.
        try:
            summary = _llm_text(
                "You are a careful travel research analyst. Never fabricate live travel data. "
                "Mistral was unavailable, so perform the same research-synthesis role now.",
                prompt,
            )
            provider = "Groq fallback"
        except Exception as fallback_exc:
            summary = (
                "Research synthesis is temporarily unavailable. Use the raw specialist "
                f"evidence with caution. (Mistral: {type(exc).__name__}; fallback: {type(fallback_exc).__name__})"
            )
            provider = "none"
    return {
        "research_summary": summary,
        "messages": [AIMessage(content=f"Research synthesis completed ({provider}).")],
        "llm_calls": 1,
    }


def _degraded_itinerary(state: TravelState, reason: str) -> str:
    """Return a useful non-LLM draft when the provider is unavailable."""
    c = state.get("trip_constraints", {})
    destination = c.get("destination") or "your destination"
    duration = c.get("duration") or "the requested duration"
    budget = c.get("budget") or "not specified"
    return f"""TripMate AI — Temporary Draft

Trip Summary
- Destination: {destination}
- Duration: {duration}
- Budget: {budget}

Research available
{_safe_text(state.get('research_summary', ''), 5000)}

Transport evidence
{_safe_text(state.get('flight_results', ''), 1400)}

Accommodation evidence
{_safe_text(state.get('hotel_results', ''), 1400)}

Weather & packing evidence
{_safe_text(state.get('weather_results', ''), 1200)}

Important
The itinerary language model is temporarily unavailable ({reason}). This draft is based on collected research and must be reviewed before booking. Verify live prices, availability, schedules, and weather.
"""

def itinerary_agent(state: TravelState) -> dict[str, Any]:
    revision = state.get("human_feedback", "").strip()
    prompt = f"""Create a practical travel itinerary draft for human review.
{"Human revision request: " + revision if revision else ""}

User request: {state['user_query']}
Trip constraints: {state.get('trip_constraints', {})}
Research synthesis: {_safe_text(state.get('research_summary', ''), 4200)}
Flight evidence: {_safe_text(state.get('flight_results', ''), 1000)}
Hotel evidence: {_safe_text(state.get('hotel_results', ''), 1200)}
Weather evidence: {_safe_text(state.get('weather_results', ''), 1000)}
Budget inputs: {_safe_text(state.get('budget_results', ''), 800)}

Return concise markdown with exactly these sections:
1. Trip Summary
2. Transport
3. Accommodation
4. Day-by-Day Plan
5. Estimated Budget
6. Weather & Packing
7. Assumptions / Verify Live

Do not invent live availability or guaranteed prices. Keep the answer concise.
"""
    try:
        draft = _llm_text("You are an expert travel itinerary designer. Be concise and factual.", prompt)
        provider = "Groq"
    except Exception as exc:
        print(f"Itinerary LLM unavailable: {type(exc).__name__}: {exc}", flush=True)
        draft = _degraded_itinerary(state, str(exc))
        provider = "degraded evidence draft"
    return {
        "itinerary": draft,
        "approval_request": "Review the draft. Approve it, or request a specific revision.",
        "messages": [AIMessage(content=f"Itinerary draft prepared ({provider}).")],
        "llm_calls": 1 if provider == "Groq" else 0,
    }

def human_approval_agent(state: TravelState) -> dict[str, Any]:
    review = interrupt({
        "question": "Do you approve this itinerary?",
        "draft_itinerary": state.get("itinerary", ""),
        "approval_request": state.get("approval_request", ""),
        "selected_agents": state.get("selected_agents", []),
        "supervisor_reasoning": state.get("supervisor_reasoning", ""),
        "approval_round": state.get("approval_round", 0) + 1,
    })
    review = review if isinstance(review, dict) else {}
    approved = bool(review.get("approved", False))
    feedback = _safe_text(review.get("feedback", ""), 4000).strip()
    return {
        "approved": approved, "human_feedback": feedback,
        "approval_round": state.get("approval_round", 0) + 1,
        "messages": [AIMessage(content="Human review decision received.")],
    }


def final_agent(state: TravelState) -> dict[str, Any]:
    # The itinerary has already passed HITL. Avoid a second synthesis request:
    # this saves a Groq call and prevents another TPM spike immediately after approval.
    final = state.get("itinerary", "").strip()
    if not final:
        final = "The approved itinerary is unavailable. Please regenerate the trip."
    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "llm_calls": 0,
    }


ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "research_synthesis": "research_synthesis",
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
    return selected or "research_synthesis"


def route_after_review(state: TravelState) -> str:
    return "final_agent" if state.get("approved", False) else "itinerary_agent"


builder = StateGraph(TravelState)
for name, fn in [
    ("supervisor", supervisor_agent),
    ("guardrail_blocked", guardrail_blocked_agent),
    ("flight_agent", flight_agent),
    ("hotel_agent", hotel_agent),
    ("weather_agent", weather_agent),
    ("budget_agent", budget_agent),
    ("research_synthesis", research_synthesis_agent),
    ("itinerary_agent", itinerary_agent),
    ("human_approval", human_approval_agent),
    ("final_agent", final_agent),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
for agent in SPECIALIST_AGENTS:
    builder.add_edge(agent, "research_synthesis")
builder.add_edge("research_synthesis", "itinerary_agent")
builder.add_edge("itinerary_agent", "human_approval")
builder.add_conditional_edges(
    "human_approval", route_after_review,
    {"itinerary_agent": "itinerary_agent", "final_agent": "final_agent"},
)
builder.add_edge("final_agent", END)
builder.add_edge("guardrail_blocked", END)

_schema_ready = False
_schema_lock = threading.Lock()


def _ensure_checkpoint_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        with psycopg.connect(get_database_url(), autocommit=True) as conn:
            PostgresSaver(conn).setup()
        _schema_ready = True


def get_travel_graph():
    _ensure_checkpoint_schema()
    conn = psycopg.connect(get_database_url(), autocommit=True)
    return builder.compile(checkpointer=PostgresSaver(conn)), conn


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
    answer = (
        payload.get("draft_itinerary") if payload
        else result.get("final_response") or last_message
    )
    return {
        "thread_id": thread_id,
        "answer": answer or result.get("itinerary", ""),
        "requires_approval": payload is not None,
        "approval_request": payload.get("approval_request", "") if payload else result.get("approval_request", ""),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "research_summary": result.get("research_summary", ""),
        "itinerary": result.get("itinerary", ""),
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
                "flight_results": "", "hotel_results": "", "weather_results": "",
                "budget_results": "", "research_summary": "", "itinerary": "", "approval_request": "",
                "approved": False, "human_feedback": "", "approval_round": 0,
                "final_response": "", "llm_calls": 0,
            },
            config={"configurable": {"thread_id": thread_id}},
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
    try:
        result = graph.invoke(
            Command(resume={"approved": bool(approved), "feedback": feedback}),
            config={"configurable": {"thread_id": thread_id}},
        )
        return _serialize_result(result, thread_id)
    finally:
        conn.close()
