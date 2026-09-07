"""FastAPI application for TripMate AI."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tripmate")

app = FastAPI(
    title="TripMate AI",
    description="Parallel multi-agent travel planner with MCP, LangGraph and human approval.",
    version="3.1.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class TravelRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, max_length=200)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    approved: bool
    feedback: str = Field(default="", max_length=4000)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    from backend import run_travel_agent

    try:
        message = request_data.message.strip()
        if not message:
            return {"success": False, "error": "Message cannot be empty."}

        # Graph work includes synchronous LLM/MCP calls. Keep it out of the
        # FastAPI event loop so concurrent HTTP requests remain responsive.
        result = await run_in_threadpool(
            run_travel_agent,
            message,
            request_data.thread_id,
        )
        return {"success": True, **result}
    except Exception as exc:
        logger.exception("Travel planning failed")
        return {
            "success": False,
            "error": f"Travel planning failed: {type(exc).__name__}: {exc}",
        }


@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    from backend import resume_travel_agent

    try:
        feedback = request_data.feedback.strip()
        if not request_data.approved and not feedback:
            return {
                "success": False,
                "error": "Revision feedback is required when requesting changes.",
            }

        result = await run_in_threadpool(
            resume_travel_agent,
            request_data.thread_id,
            request_data.approved,
            feedback,
        )
        return {"success": True, **result}
    except Exception as exc:
        logger.exception("Approval handling failed")
        return {
            "success": False,
            "error": f"Approval handling failed: {type(exc).__name__}: {exc}",
        }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "TripMate AI",
        "version": app.version,
        "architecture": {
            "orchestration": "LangGraph",
            "specialists": "parallel fan-out",
            "tools": "MCP",
            "persistence": "PostgreSQL checkpoints",
            "approval": "LangGraph interrupt / HITL",
        },
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
