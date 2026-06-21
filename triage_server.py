"""
triage_server.py
Standalone Healthcare Triage Assistant Server
Serves only the /triage feature — no crop disease or learning path dependencies.
No patient data is persisted; all processing is ephemeral and local.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from run_triage import (
    classify_triage,
    generate_triage_report,
    TRIAGE_QUICK_SCENARIOS,
    DISCLAIMER as TRIAGE_DISCLAIMER,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI app setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AuraTriage — Healthcare Triage Assistant",
    description="AI-powered healthcare symptom triage assistant for India",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────
def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────
# Triage streaming pipeline
# ─────────────────────────────────────────────────────────────
async def run_triage_streaming(
    messages: list,
    location: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Simulates a multi-agent Healthcare Triage pipeline and yields SSE events.
    No patient data is persisted — all processing is ephemeral and local.
    """
    yield _sse_event({
        "type": "pipeline_start",
        "message": "🚀 Starting Healthcare Triage Pipeline...",
        "session_id": session_id,
        "disclaimer": TRIAGE_DISCLAIMER,
    })
    await asyncio.sleep(0.3)

    # 1. SymptomCollector Agent
    yield _sse_event({"type": "agent_start", "agent": "SymptomCollector", "emoji": "🩺",
                      "message": "Collecting and structuring reported symptoms..."})
    await asyncio.sleep(0.8)
    yield _sse_event({"type": "tool_call", "agent": "SymptomCollector", "tool": "extract_symptoms",
                      "message": "🔧 Calling tool: extract_symptoms"})
    await asyncio.sleep(0.5)
    yield _sse_event({"type": "state_update", "key": "symptom_profile",
                      "message": "💾 State updated: symptom_profile"})
    yield _sse_event({"type": "agent_complete", "agent": "SymptomCollector",
                      "message": "✅ SymptomCollector completed"})
    await asyncio.sleep(0.3)

    # 2. TriageClassifier Agent
    yield _sse_event({"type": "agent_start", "agent": "TriageClassifier", "emoji": "🔬",
                      "message": "Classifying urgency level using clinical keyword analysis..."})
    await asyncio.sleep(0.8)
    yield _sse_event({"type": "tool_call", "agent": "TriageClassifier", "tool": "classify_urgency",
                      "message": "🔧 Calling tool: classify_urgency"})
    await asyncio.sleep(0.5)

    result = classify_triage(messages)
    yield _sse_event({"type": "state_update", "key": "triage_result",
                      "message": f"💾 State updated: triage_result (urgency={result.urgency})"})
    yield _sse_event({"type": "agent_complete", "agent": "TriageClassifier",
                      "message": "✅ TriageClassifier completed"})
    await asyncio.sleep(0.3)

    # 3. GuidanceAgent
    yield _sse_event({"type": "agent_start", "agent": "GuidanceAgent", "emoji": "📋",
                      "message": "Generating actionable next steps and facility recommendation..."})
    await asyncio.sleep(0.7)
    yield _sse_event({"type": "tool_call", "agent": "GuidanceAgent", "tool": "generate_guidance",
                      "message": "🔧 Calling tool: generate_guidance"})
    await asyncio.sleep(0.5)
    yield _sse_event({"type": "state_update", "key": "guidance_plan",
                      "message": "💾 State updated: guidance_plan"})
    yield _sse_event({"type": "agent_complete", "agent": "GuidanceAgent",
                      "message": "✅ GuidanceAgent completed"})
    await asyncio.sleep(0.3)

    # 4. ReportAgent — stream the report
    yield _sse_event({"type": "agent_start", "agent": "ReportAgent", "emoji": "✅",
                      "message": "Compiling final triage report with disclaimer..."})
    await asyncio.sleep(0.5)

    report_text = generate_triage_report(result, location)
    chunk_size = 100
    for i in range(0, len(report_text), chunk_size):
        chunk = report_text[i:i + chunk_size]
        yield _sse_event({"type": "text_chunk", "agent": "ReportAgent", "chunk": chunk})
        await asyncio.sleep(0.015)

    yield _sse_event({"type": "agent_complete", "agent": "ReportAgent",
                      "message": "✅ ReportAgent completed"})

    yield _sse_event({
        "type": "pipeline_complete",
        "message": "✅ Triage assessment complete.",
        "urgency": result.urgency,
        "urgency_emoji": result.urgency_emoji,
        "urgency_color": result.urgency_color,
        "matched_symptoms": result.matched_symptoms,
        "reasoning": result.reasoning,
        "next_steps": result.next_steps,
        "facility_type": result.facility_type,
        "call_to_action": result.call_to_action,
        "disclaimer": TRIAGE_DISCLAIMER,
        "report": report_text,
    })
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_triage_ui():
    """Serve the Healthcare Triage Assistant web UI."""
    triage_path = Path(__file__).parent / "web_ui" / "triage.html"
    if not triage_path.exists():
        raise HTTPException(status_code=404, detail="triage.html not found")
    return HTMLResponse(content=triage_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AuraTriage Healthcare Triage Assistant",
        "version": "1.0.0",
        "mode": "offline-keyword-based",
    }


@app.post("/triage/assess")
async def assess_triage(payload: dict):
    """
    Healthcare triage endpoint.
    Payload: { "messages": ["symptom text..."], "location": "optional location" }
    Returns SSE stream of agent events + final triage result.
    NOTE: No patient data is stored or logged beyond this request lifecycle.
    """
    msgs = payload.get("messages", [])
    # Validate messages: must be non‑empty strings, each ≤ 500 characters
    if not msgs or not any(isinstance(m, str) and m.strip() for m in msgs):
        raise HTTPException(status_code=400, detail="At least one symptom message is required.")
    if any(len(m) > 500 for m in msgs):
        raise HTTPException(status_code=400, detail="Each symptom message must be ≤ 500 characters.")
    location = payload.get("location", "").strip()
    if len(location) > 200:
        raise HTTPException(status_code=400, detail="Location must be ≤ 200 characters.")
    session_id = str(uuid.uuid4())
    logger.info(f"[Triage] New assessment | session={session_id} | msgs={len(msgs)} | location='{location}'")
    return StreamingResponse(
        run_triage_streaming(msgs, location, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/triage/scenarios")
async def get_triage_scenarios():
    """Return the list of quick triage scenarios for the UI."""
    return {"scenarios": TRIAGE_QUICK_SCENARIOS}


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print("🚀 Starting AuraTriage Healthcare Triage Assistant...")
    print(f"   URL: http://localhost:{port}")
    print(f"   Health: http://localhost:{port}/health")
    uvicorn.run(
        "triage_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
