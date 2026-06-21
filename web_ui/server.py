# ============================================================
# web_ui/server.py
# FastAPI server wrapping the ADK 2.0 Crop Disease Diagnostic Agent
# Provides SSE streaming so the UI can show real-time agent progress
# ============================================================

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator

# Ensure project root is on sys.path so crop_disease_agent is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# # Crop disease and learning path imports removed
from run_triage import classify_triage, generate_triage_report, TRIAGE_QUICK_SCENARIOS, DISCLAIMER as TRIAGE_DISCLAIMER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI app setup
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AuraTriage Multi-Agent API",
    description="Unified multi-agent platform for healthcare triage, learning paths, and optional crop disease diagnostics.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the web UI static files
ui_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

# ─────────────────────────────────────────────────────────────
# ADK Session & Runner setup
# ─────────────────────────────────────────────────────────────
session_service = InMemorySessionService()
APP_NAME = "auratriage"
# Keep crop disease functionality optional; set to None if module unavailable


# ─────────────────────────────────────────────────────────────
# SSE Streaming helper
# ─────────────────────────────────────────────────────────────
# Crop disease pipeline removed

def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data event."""
    return f"data: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────
# Crop disease UI routes removed


@app.post("/diagnose/base64")
async def diagnose_crop_base64(payload: dict):
    """
    Alternative endpoint accepting base64-encoded image (useful for JS fetch without FormData).
    Payload: { "image_b64": "...", "mime_type": "image/jpeg", "crop_type": "tomato" }
    """
    raise HTTPException(status_code=501, detail="Crop disease diagnostic pipeline is currently disabled.")


# ─────────────────────────────────────────────────────────────# ─────────────────────────────────────────────────────────────
# Learning Path Routes (Disabled)
# ─────────────────────────────────────────────────────────────

@app.post("/learning-path/generate")
async def generate_learning_path(payload: dict):
    """Learning Path functionality has been removed."""
    raise HTTPException(status_code=501, detail="Learning Path feature is no longer available.")

@app.get("/learning-path", response_class=HTMLResponse)
async def serve_learning_path_ui():
    """Learning Path UI has been removed."""
    raise HTTPException(status_code=501, detail="Learning Path UI is no longer available.")
# ─────────────────────────────────────────────────────────────
LP_APP_NAME = "personalized_learning_path"

async def run_learning_path_streaming(
    prompt: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Runs the ADK 2.0 PersonalizedLearningPathAgent pipeline and yields SSE events
    for each agent step (AssessmentAgent, CurriculumAgent, TutorAgent).
    If GOOGLE_API_KEY is not set or is placeholder, streams a simulated execution.
    """
    user_id = f"student_{session_id[:8]}"
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "your_google_api_key_here":
        yield _sse_event({
            "type": "pipeline_start",
            "message": "🚀 Starting Learning Path Simulation (Offline Mode)...",
            "session_id": session_id,
        })
        await asyncio.sleep(0.5)

        # 1. AssessmentAgent
        yield _sse_event({
            "type": "agent_start",
            "agent": "AssessmentAgent",
            "emoji": "🔬",
            "message": "Analyzing learning goals, skill level, and learning style...",
        })
        await asyncio.sleep(1.0)
        yield _sse_event({
            "type": "tool_call",
            "agent": "AssessmentAgent",
            "tool": "save_student_assessment",
            "message": "🔧 Calling tool: save_student_assessment",
        })
        await asyncio.sleep(0.6)
        # Get dynamic or preset simulation data
        from run_learning_path import get_demo_data
        demo_data = get_demo_data(prompt)
        style = demo_data["style"]
        level = demo_data["level"]
        concepts = demo_data["concepts"]
        topics = demo_data["topics"]
        demo_report = demo_data["report"]
        
        yield _sse_event({
            "type": "state_update",
            "key": "student_assessment_result",
            "message": f"💾 State updated: student_assessment_result (style={style}, level={level})",
        })
        yield _sse_event({
            "type": "agent_complete",
            "agent": "AssessmentAgent",
            "message": "✅ AssessmentAgent completed",
        })
        await asyncio.sleep(0.5)

        # 2. CurriculumAgent
        yield _sse_event({
            "type": "agent_start",
            "agent": "CurriculumAgent",
            "emoji": "📚",
            "message": "Resolving prerequisites and sequencing curriculum modules...",
        })
        await asyncio.sleep(1.0)
        yield _sse_event({
            "type": "tool_call",
            "agent": "CurriculumAgent",
            "tool": "fetch_and_sequence_curriculum",
            "message": "🔧 Calling tool: fetch_and_sequence_curriculum",
        })
        await asyncio.sleep(0.6)
        
        yield _sse_event({
            "type": "state_update",
            "key": "sequenced_curriculum",
            "message": f"💾 State updated: sequenced_curriculum ({len(topics)} topics sequenced)",
        })
        yield _sse_event({
            "type": "agent_complete",
            "agent": "CurriculumAgent",
            "message": "✅ CurriculumAgent completed",
        })
        await asyncio.sleep(0.5)

        # 3. TutorAgent
        yield _sse_event({
            "type": "agent_start",
            "agent": "TutorAgent",
            "emoji": "👨‍🏫",
            "message": "Formulating tailored lesson plan and interactive exercises...",
        })
        await asyncio.sleep(1.0)
        yield _sse_event({
            "type": "tool_call",
            "agent": "TutorAgent",
            "tool": "save_progress_report",
            "message": "🔧 Calling tool: save_progress_report",
        })
        await asyncio.sleep(0.5)
        yield _sse_event({
            "type": "state_update",
            "key": "progress_report",
            "message": "💾 State updated: progress_report",
        })

        # Stream report text
        chunk_size = 120
        for i in range(0, len(demo_report), chunk_size):
            chunk = demo_report[i:i+chunk_size]
            yield _sse_event({
                "type": "text_chunk",
                "agent": "TutorAgent",
                "chunk": chunk,
            })
            await asyncio.sleep(0.02)

        yield _sse_event({
            "type": "agent_complete",
            "agent": "TutorAgent",
            "message": "✅ TutorAgent completed",
        })
        
        yield _sse_event({
            "type": "pipeline_complete",
            "message": "🎉 Personalized learning path report generated successfully!",
            "state_summary": {
                "assessed_learning_style": style,
                "assessed_skill_level": level,
                "assessed_concepts": concepts,
                "sequenced_topics": topics
            },
            "report": demo_report,
        })
        yield "data: [DONE]\n\n"
        return

    # Real execution mode (requires API key)
    session = await session_service.create_session(
        app_name=LP_APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={"student_input": prompt},
    )

    runner = Runner(
        agent=learning_path_agent,
        app_name=LP_APP_NAME,
        session_service=session_service,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[
            genai_types.Part(text=(
                f"Student Input: \"{prompt}\"\n\n"
                f"Please start the assessment, sequence the appropriate syllabus, "
                f"and deliver a customized lesson and progress report."
            ))
        ],
    )

    yield _sse_event({
        "type": "pipeline_start",
        "message": "🚀 Starting Personalized Learning Path Pipeline...",
        "session_id": session_id,
    })

    current_agent = None
    full_report = []

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message,
        ):
            agent_name = getattr(event, "author", None)

            if agent_name and agent_name != current_agent:
                current_agent = agent_name
                node_labels = {
                    "AssessmentAgent": ("🔬", "Analyzing student goals, skill levels, and learning style..."),
                    "CurriculumAgent": ("📚", "Fetching modules and sequencing prerequisites..."),
                    "TutorAgent": ("👨‍🏫", "Synthesizing personalized explanations & exercises..."),
                }
                emoji, label = node_labels.get(agent_name, ("⚙️", f"Executing {agent_name}..."))
                yield _sse_event({
                    "type": "agent_start",
                    "agent": agent_name,
                    "emoji": emoji,
                    "message": label,
                })

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        chunk = part.text
                        full_report.append(chunk)
                        yield _sse_event({
                            "type": "text_chunk",
                            "agent": agent_name or current_agent,
                            "chunk": chunk,
                        })

            if hasattr(event, "get_function_calls"):
                for call in event.get_function_calls() or []:
                    yield _sse_event({
                        "type": "tool_call",
                        "agent": agent_name or current_agent,
                        "tool": call.name,
                        "message": f"🔧 Calling tool: {call.name}",
                    })

            if hasattr(event, "actions") and event.actions:
                if hasattr(event.actions, "state_delta") and event.actions.state_delta:
                    for key in event.actions.state_delta:
                        if key in ("student_assessment_result", "sequenced_curriculum", "progress_report"):
                            yield _sse_event({
                                "type": "state_update",
                                "key": key,
                                "message": f"💾 State updated: {key}",
                            })

            if event.is_final_response() and agent_name:
                yield _sse_event({
                    "type": "agent_complete",
                    "agent": agent_name,
                    "message": f"✅ {agent_name} completed",
                })

        final_session = await session_service.get_session(
            app_name=LP_APP_NAME, user_id=user_id, session_id=session_id
        )
        state_summary = {}
        if final_session:
            for key in ("assessed_learning_style", "assessed_skill_level", "assessed_concepts"):
                if key in final_session.state:
                    state_summary[key] = final_session.state[key]
            if "sequenced_content" in final_session.state:
                try:
                    import json
                    seq = final_session.state["sequenced_content"]
                    if isinstance(seq, str):
                        seq = json.loads(seq)
                    state_summary["sequenced_topics"] = [t["key"] for t in seq]
                except Exception:
                    pass

        yield _sse_event({
            "type": "pipeline_complete",
            "message": "🎉 Learning Path report generated successfully!",
            "state_summary": state_summary,
            "report": "".join(full_report),
        })

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        yield _sse_event({
            "type": "error",
            "message": f"❌ Pipeline error: {str(e)}",
        })
    finally:
        yield "data: [DONE]\n\n"

@app.get("/learning-path", response_class=HTMLResponse)
async def serve_learning_path_ui():
    """Serve the personalized learning path web UI."""
    index_path = ui_dir / "learning_path.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="learning_path.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

@app.post("/learning-path/generate")
async def generate_learning_path(payload: dict):
    prompt = payload.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Student prompt cannot be empty.")
    session_id = str(uuid.uuid4())
    logger.info(f"[Server] New learning path request | session={session_id} | prompt='{prompt[:50]}'")
    return StreamingResponse(
        run_learning_path_streaming(prompt, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )



# ─────────────────────────────────────────────────────────────
# Healthcare Triage Routes
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

    # Run actual classification
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


@app.get("/triage", response_class=HTMLResponse)
async def serve_triage_ui():
    """Serve the Healthcare Triage Assistant web UI."""
    triage_path = ui_dir / "triage.html"
    if not triage_path.exists():
        raise HTTPException(status_code=404, detail="triage.html not found")
    return HTMLResponse(content=triage_path.read_text(encoding="utf-8"))


@app.post("/triage/assess")
async def assess_triage(payload: dict):
    """
    Healthcare triage endpoint.
    Payload: { "messages": ["symptom text..."], "location": "optional location" }
    Returns SSE stream of agent events + final triage result.
    NOTE: No patient data is stored or logged beyond this request lifecycle.
    """
    msgs = payload.get("messages", [])
    if not msgs or not any(m.strip() for m in msgs):
        raise HTTPException(status_code=400, detail="At least one symptom message is required.")
    location = payload.get("location", "").strip()
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
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  WARNING: GOOGLE_API_KEY not set. Set it in .env before running.")
    else:
        print(f"✅ GOOGLE_API_KEY found ({api_key[:8]}...)")

    print("🚀 Starting AuraTriage Multi-Agent Server...")
    print("   UI:    http://localhost:8000")
    print("   Docs:  http://localhost:8000/docs")

    uvicorn.run(
        "web_ui.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
