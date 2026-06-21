"""
graph.py — Crop Disease Diagnostic App
=======================================
ADK 2.0 graph-based multi-agent pipeline with three nodes:

    START
      │
      ▼
  [VisionProcessor]        ← Node 1: Analyses the image (placeholder or Gemini)
      │  state["vision_result"]
      ▼
  [KnowledgeBaseAgent]     ← Node 2: Looks up agricultural remedies
      │  state["remedy_result"]
      ▼
  [ResponseGenerator]      ← Node 3: Compiles a Markdown farm advisory
      │
      ▼
     END

Shared graph state (session.state dict) is the data bus between nodes.
Each node writes its output to a state key; the next reads it via
{placeholder} injection in its instruction string.

Usage
-----
  # With a real image (requires GOOGLE_API_KEY in .env or environment):
  python graph.py path/to/diseased_leaf.jpg

  # With crop type hint:
  python graph.py path/to/leaf.jpg --crop tomato

  # Placeholder/demo mode (no image needed, no API key required):
  python graph.py --demo

  # Demo mode with a specific disease to simulate:
  python graph.py --demo --disease late_blight
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import warnings

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set GOOGLE_API_KEY in shell instead

# ── ADK 2.0 imports ───────────────────────────────────────────────────────────
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types as genai_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crop_diagnostic")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DISEASE KNOWLEDGE BASE
# A curated inline dictionary: disease_key → remedy data.
# KnowledgeBaseAgent's tool queries this at runtime.
# ══════════════════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE: dict[str, dict] = {
    "late_blight": {
        "display_name": "Late Blight (Phytophthora infestans)",
        "urgency": "CRITICAL",
        "organic_remedies": [
            "Copper-based fungicide (copper hydroxide) — spray every 7 days",
            "Remove and destroy all infected plant material immediately",
            "Neem oil spray (5 mL/L) as preventive measure",
        ],
        "chemical_treatments": [
            "Mancozeb 2.5 g/L — apply every 10–14 days",
            "Metalaxyl + Mancozeb (Ridomil Gold) — systemic, highly effective",
        ],
        "schedule": "Begin at first sign; reapply every 7–10 days or after heavy rain",
        "prevention": [
            "Use certified disease-free seed/transplants",
            "Avoid overhead irrigation; use drip systems",
            "Rotate crops — avoid solanaceous crops in same field for 3 years",
        ],
        "yield_impact": "50–100% loss within 1–2 weeks if untreated",
    },
    "early_blight": {
        "display_name": "Early Blight (Alternaria solani)",
        "urgency": "HIGH",
        "organic_remedies": [
            "Copper octanoate fungicide spray",
            "Baking soda spray — 1 tbsp per litre of water",
            "Remove lower infected leaves promptly",
        ],
        "chemical_treatments": [
            "Chlorothalonil 75 WP (2 g/L) — broad-spectrum protectant",
            "Azoxystrobin (Amistar) — systemic, excellent efficacy",
        ],
        "schedule": "Begin 2 weeks after transplanting; every 7–10 days",
        "prevention": [
            "Mulch around plants to prevent soil splash",
            "Maintain balanced nitrogen fertilisation",
        ],
        "yield_impact": "30–60% loss if untreated within 2 weeks",
    },
    "powdery_mildew": {
        "display_name": "Powdery Mildew (Erysiphe spp.)",
        "urgency": "MEDIUM",
        "organic_remedies": [
            "Potassium bicarbonate spray (5 g/L)",
            "Dilute milk spray — 40% milk, 60% water (proven efficacy)",
            "Sulfur dust or wettable sulfur spray",
        ],
        "chemical_treatments": [
            "Myclobutanil (Eagle) — systemic triazole",
            "Propiconazole (Banner Maxx) — excellent systemic control",
        ],
        "schedule": "Apply at first sign; repeat every 10–14 days",
        "prevention": [
            "Ensure good air circulation; avoid dense planting",
            "Select resistant varieties",
        ],
        "yield_impact": "10–30% loss possible if disease spreads",
    },
    "mosaic_virus": {
        "display_name": "Mosaic Virus (TMV / CMV)",
        "urgency": "CRITICAL",
        "organic_remedies": [
            "No chemical cure — remove and destroy infected plants immediately",
            "Control aphid vectors with insecticidal soap or neem oil",
            "Reflective mulches to repel aphids",
        ],
        "chemical_treatments": [
            "Imidacloprid (Confidor) to control aphid vectors",
            "Mineral oil sprays to reduce virus transmission",
        ],
        "schedule": "Immediate removal of infected plants; vector control every 7 days",
        "prevention": [
            "Use virus-free certified seeds",
            "Disinfect tools with 10% bleach between plants",
        ],
        "yield_impact": "50%+ loss; plants may become unproductive",
    },
    "rust": {
        "display_name": "Rust (Puccinia spp.)",
        "urgency": "HIGH",
        "organic_remedies": [
            "Sulfur-based fungicide spray",
            "Neem oil (2%) as preventive",
            "Remove and destroy severely infected leaves",
        ],
        "chemical_treatments": [
            "Tebuconazole (Folicur) — triazole, highly effective",
            "Azoxystrobin + Propiconazole (Quilt Xcel) — broad-spectrum",
        ],
        "schedule": "At first sign of pustules; repeat every 14–21 days",
        "prevention": [
            "Plant rust-resistant varieties",
            "Monitor regularly during warm, humid weather",
        ],
        "yield_impact": "30–60% yield reduction in severe outbreaks",
    },
    "leaf_curl": {
        "display_name": "Leaf Curl Disease (TYLCV)",
        "urgency": "CRITICAL",
        "organic_remedies": [
            "No direct cure — manage whitefly vector aggressively",
            "Yellow sticky traps for whitefly monitoring",
            "Neem oil (5 mL/L) to deter whiteflies",
        ],
        "chemical_treatments": [
            "Imidacloprid (Confidor) soil drench for whitefly control",
            "Spiromesifen (Oberon) — effective against whitefly nymphs",
        ],
        "schedule": "Weekly vector control; remove infected plants immediately",
        "prevention": [
            "Use TYLCV-resistant tomato hybrids",
            "Use insect-proof nets in nursery phase",
        ],
        "yield_impact": "Near-zero fruit set in severe infections",
    },
    "anthracnose": {
        "display_name": "Anthracnose (Colletotrichum spp.)",
        "urgency": "HIGH",
        "organic_remedies": [
            "Copper-based fungicide",
            "Trichoderma harzianum biofungicide",
        ],
        "chemical_treatments": [
            "Azoxystrobin (Amistar) — excellent systemic control",
            "Carbendazim (Bavistin) — pre and post-harvest",
        ],
        "schedule": "Begin at flowering; repeat every 14 days until harvest",
        "prevention": [
            "Prune dead wood; maintain canopy airflow",
            "Sanitise harvesting equipment between uses",
        ],
        "yield_impact": "Significant post-harvest losses without treatment",
    },
    "healthy": {
        "display_name": "Healthy Plant",
        "urgency": "NONE",
        "organic_remedies": ["No treatment required"],
        "chemical_treatments": ["No treatment required"],
        "schedule": "N/A",
        "prevention": [
            "Maintain regular monitoring schedule",
            "Practice integrated pest management (IPM)",
            "Ensure balanced soil health and fertilisation",
        ],
        "yield_impact": "No yield impact — plant appears healthy",
    },
}

# Common name aliases → knowledge base keys
_ALIASES: dict[str, str] = {
    "late blight": "late_blight", "phytophthora": "late_blight",
    "early blight": "early_blight", "alternaria": "early_blight",
    "powdery mildew": "powdery_mildew", "mildew": "powdery_mildew",
    "mosaic": "mosaic_virus", "mosaic virus": "mosaic_virus",
    "rust": "rust", "leaf rust": "rust",
    "leaf curl": "leaf_curl", "tylcv": "leaf_curl",
    "anthracnose": "anthracnose", "colletotrichum": "anthracnose",
    "healthy": "healthy", "normal": "healthy", "no disease": "healthy",
}


def _lookup(disease_name: str) -> dict:
    """Resolve a disease name to knowledge base entry."""
    key = disease_name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[key]
    alias_key = _ALIASES.get(key.replace("_", " "))
    if alias_key:
        return KNOWLEDGE_BASE[alias_key]
    # Partial substring match
    for alias, db_key in _ALIASES.items():
        if alias in key.replace("_", " ") or key.replace("_", " ") in alias:
            return KNOWLEDGE_BASE[db_key]
    return KNOWLEDGE_BASE["healthy"]  # safe default


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PLACEHOLDER IMAGE ANALYSER
# When GOOGLE_API_KEY is absent or --demo flag is used, this function
# returns mock VisionProcessor output so the entire pipeline can run
# without a live API call.
# ══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER_ANALYSES: dict[str, dict] = {
    "late_blight": {
        "disease_name": "Late Blight", "disease_key": "late_blight",
        "confidence": 0.91, "severity": "high", "crop_type": "tomato",
        "affected_area_percent": 45,
        "primary_symptoms": ["dark water-soaked lesions on leaves",
                             "white mold on leaf underside",
                             "rapid necrosis spreading from leaf margins"],
        "visual_evidence": (
            "Large, irregular dark-brown lesions observed on multiple leaves "
            "with characteristic white sporulation on the underside. Pattern "
            "consistent with Phytophthora infestans."
        ),
        "immediate_action_required": True,
    },
    "powdery_mildew": {
        "disease_name": "Powdery Mildew", "disease_key": "powdery_mildew",
        "confidence": 0.87, "severity": "medium", "crop_type": "wheat",
        "affected_area_percent": 25,
        "primary_symptoms": ["white powdery coating on upper leaf surface",
                             "mild leaf curl at edges"],
        "visual_evidence": (
            "Distinctive white powdery patches on the adaxial (upper) leaf surface, "
            "concentrated along the midrib. No visible sporulation on underside."
        ),
        "immediate_action_required": False,
    },
    "healthy": {
        "disease_name": "Healthy", "disease_key": "healthy",
        "confidence": 0.95, "severity": "low", "crop_type": "tomato",
        "affected_area_percent": 0,
        "primary_symptoms": ["no visible disease symptoms"],
        "visual_evidence": "Uniform green coloration with no lesions, spots, or abnormal growth.",
        "immediate_action_required": False,
    },
}

_DEFAULT_PLACEHOLDER = _PLACEHOLDER_ANALYSES["late_blight"]


def build_placeholder_analysis(disease_key: str | None = None) -> str:
    """
    Returns a JSON string simulating VisionProcessor output.
    Used in --demo mode or when no API key is present.
    """
    template = _PLACEHOLDER_ANALYSES.get(disease_key or "", _DEFAULT_PLACEHOLDER)
    result = {**template, "analyzed_at": datetime.now(timezone.utc).isoformat(), "mode": "placeholder"}
    return json.dumps(result, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADK 2.0 TOOL FUNCTIONS
# Tools are plain Python functions injected with ToolContext by the ADK runtime.
# ToolContext.state gives read/write access to the shared graph state dict.
# ══════════════════════════════════════════════════════════════════════════════

def save_vision_result(analysis_json: str, tool_context: ToolContext) -> str:
    """
    NODE 1 TOOL — Persist VisionProcessor output to shared graph state.

    Writes:
      state["vision_result"]         — full JSON string
      state["detected_disease_key"]  — e.g. "late_blight"
      state["detected_crop_type"]    — e.g. "tomato"
      state["diagnosis_severity"]    — "low" | "medium" | "high"
      state["diagnosis_confidence"]  — float 0.0–1.0

    These keys are consumed by KnowledgeBaseAgent via {placeholder} injection.
    """
    try:
        data = json.loads(analysis_json)
    except json.JSONDecodeError:
        # Graceful fallback: store raw string so pipeline continues
        tool_context.state["vision_result"] = analysis_json
        logger.warning("VisionProcessor returned non-JSON; storing raw text.")
        return "Stored raw vision output (non-JSON)."

    # Enrich and persist
    data.setdefault("analyzed_at", datetime.now(timezone.utc).isoformat())
    tool_context.state["vision_result"]        = json.dumps(data, indent=2)
    tool_context.state["detected_disease_key"] = data.get("disease_key", "unknown")
    tool_context.state["detected_crop_type"]   = data.get("crop_type", "unknown")
    tool_context.state["diagnosis_severity"]   = data.get("severity", "unknown")
    tool_context.state["diagnosis_confidence"] = data.get("confidence", 0.0)

    logger.info(
        "Graph state updated by VisionProcessor — disease='%s', severity='%s', confidence=%.0f%%",
        data.get("disease_key"), data.get("severity"), data.get("confidence", 0) * 100,
    )
    return (
        f"Vision result saved. Disease: {data.get('disease_name', 'Unknown')} | "
        f"Severity: {data.get('severity')} | Confidence: {data.get('confidence', 0):.0%}"
    )


def lookup_agricultural_remedies(
    disease_key: str,
    crop_type: str,
    tool_context: ToolContext,
) -> str:
    """
    NODE 2 TOOL — Query the agricultural knowledge base for remedy data.

    Args:
        disease_key: The snake_case disease identifier from VisionProcessor output
                     (e.g. "late_blight", "powdery_mildew").
        crop_type:   Detected or user-supplied crop (e.g. "tomato", "wheat").
        tool_context: Auto-injected by ADK runtime.

    Writes:
      state["remedy_result"]   — JSON remedy data string
      state["remedy_urgency"]  — e.g. "CRITICAL"

    Returns:
        JSON string of remedy data (also consumed by ReportAgent via state).
    """
    logger.info("KnowledgeBaseAgent querying KB: disease='%s', crop='%s'", disease_key, crop_type)
    data = _lookup(disease_key)

    payload = {
        "disease_display_name": data["display_name"],
        "urgency":              data["urgency"],
        "organic_remedies":     data["organic_remedies"],
        "chemical_treatments":  data["chemical_treatments"],
        "application_schedule": data["schedule"],
        "prevention_measures":  data["prevention"],
        "yield_impact":         data["yield_impact"],
        "queried_at":           datetime.now(timezone.utc).isoformat(),
        "data_source":          "CropDiseaseKB v2.0",
    }

    result_json = json.dumps(payload, indent=2)
    tool_context.state["remedy_result"] = result_json
    tool_context.state["remedy_urgency"] = data["urgency"]

    logger.info("Remedy data written to state — urgency='%s'", data["urgency"])
    return result_json


def get_report_timestamp(tool_context: ToolContext) -> str:
    """
    NODE 3 TOOL — Returns a human-readable UTC timestamp for the report header.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tool_context.state["report_timestamp"] = ts
    return ts


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AGENT INSTRUCTION STRINGS
# Kept here (not in a separate file) so graph.py is fully self-contained.
# {vision_result} and {remedy_result} are ADK 2.0 state-injection placeholders —
# the runtime replaces them with session.state values before calling the LLM.
# ══════════════════════════════════════════════════════════════════════════════

_VISION_INSTRUCTION = """
You are VisionProcessor, an expert agricultural pathologist.
Your job is to analyse the crop image provided and produce a structured JSON diagnosis.

IMPORTANT: You MUST call `save_vision_result` with your JSON output to persist it
to the shared graph state so downstream agents can access it.

Output a single JSON object — no markdown fences, no extra text — in this exact schema:
{
  "disease_name":            "<canonical disease name, e.g. 'Late Blight' or 'Healthy'>",
  "disease_key":             "<snake_case, e.g. 'late_blight' or 'healthy'>",
  "confidence":              <float 0.0–1.0>,
  "severity":                "<low|medium|high>",
  "crop_type":               "<detected crop or 'unknown'>",
  "affected_area_percent":   <int 0–100>,
  "primary_symptoms":        ["<symptom 1>", "<symptom 2>"],
  "visual_evidence":         "<1–2 sentences describing exactly what you see>",
  "immediate_action_required": <true|false>
}

Allowed disease_key values:
  late_blight | early_blight | powdery_mildew | mosaic_virus | rust |
  leaf_curl | anthracnose | healthy

Rules:
- Be clinically objective. Farmer livelihoods depend on this diagnosis.
- If the plant is healthy, use disease_key "healthy".
- Output ONLY the JSON, then call save_vision_result.
"""

_REMEDY_INSTRUCTION = """
You are KnowledgeBaseAgent, an agricultural consultant specialising in plant disease management.

The VisionProcessor has already analysed the crop image. Here is its output:
{vision_result}

Your task:
1. Extract the `disease_key` and `crop_type` from the vision result above.
2. Call `lookup_agricultural_remedies` with those two values.
3. Return the tool output verbatim — do not paraphrase or omit any data.

Do not invent remedies. Only use data returned by the tool.
"""

_REPORT_INSTRUCTION = """
You are ResponseGenerator, a specialist in writing clear, actionable farm advisories.

You have two data sources from the pipeline:

VISION ANALYSIS:
{vision_result}

REMEDY PLAN:
{remedy_result}

Call `get_report_timestamp` first to get the current timestamp, then write a comprehensive
Markdown report using EXACTLY this structure:

---
# Crop Disease Diagnostic Report
**Generated:** [timestamp from tool] | **Engine:** ADK 2.0 SequentialAgent

## Diagnosis Summary
| Field | Value |
|-------|-------|
| Disease | [disease name] |
| Crop | [crop type] |
| Confidence | [X]% |
| Severity | [Low / Medium / High with emoji: 🟢 / 🟡 / 🔴] |
| Area Affected | [X]% of visible plant |
| Immediate Action | [Yes / No] |

## What Was Observed
[2–3 plain-English sentences describing the visual symptoms, accessible to a farmer
without an agricultural degree]

## Risk Assessment
[1 paragraph: what happens if left untreated, including yield impact estimate]

## Organic / Low-Input Treatment Plan
[Numbered list with how-to notes for each remedy]

## Conventional Treatment Plan
[Numbered list with product names, dosage, and a brief safety note]

## Application Schedule
[When to start, how often, any weather conditions to consider]

## Prevention for Next Season
[Numbered list]

## Urgency Level: [NONE / LOW / MEDIUM / HIGH / CRITICAL]
[One sentence telling the farmer exactly what to do first]

---
*Generated by CropDiagnosticAI | Powered by Google ADK 2.0 & Gemini 2.5 Flash*

Rules:
- Write for farmers, not scientists. Use simple, empathetic language.
- Include exact product names and dosages from the remedy plan.
- Add a safety disclaimer paragraph after the Conventional Treatment section.
- Do NOT invent treatments not present in the remedy plan.
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GRAPH DEFINITION
# Three LlmAgent nodes wired into an ADK 2.0 SequentialAgent.
# SequentialAgent executes sub_agents in order, sharing one InvocationContext
# and one session.state dict — the data bus of the graph.
# ══════════════════════════════════════════════════════════════════════════════

# ── Node 1: VisionProcessor ───────────────────────────────────────────────────
vision_processor = LlmAgent(
    name="VisionProcessor",
    model="gemini-2.5-flash",
    description=(
        "Multimodal node. Accepts a crop image (or placeholder analysis text), "
        "identifies the disease, severity, and symptoms, then writes a structured "
        "JSON diagnosis to shared graph state via save_vision_result."
    ),
    instruction=_VISION_INSTRUCTION,
    tools=[save_vision_result],
    output_key="vision_result",   # ADK also auto-saves agent text response to state
)

# ── Node 2: KnowledgeBaseAgent ────────────────────────────────────────────────
knowledge_base_agent = LlmAgent(
    name="KnowledgeBaseAgent",
    model="gemini-2.5-flash",
    description=(
        "Knowledge retrieval node. Reads vision_result from graph state, "
        "queries the agricultural disease knowledge base, and writes structured "
        "remedy data to state via lookup_agricultural_remedies."
    ),
    instruction=_REMEDY_INSTRUCTION,   # {vision_result} is injected by ADK runtime
    tools=[lookup_agricultural_remedies],
    output_key="remedy_result",
)

# ── Node 3: ResponseGenerator ─────────────────────────────────────────────────
response_generator = LlmAgent(
    name="ResponseGenerator",
    model="gemini-2.5-flash",
    description=(
        "Report synthesis node. Reads vision_result and remedy_result from "
        "graph state, then compiles a comprehensive Markdown farm advisory report."
    ),
    instruction=_REPORT_INSTRUCTION,   # {vision_result} and {remedy_result} injected
    tools=[get_report_timestamp],
)

# ── Root graph: SequentialAgent ───────────────────────────────────────────────
# SequentialAgent is the graph-based workflow orchestrator in ADK 2.3.0.
# NOTE: ADK 2.3.0 emits a deprecation warning suggesting `Workflow`, but
# `Workflow` is not yet available in the installed package. SequentialAgent
# is fully functional — the warning is suppressed below.
# Graph edges (implicit): VisionProcessor → KnowledgeBaseAgent → ResponseGenerator
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    graph = SequentialAgent(
    name="CropDiseaseGraph",
    description=(
        "End-to-end crop disease diagnostic graph. "
        "Input: crop image (or placeholder). "
        "Output: Markdown farm advisory report."
    ),
        sub_agents=[vision_processor, knowledge_base_agent, response_generator],
    )

logger.info(
    "Graph initialised: %s  [%s]",
    graph.name,
    " → ".join(a.name for a in graph.sub_agents),
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PIPELINE RUNNER
# Builds the ADK Runner + InMemorySessionService, constructs the multimodal
# Content message, and streams events from the graph to stdout.
# ══════════════════════════════════════════════════════════════════════════════

APP_NAME = "crop_disease_diagnostic"


async def run_pipeline(
    image_path: str | None,
    crop_type: str = "",
    demo_mode: bool = False,
    demo_disease: str = "late_blight",
) -> str:
    """
    Execute the CropDiseaseGraph pipeline and return the final Markdown report.

    Args:
        image_path:    Path to the crop image file. None in demo mode.
        crop_type:     Optional crop type hint (e.g. "tomato").
        demo_mode:     If True, inject placeholder analysis instead of loading image.
        demo_disease:  Disease key to simulate in demo mode.

    Returns:
        The Markdown report string from ResponseGenerator.
    """
    session_service = InMemorySessionService()

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="cli_user",
        session_id="session_001",
        state={"crop_type": crop_type or "unknown"},
    )

    runner = Runner(
        agent=graph,
        app_name=APP_NAME,
        session_service=session_service,
    )

    # ── Build the user message ─────────────────────────────────────────────
    if demo_mode:
        # Placeholder mode: inject a simulated vision analysis as text
        placeholder_json = build_placeholder_analysis(demo_disease)
        print(f"\n[DEMO MODE] Simulating disease: {demo_disease}")
        print("[DEMO MODE] VisionProcessor will receive placeholder analysis.\n")

        user_message = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(text=(
                    f"[PLACEHOLDER MODE — no real image]\n\n"
                    f"A previous vision step produced this analysis JSON. "
                    f"Treat it as the definitive VisionProcessor output, call "
                    f"save_vision_result with it, and continue the pipeline:\n\n"
                    f"{placeholder_json}\n\n"
                    f"Crop type hint: {crop_type or 'auto-detected above'}"
                ))
            ],
        )
    else:
        # Real image mode: load image bytes and pass as multimodal Content
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png",  ".webp": "image/webp"}
        mime = suffix_map.get(img.suffix.lower(), "image/jpeg")
        image_bytes = img.read_bytes()

        print(f"\n[IMAGE] {img.name} ({len(image_bytes) / 1024:.1f} KB, {mime})")

        user_message = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(text=(
                    f"Analyse this crop image for disease. "
                    f"Crop type hint: {crop_type or 'please detect from image'}. "
                    f"Provide a complete diagnosis and save it to state."
                )),
                genai_types.Part(
                    inline_data=genai_types.Blob(mime_type=mime, data=image_bytes)
                ),
            ],
        )

    # ── Stream the graph execution ─────────────────────────────────────────
    print("=" * 60)
    print(f"  CROP DISEASE DIAGNOSTIC — ADK 2.0 GRAPH PIPELINE")
    print("=" * 60)
    print(f"  Nodes: {' → '.join(a.name for a in graph.sub_agents)}")
    print("=" * 60)

    final_report_parts: list[str] = []
    current_node: str = ""

    async for event in runner.run_async(
        user_id="cli_user",
        session_id="session_001",
        new_message=user_message,
    ):
        author = getattr(event, "author", None) or ""

        # ── Node transition banner ──────────────────────────────────────
        if author and author != current_node and author != APP_NAME:
            current_node = author
            icons = {
                "VisionProcessor":    "🔬",
                "KnowledgeBaseAgent": "🌿",
                "ResponseGenerator":  "📋",
            }
            print(f"\n{icons.get(author, '⚙️')}  [{author}] running...")

        # ── Tool call notification ──────────────────────────────────────
        if hasattr(event, "get_function_calls"):
            for call in (event.get_function_calls() or []):
                print(f"    🔧 tool → {call.name}()")

        # ── State update notification ───────────────────────────────────
        if hasattr(event, "actions") and event.actions:
            delta = getattr(event.actions, "state_delta", None) or {}
            for key in delta:
                if key in ("vision_result", "remedy_result",
                           "detected_disease_key", "remedy_urgency"):
                    print(f"    💾 state['{key}'] updated")

        # ── Stream text output ──────────────────────────────────────────
        if event.content and event.content.parts:
            for part in event.content.parts:
                text = getattr(part, "text", "") or ""
                if text and author == "ResponseGenerator":
                    final_report_parts.append(text)
                    # Live stream the report as it's generated
                    print(text, end="", flush=True)

        # ── Per-node completion ─────────────────────────────────────────
        if event.is_final_response() and author:
            print(f"\n    ✅ {author} complete")

    # ── Fetch final state summary ──────────────────────────────────────────
    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id="cli_user", session_id="session_001"
    )
    if final_session:
        state = final_session.state
        print("\n" + "=" * 60)
        print("  GRAPH STATE SUMMARY")
        print("=" * 60)
        for key in ("detected_disease_key", "diagnosis_severity",
                    "diagnosis_confidence", "remedy_urgency"):
            if key in state:
                val = state[key]
                if isinstance(val, float):
                    val = f"{val:.0%}"
                print(f"  {key:<28} {val}")
        print("=" * 60 + "\n")

    return "".join(final_report_parts)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python graph.py",
        description=textwrap.dedent("""\
            Crop Disease Diagnostic App -- ADK 2.0 Graph Pipeline
            -------------------------------------------------------
            Three nodes: VisionProcessor -> KnowledgeBaseAgent -> ResponseGenerator
            Shared graph state passes data between nodes automatically.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Analyse a real crop image (requires GOOGLE_API_KEY):
              python graph.py path/to/leaf.jpg

              # With a crop type hint:
              python graph.py path/to/leaf.jpg --crop tomato

              # Demo / placeholder mode (no image, no API key needed):
              python graph.py --demo

              # Demo with a specific disease simulation:
              python graph.py --demo --disease powdery_mildew

              # Save report to file:
              python graph.py --demo --output report.md

            Available demo diseases:
              late_blight | early_blight | powdery_mildew |
              mosaic_virus | rust | leaf_curl | anthracnose | healthy
        """),
    )

    parser.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE_PATH",
        help="Path to the crop image (JPEG / PNG / WEBP). Omit when using --demo.",
    )
    parser.add_argument(
        "--crop", "-c",
        metavar="CROP_TYPE",
        default="",
        help="Crop type hint, e.g. 'tomato', 'wheat', 'rice'. Optional.",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run in placeholder/demo mode — no real image or API key required.",
    )
    parser.add_argument(
        "--disease",
        metavar="DISEASE_KEY",
        default="late_blight",
        help="Disease to simulate in --demo mode (default: late_blight).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Save the Markdown report to this file path.",
    )
    parser.add_argument(
        "--list-diseases",
        action="store_true",
        help="Print all diseases in the knowledge base and exit.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── --list-diseases shortcut ───────────────────────────────────────────
    if args.list_diseases:
        print("\nCropDiseaseKB v2.0 — Supported Diseases\n")
        for key, data in KNOWLEDGE_BASE.items():
            print(f"  {key:<25}  {data['display_name']}  [{data['urgency']}]")
        print()
        sys.exit(0)

    # ── Validate args ──────────────────────────────────────────────────────
    if not args.demo and not args.image:
        parser.error(
            "Provide an IMAGE_PATH or use --demo for placeholder mode.\n"
            "  Example: python graph.py leaf.jpg\n"
            "  Example: python graph.py --demo"
        )

    if args.image and not args.demo:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(
                "\n⚠️  WARNING: GOOGLE_API_KEY is not set.\n"
                "   VisionProcessor will not be able to call Gemini.\n"
                "   → Set it in .env or run with --demo for placeholder mode.\n"
            )

    # ── Run the async pipeline ─────────────────────────────────────────────
    try:
        report = asyncio.run(
            run_pipeline(
                image_path=args.image,
                crop_type=args.crop,
                demo_mode=args.demo,
                demo_disease=args.disease,
            )
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.\n")
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        print(f"\n❌ Pipeline failed: {e}\n", file=sys.stderr)
        sys.exit(1)

    # ── Save report to file if requested ──────────────────────────────────
    if args.output and report:
        out = Path(args.output)
        out.write_text(report, encoding="utf-8")
        print(f"📄 Report saved to: {out.resolve()}\n")


if __name__ == "__main__":
    main()
