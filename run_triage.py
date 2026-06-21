# ============================================================
# run_triage.py
# Healthcare Triage Logic Module — Agents for Good Track
# Rule-based symptom matching for offline simulation mode
# NO patient data is persisted; all processing is ephemeral.
# ============================================================

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────

DISCLAIMER = (
    "⚕️ I am an AI assistant, not a doctor. This assessment is for informational "
    "purposes only and does not replace professional medical advice. "
    "If you are having a medical emergency, call emergency services (112 / 911) immediately."
)


@dataclass
class TriageResult:
    urgency: str          # "EMERGENCY" | "URGENT" | "ROUTINE"
    urgency_emoji: str    # 🔴 🟡 🟢
    urgency_color: str    # CSS color token
    matched_symptoms: List[str]
    reasoning: str
    next_steps: List[str]
    facility_type: str
    call_to_action: str
    disclaimer: str = DISCLAIMER


# ─────────────────────────────────────────────────────────────
# Symptom Keyword Rules
# Priority order: EMERGENCY > URGENT > ROUTINE
# ─────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS: List[Tuple[str, str]] = [
    # (regex pattern, symptom label)
    (r"\bchest\s*(pain|pressure|tightness|discomfort)\b", "Chest pain / pressure"),
    (r"\bheart\s*attack\b", "Suspected heart attack"),
    (r"\bcan'?t\s*breath|difficulty\s*breath|trouble\s*breath|shortness\s*of\s*breath\b", "Severe breathing difficulty"),
    (r"\bstroke\b", "Suspected stroke"),
    (r"\bface\s*drooping|arm\s*weakness|speech\s*difficult\b", "Stroke warning signs (FAST)"),
    (r"\bunconscious|not\s*breathing|stopped\s*breathing\b", "Unconsciousness / no breathing"),
    (r"\bsevere\s*bleed|heavy\s*bleed|uncontrolled\s*bleed\b", "Severe/uncontrolled bleeding"),
    (r"\bsuicid|self.harm|want\s*to\s*die|end\s*my\s*life\b", "Mental health emergency"),
    (r"\boverdos\b", "Overdose"),
    (r"\bseizure|convuls\b", "Seizure / convulsions"),
    (r"\bsevere\s*allergic|anaphylax|throat\s*swelling|can'?t\s*swallow\b", "Anaphylaxis / severe allergic reaction"),
    (r"\bpoisoning|ate\s*poison|swallowed\s*(bleach|chemical)\b", "Poisoning / toxic ingestion"),
    (r"\bsevere\s*head\s*(injury|trauma)|skull\b", "Severe head injury"),
    (r"\bsudden\s*(blindness|vision\s*loss|hearing\s*loss)\b", "Sudden sensory loss"),
    (r"\bsevere\s*burn|third.degree\s*burn\b", "Severe burns"),
    (r"\bnewborn.*not\s*breath|baby.*blue\b", "Newborn / infant breathing emergency"),
    (r"\bparalys|can'?t\s*move\s*(arm|leg|hand|body)\b", "Sudden paralysis"),
    (r"\bsevere\s*abdom|abdomen\s*rigid\b", "Severe abdominal emergency"),
]

URGENT_KEYWORDS: List[Tuple[str, str]] = [
    (r"\bhigh\s*fever|fever\s*(above|over)\s*10[23456789]|104|105|103\s*[°f]?\b", "High fever (≥103°F / 39.4°C)"),
    (r"\bfever.*child|child.*fever\b", "Fever in child"),
    (r"\bbroken\s*bone|fracture|sprain\b", "Possible fracture / sprain"),
    (r"\bdeep\s*cut|laceration|wound\s*that\s*won'?t\s*stop\b", "Deep cut / laceration"),
    (r"\bpersistent\s*vomit|vomiting.*hour|can'?t\s*keep\s*food\b", "Persistent vomiting"),
    (r"\bsevere\s*headache|worst\s*headache\b", "Severe headache"),
    (r"\bsudden\s*confusion|disoriented\b", "Sudden confusion / disorientation"),
    (r"\burinary\s*tract\s*infection|uti\b", "Urinary tract infection"),
    (r"\bkidney\s*(pain|stone)\b", "Kidney pain / stone"),
    (r"\bsevere\s*(back|shoulder|joint)\s*pain\b", "Severe joint / back pain"),
    (r"\binfect.*wound|wound.*infect|red\s*spreading\b", "Infected wound"),
    (r"\bdehydrat\b", "Dehydration"),
    (r"\bmoderate\s*(pain|burn)\b", "Moderate pain / burn"),
    (r"\bblood\s+in\b.{0,20}\b(urine|stool|vomit|pee|poo|feces|faeces)\b", "Blood in urine / stool / vomit"),
    (r"\bsevere\s*(ear|tooth)\s*pain\b", "Severe ear or tooth pain"),
    (r"\bdifficulty\s*swallow\b", "Difficulty swallowing"),
    (r"\beye\s*(injury|scratch|foreign\s*object)\b", "Eye injury / foreign object"),
    (r"\bmigraine\b", "Migraine"),
    (r"\bchild.*vomit|vomit.*child|baby.*vomit\b", "Vomiting in child / infant"),
    (r"\bfever.*2\s*days|persistent\s*fever\b", "Fever lasting more than 2 days"),
    (r"\bpneumonia\b", "Suspected pneumonia"),
    (r"\basthma\s*attack\b", "Asthma attack"),
]

ROUTINE_KEYWORDS: List[Tuple[str, str]] = [
    (r"\bcommon\s*cold|runny\s*nose|stuffy\s*nose|congestion\b", "Common cold / congestion"),
    (r"\bsore\s*throat\b", "Sore throat"),
    (r"\bmild\s*(cough|fever|headache|pain|rash|burn)\b", "Mild symptoms"),
    (r"\bminor\s*(cut|scrape|bruise|rash|itch)\b", "Minor injury / rash"),
    (r"\bseasonal\s*allergy|hay\s*fever\b", "Seasonal allergies"),
    (r"\bfatigue|tiredness|low\s*energy\b", "Fatigue / tiredness"),
    (r"\bnausea\b", "Mild nausea"),
    (r"\binsomnia|sleep\s*(problem|issue)\b", "Sleep issues"),
    (r"\bback\s*pain\b", "General back pain"),
    (r"\bheadache\b", "Headache"),
    (r"\bfever\b", "Low-grade fever"),
    (r"\bcough\b", "Cough"),
    (r"\bstomach\s*ache|indigestion|heartburn|acid\s*reflux\b", "Stomach discomfort / indigestion"),
    (r"\bcheck.?up|routine\s*visit|annual\s*exam|prescription\s*refill\b", "Routine checkup / prescription"),
    (r"\bminor\s*infection|pink\s*eye|conjunctivitis\b", "Minor infection / pink eye"),
    (r"\banxiety|stress|feel\s*anxious\b", "Anxiety / stress (non-emergency)"),
    (r"\bitch|skin\s*rash\b", "Skin rash / itch"),
    (r"\bdiarrhea\b", "Mild diarrhea"),
    (r"\bswollen\s*(lymph|gland)\b", "Swollen glands"),
    (r"\bcold\s*sore|canker\s*sore\b", "Cold sore / canker sore"),
]


# ─────────────────────────────────────────────────────────────
# Core Triage Logic
# ─────────────────────────────────────────────────────────────

def _match_keywords(text: str, rules: List[Tuple[str, str]]) -> List[str]:
    """Return list of matched symptom labels for given text."""
    found = []
    t = text.lower()
    for pattern, label in rules:
        if re.search(pattern, t, re.IGNORECASE):
            if label not in found:
                found.append(label)
    return found


def classify_triage(user_messages: List[str]) -> TriageResult:
    """
    Classify a list of user messages into a triage level.
    Returns a TriageResult with urgency, reasoning, and next steps.
    No external calls — fully local, no data stored.
    """
    combined_text = " ".join(user_messages)

    # Check emergency first (highest priority)
    emergency_matches = _match_keywords(combined_text, EMERGENCY_KEYWORDS)
    if emergency_matches:
        return TriageResult(
            urgency="EMERGENCY",
            urgency_emoji="🔴",
            urgency_color="#ef4444",
            matched_symptoms=emergency_matches,
            reasoning=(
                f"Your description includes symptoms associated with a potential medical emergency: "
                f"{', '.join(emergency_matches)}. "
                "Emergency symptoms require immediate professional evaluation."
            ),
            next_steps=[
                "📞 Call emergency services (112 or 911) NOW",
                "🚨 Go to the nearest Emergency Room immediately",
                "🧑‍🤝‍🧑 Do not drive yourself — ask someone to take you or call an ambulance",
                "📱 Stay on the line with emergency services until help arrives",
                "🧘 Try to stay calm and lie down if feeling faint",
            ],
            facility_type="Emergency Room (ER)",
            call_to_action="Call 112 / 911 immediately",
        )

    # Check urgent next
    urgent_matches = _match_keywords(combined_text, URGENT_KEYWORDS)
    if urgent_matches:
        return TriageResult(
            urgency="URGENT",
            urgency_emoji="🟡",
            urgency_color="#f59e0b",
            matched_symptoms=urgent_matches,
            reasoning=(
                f"Your description includes symptoms that need prompt medical attention: "
                f"{', '.join(urgent_matches)}. "
                "These are not immediately life-threatening, but should be evaluated within a few hours."
            ),
            next_steps=[
                "🏥 Visit an Urgent Care Center within the next 2–4 hours",
                "📞 Call your primary care doctor to check for same-day availability",
                "💊 Do not self-medicate without guidance from a healthcare professional",
                "💧 Stay hydrated and rest in the meantime",
                "📋 Note the time symptoms started and any medications taken",
            ],
            facility_type="Urgent Care Center",
            call_to_action="Visit Urgent Care within 2–4 hours",
        )

    # Check routine
    routine_matches = _match_keywords(combined_text, ROUTINE_KEYWORDS)
    if routine_matches:
        return TriageResult(
            urgency="ROUTINE",
            urgency_emoji="🟢",
            urgency_color="#4ade80",
            matched_symptoms=routine_matches,
            reasoning=(
                f"Your description includes symptoms that are generally manageable at home or with routine care: "
                f"{', '.join(routine_matches)}. "
                "These symptoms are typically non-urgent but should still be monitored."
            ),
            next_steps=[
                "🏠 Rest at home and monitor your symptoms",
                "📅 Schedule an appointment with your Primary Care physician at your convenience",
                "💊 Over-the-counter remedies may help — follow package directions carefully",
                "💧 Stay well-hydrated and maintain adequate rest",
                "📈 Seek care sooner if symptoms worsen or persist beyond 3–5 days",
            ],
            facility_type="Primary Care / Family Doctor",
            call_to_action="Schedule a routine appointment",
        )

    # Fallback — not enough information
    return TriageResult(
        urgency="ROUTINE",
        urgency_emoji="🟢",
        urgency_color="#4ade80",
        matched_symptoms=[],
        reasoning=(
            "I wasn't able to clearly identify specific symptoms from your description. "
            "Please provide more details about what you're experiencing, including the location, "
            "severity (1-10), and how long the symptoms have been present."
        ),
        next_steps=[
            "📝 Describe your symptoms in more detail (location, severity 1-10, duration)",
            "📅 If unsure, consult your Primary Care physician",
            "📞 Call a nurse helpline for guidance",
            "🔍 Monitor for any worsening symptoms",
        ],
        facility_type="Primary Care",
        call_to_action="Provide more symptom details or consult your doctor",
    )


def generate_triage_report(result: TriageResult, location: str = "") -> str:
    """Generate a full text triage report from a TriageResult."""
    lines = [
        f"# {result.urgency_emoji} Healthcare Triage Assessment\n",
        f"**Urgency Level:** {result.urgency_emoji} {result.urgency}\n",
        f"\n{DISCLAIMER}\n",
        f"\n## 🩺 Symptom Assessment\n",
        f"{result.reasoning}\n",
    ]

    if result.matched_symptoms:
        lines.append(f"\n**Identified Symptoms:**\n")
        for s in result.matched_symptoms:
            lines.append(f"- {s}\n")

    lines.append(f"\n## 📋 Recommended Next Steps\n")
    for step in result.next_steps:
        lines.append(f"{step}\n")

    lines.append(f"\n## 🏥 Recommended Facility Type\n")
    lines.append(f"**{result.facility_type}**\n")

    if location:
        lines.append(f"\n## 📍 Your Location\n")
        lines.append(f"You mentioned: **{location}**\n")
        lines.append(
            f"Please search for the nearest **{result.facility_type}** in {location} "
            f"via Google Maps, Apple Maps, or your local health directory.\n"
        )

    lines.append(f"\n---\n")
    lines.append(f"*{DISCLAIMER}*\n")

    return "".join(lines)


# ─────────────────────────────────────────────────────────────
# Quick Scenario Presets for UI
# ─────────────────────────────────────────────────────────────

TRIAGE_QUICK_SCENARIOS = [
    {
        "id": "chest_pain",
        "label": "Chest Pain",
        "emoji": "💔",
        "query": "I have severe chest pain and pressure, it's spreading to my left arm",
        "category": "EMERGENCY",
    },
    {
        "id": "breathing",
        "label": "Breathing Difficulty",
        "emoji": "🫁",
        "query": "I can't breathe properly, I'm struggling to get air",
        "category": "EMERGENCY",
    },
    {
        "id": "stroke_signs",
        "label": "Stroke Warning Signs",
        "emoji": "🧠",
        "query": "My face is drooping on one side and I have arm weakness and difficulty speaking",
        "category": "EMERGENCY",
    },
    {
        "id": "high_fever",
        "label": "High Fever",
        "emoji": "🌡️",
        "query": "I have a fever of 104°F since yesterday and I'm getting chills and shaking",
        "category": "URGENT",
    },
    {
        "id": "fracture",
        "label": "Possible Fracture",
        "emoji": "🦴",
        "query": "I fell and I think I may have a broken bone in my arm, it's very painful to move",
        "category": "URGENT",
    },
    {
        "id": "severe_headache",
        "label": "Severe Headache",
        "emoji": "🤕",
        "query": "I have the worst headache of my life, it came on suddenly",
        "category": "URGENT",
    },
    {
        "id": "deep_cut",
        "label": "Deep Cut / Wound",
        "emoji": "🩹",
        "query": "I have a deep laceration that won't stop bleeding",
        "category": "URGENT",
    },
    {
        "id": "asthma",
        "label": "Asthma Attack",
        "emoji": "💨",
        "query": "I'm having an asthma attack, my inhaler isn't working well",
        "category": "URGENT",
    },
    {
        "id": "cold",
        "label": "Common Cold",
        "emoji": "🤧",
        "query": "I have a runny nose, mild cough, and sore throat for about 2 days",
        "category": "ROUTINE",
    },
    {
        "id": "mild_fever",
        "label": "Mild Fever",
        "emoji": "🌡️",
        "query": "I have a mild fever of 100°F, headache and feel a bit tired",
        "category": "ROUTINE",
    },
    {
        "id": "stomach",
        "label": "Stomach Ache",
        "emoji": "🤢",
        "query": "I have a stomach ache and some indigestion after eating",
        "category": "ROUTINE",
    },
    {
        "id": "checkup",
        "label": "Routine Checkup",
        "emoji": "📋",
        "query": "I need to schedule my annual physical exam and prescription refill",
        "category": "ROUTINE",
    },
]
