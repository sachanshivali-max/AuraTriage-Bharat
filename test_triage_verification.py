"""
Verification tests for run_triage.py (Capstone Project)
Tests the classify_triage() logic, generate_triage_report(), and quick scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_triage import classify_triage, generate_triage_report, TRIAGE_QUICK_SCENARIOS, DISCLAIMER, TriageResult

pass_count = 0
fail_count = 0

def check(condition, label):
    global pass_count, fail_count
    if condition:
        print(f"[PASS] {label}")
        pass_count += 1
    else:
        print(f"[FAIL] {label}")
        fail_count += 1

print("\n=== TEST 1: Emergency Symptom Classification ===")
r = classify_triage(["I have severe chest pain and pressure radiating to my left arm"])
check(r.urgency == "EMERGENCY", "Chest pain classified as EMERGENCY")
check(r.urgency_emoji == "🔴", "Emergency has red emoji")
check("Chest pain" in r.matched_symptoms[0] or "chest" in r.matched_symptoms[0].lower(), "Chest pain in matched symptoms")
check(len(r.next_steps) > 0, "Emergency has next steps")
check("112" in " ".join(r.next_steps) or "911" in " ".join(r.next_steps), "Emergency includes 112/911 guidance")
check(r.facility_type == "Emergency Room (ER)", "Emergency routes to ER")

r2 = classify_triage(["I can't breathe, difficulty breathing"])
check(r2.urgency == "EMERGENCY", "Breathing difficulty classified as EMERGENCY")

r3 = classify_triage(["My face is drooping and I have arm weakness and difficulty speaking"])
check(r3.urgency == "EMERGENCY", "Stroke FAST signs classified as EMERGENCY")

r4 = classify_triage(["I want to end my life"])
check(r4.urgency == "EMERGENCY", "Mental health emergency classified as EMERGENCY")

r5 = classify_triage(["I am having a seizure", "convulsions"])
check(r5.urgency == "EMERGENCY", "Seizure classified as EMERGENCY")

r6 = classify_triage(["Severe allergic reaction, throat swelling"])
check(r6.urgency == "EMERGENCY", "Anaphylaxis classified as EMERGENCY")

print("\n=== TEST 2: Urgent Symptom Classification ===")
r7 = classify_triage(["I have a high fever of 104 degrees since yesterday"])
check(r7.urgency == "URGENT", "High fever classified as URGENT")
check(r7.urgency_emoji == "🟡", "Urgent has yellow emoji")
check(r7.facility_type == "Urgent Care Center", "Urgent routes to Urgent Care")

r8 = classify_triage(["I think I have a fracture in my arm, very painful"])
check(r8.urgency == "URGENT", "Fracture classified as URGENT")

r9 = classify_triage(["I have the worst headache of my life, came on suddenly"])
check(r9.urgency == "URGENT", "Severe headache classified as URGENT")

r10 = classify_triage(["Deep laceration that won't stop bleeding"])
check(r10.urgency == "URGENT", "Deep cut classified as URGENT")

r11 = classify_triage(["I'm having an asthma attack, my inhaler isn't working"])
check(r11.urgency == "URGENT", "Asthma attack classified as URGENT")

r12 = classify_triage(["I have blood in my urine"])
check(r12.urgency == "URGENT", "Blood in urine classified as URGENT")

print("\n=== TEST 3: Routine Symptom Classification ===")
r13 = classify_triage(["I have a runny nose and mild cough for 2 days"])
check(r13.urgency == "ROUTINE", "Common cold classified as ROUTINE")
check(r13.urgency_emoji == "🟢", "Routine has green emoji")
check(r13.facility_type == "Primary Care / Family Doctor", "Routine routes to Primary Care")

r14 = classify_triage(["I have a stomach ache and some indigestion after eating"])
check(r14.urgency == "ROUTINE", "Indigestion classified as ROUTINE")

r15 = classify_triage(["I need my annual physical exam and prescription refill"])
check(r15.urgency == "ROUTINE", "Routine checkup classified as ROUTINE")

r16 = classify_triage(["I have mild anxiety and stress"])
check(r16.urgency == "ROUTINE", "Mild anxiety classified as ROUTINE")

print("\n=== TEST 4: Priority Escalation (Emergency > Urgent > Routine) ===")
# Mixed symptoms — emergency should win
r_mixed = classify_triage(["I have a headache and runny nose", "but also chest pain"])
check(r_mixed.urgency == "EMERGENCY", "Emergency beats routine in mixed input")

r_mixed2 = classify_triage(["I have indigestion and also a high fever"])
check(r_mixed2.urgency == "URGENT", "Urgent beats routine in mixed input")

print("\n=== TEST 5: Multi-message Input ===")
r_multi = classify_triage([
    "I've been feeling unwell",
    "I have severe chest pain",
    "Also some difficulty breathing"
])
check(r_multi.urgency == "EMERGENCY", "Multi-message emergency detection works")
check(len(r_multi.matched_symptoms) >= 2, "Multiple symptoms detected in multi-message input")

print("\n=== TEST 6: Fallback / Unknown Symptoms ===")
r_fallback = classify_triage(["Hello, I need to talk to someone"])
check(r_fallback.urgency == "ROUTINE", "Vague input defaults to ROUTINE")
check(len(r_fallback.matched_symptoms) == 0 or r_fallback.urgency in ["ROUTINE", "URGENT"], "Fallback has no matched symptoms or sensible urgency")
check("more details" in r_fallback.reasoning.lower() or "wasn" in r_fallback.reasoning.lower(), "Fallback reasoning requests more info")

print("\n=== TEST 7: Report Generation ===")
result = classify_triage(["chest pain"])
report = generate_triage_report(result, location="Mumbai")
check("EMERGENCY" in report, "Emergency level in report")
check("Mumbai" in report, "Location included in report")
check(DISCLAIMER[:30] in report, "Disclaimer included in report")
check("Next Steps" in report or "next" in report.lower(), "Next steps section in report")
check("Facility" in report, "Facility type in report")
check("Emergency Room" in report, "ER facility type in emergency report")
check("Mumbai" in report, "Location guidance in report")

report2 = generate_triage_report(result)  # No location
check("Mumbai" not in report2, "No location = no location section")

print("\n=== TEST 8: Quick Scenarios Data Integrity ===")
check(len(TRIAGE_QUICK_SCENARIOS) >= 10, "At least 10 quick scenarios defined")
categories = [s["category"] for s in TRIAGE_QUICK_SCENARIOS]
check("EMERGENCY" in categories, "Emergency quick scenarios present")
check("URGENT" in categories, "Urgent quick scenarios present")
check("ROUTINE" in categories, "Routine quick scenarios present")

for scenario in TRIAGE_QUICK_SCENARIOS:
    check("id" in scenario and "label" in scenario and "query" in scenario and "category" in scenario,
          f"Scenario '{scenario.get('label', '?')}' has required fields")
    check(scenario["category"] in ("EMERGENCY", "URGENT", "ROUTINE"),
          f"Scenario '{scenario.get('label', '?')}' has valid category")

print("\n=== TEST 9: Scenario Labels Match Their Classifications ===")
for scenario in TRIAGE_QUICK_SCENARIOS:
    r = classify_triage([scenario["query"]])
    expected = scenario["category"]
    # For this test, accept within 1 level (e.g., URGENT scenario might be EMERGENCY — that's fine for safety)
    level_map = {"ROUTINE": 1, "URGENT": 2, "EMERGENCY": 3}
    actual_level = level_map[r.urgency]
    expected_level = level_map[expected]
    # The actual urgency should be >= expected (over-triage is acceptable; under-triage is not)
    ok = actual_level >= expected_level
    check(ok, f"Scenario '{scenario['label']}': expected {expected}, got {r.urgency}")

print("\n=== TEST 10: Disclaimer Content ===")
check(len(DISCLAIMER) > 50, "Disclaimer has substantive content")
check("AI" in DISCLAIMER, "Disclaimer mentions AI status")
check("112" in DISCLAIMER or "911" in DISCLAIMER, "Disclaimer includes emergency number")
check("doctor" in DISCLAIMER.lower() or "medical" in DISCLAIMER.lower(), "Disclaimer references medical professional")

print("\n=== TEST 11: TriageResult Data Structure ===")
result = classify_triage(["chest pain"])
check(hasattr(result, 'urgency'), "TriageResult has urgency field")
check(hasattr(result, 'urgency_emoji'), "TriageResult has urgency_emoji field")
check(hasattr(result, 'urgency_color'), "TriageResult has urgency_color field")
check(hasattr(result, 'matched_symptoms'), "TriageResult has matched_symptoms field")
check(hasattr(result, 'reasoning'), "TriageResult has reasoning field")
check(hasattr(result, 'next_steps'), "TriageResult has next_steps field")
check(hasattr(result, 'facility_type'), "TriageResult has facility_type field")
check(hasattr(result, 'call_to_action'), "TriageResult has call_to_action field")
check(isinstance(result.matched_symptoms, list), "matched_symptoms is a list")
check(isinstance(result.next_steps, list), "next_steps is a list")
check(result.urgency_color.startswith("#"), "urgency_color is a hex color")

print("\n==========================================")
print(f"RESULTS: {pass_count} passed, {fail_count} failed out of {pass_count + fail_count} tests")
if fail_count == 0:
    print("STATUS: ALL TESTS PASSED ✅")
else:
    print(f"STATUS: {fail_count} TEST(S) FAILED [FAILED]")
print("==========================================\n")

sys.exit(1 if fail_count > 0 else 0)
