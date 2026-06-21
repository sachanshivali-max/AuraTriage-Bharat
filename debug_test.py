import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_triage import classify_triage

# Test blood in urine
r = classify_triage(["I have blood in my urine"])
print(f"urgency: {r.urgency}")
print(f"matched: {r.matched_symptoms}")
print(f"reasoning: {r.reasoning[:200]}")
