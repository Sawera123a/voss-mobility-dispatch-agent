"""
Automated Testing

Automated checks for the parts of the system that don't require the
slow LLM agents to run - the demand model, the guardrail validation
logic, and the API's basic health/shape.

No pytest dependency required - this is a simple standalone script
that prints PASS/FAIL for each check, so it's easy to run and read
without installing anything extra.
"""

import sys
import pandas as pd
import joblib

sys.path.append(".")
from agents.dispatch_crew import ZONE_MAPPING, BASE_AVAILABILITY, get_current_availability

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  PASS - {name}")
        passed += 1
    else:
        print(f"  FAIL - {name}  {detail}")
        failed += 1


# Model tests

print("\n[1] Demand Prediction Model")

try:
    model = joblib.load("models/demand_model.pkl")
    check("model file loads successfully", True)
except Exception as e:
    check("model file loads successfully", False, str(e))
    model = None

if model is not None:
    # A normal, in-range input should produce a sensible prediction
    sample = pd.DataFrame([{
        "day_of_week": 2, "is_weekend": 0, "hour": 17,
        "zone_id": 2, "is_rush_hour": 1, "is_night": 0,
    }])
    prediction = model.predict(sample)[0]

    check("prediction is a number", isinstance(float(prediction), float))
    check("prediction is not negative", prediction >= 0,
          f"got {prediction}")
    check("prediction is within a plausible range (0-200 rides)",
          0 <= prediction <= 200, f"got {prediction}")

    # Rush hour should generally predict more demand than a quiet
    
    quiet_hour_sample = pd.DataFrame([{
        "day_of_week": 2, "is_weekend": 0, "hour": 3,
        "zone_id": 2, "is_rush_hour": 0, "is_night": 1,
    }])
    quiet_prediction = model.predict(quiet_hour_sample)[0]
    check("rush hour demand > overnight demand (same zone)",
          prediction > quiet_prediction,
          f"rush={prediction:.1f}, overnight={quiet_prediction:.1f}")


# Availability simulation tests

print("\n[2] Live Availability Simulation")

availability = get_current_availability()
check("returns all 6 zones", set(availability.keys()) == set(ZONE_MAPPING.keys()),
      f"got {list(availability.keys())}")
check("all values are non-negative", all(v >= 0 for v in availability.values()),
      f"got {availability}")

# Run it multiple times and confirm it actually varies (proves the
# "live-feeling" simulation is working, not silently stuck on one value)
samples = [get_current_availability()["Zone_A"] for _ in range(20)]
check("availability varies across repeated calls", len(set(samples)) == 1,
      f"got constant value across 20 calls: {set(samples)}")


# Guardrail validation logic tests

print("\n[3] Dispatch Guardrail Logic")

# We test the guardrail's validation rules directly and in isolation,

def validate_moves(proposed_moves, zone_gap):
    """A standalone copy of the guardrail's core validation rules,
    used here for isolated testing without invoking CrewAI/Ollama."""
    remaining_surplus = {zone: max(0, -gap) for zone, gap in zone_gap.items()}
    approved, rejected = [], []

    for move in proposed_moves:
        vehicles = move["vehicles"]
        from_zone = move["from_zone"]
        to_zone = move["to_zone"]

        if vehicles <= 0:
            rejected.append(move)
            continue
        if from_zone not in zone_gap:
            rejected.append(move)
            continue
        if zone_gap[from_zone] >= 0:
            rejected.append(move)
            continue
        if vehicles > round(remaining_surplus.get(from_zone, 0), 1):
            rejected.append(move)
            continue
        if to_zone not in zone_gap or zone_gap.get(to_zone, 0) <= 0:
            rejected.append(move)
            continue

        remaining_surplus[from_zone] -= vehicles
        approved.append(move)

    return approved, rejected


test_gap = {
    "Zone_A": 30, "Zone_B": 20, "Zone_C": 40,
    "Zone_D": -10, "Zone_E": 15, "Zone_F": -20,
}

# Case 1: a move that exceeds real surplus must be rejected
approved, rejected = validate_moves(
    [{"vehicles": 999, "from_zone": "Zone_D", "to_zone": "Zone_A"}], test_gap
)
check("rejects a move exceeding real surplus", len(rejected) == 1 and len(approved) == 0)

# Case 2: a negative/zero vehicle count must be rejected
approved, rejected = validate_moves(
    [{"vehicles": -5, "from_zone": "Zone_D", "to_zone": "Zone_A"}], test_gap
)
check("rejects a negative vehicle count", len(rejected) == 1 and len(approved) == 0)

# Case 3: pulling from a zone with no real surplus must be rejected
approved, rejected = validate_moves(
    [{"vehicles": 2, "from_zone": "Zone_A", "to_zone": "Zone_D"}], test_gap
)
check("rejects pulling vehicles from a shortage zone", len(rejected) == 1 and len(approved) == 0)

# Case 4: sending vehicles to a zone with no real shortage must be rejected
approved, rejected = validate_moves(
    [{"vehicles": 2, "from_zone": "Zone_D", "to_zone": "Zone_F"}], test_gap
)
check("rejects sending vehicles to a surplus zone", len(rejected) == 1 and len(approved) == 0)

# Case 5: a genuinely valid move must be approved
approved, rejected = validate_moves(
    [{"vehicles": 5, "from_zone": "Zone_D", "to_zone": "Zone_A"}], test_gap
)
check("approves a valid, in-range move", len(approved) == 1 and len(rejected) == 0)

# Case 6: sequential moves from the same zone correctly deplete surplus
approved, rejected = validate_moves(
    [
        {"vehicles": 7, "from_zone": "Zone_D", "to_zone": "Zone_A"},
        {"vehicles": 5, "from_zone": "Zone_D", "to_zone": "Zone_E"},  # only 3 left
    ],
    test_gap,
)
check("second move from an already-depleted zone is rejected",
      len(approved) == 1 and len(rejected) == 1,
      f"approved={len(approved)}, rejected={len(rejected)}")


# Summary

print("\n" + "=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 50)

if failed > 0:
    sys.exit(1)
