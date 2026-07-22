"""
CrewAI Agents

Two agents work together (a "Crew") to solve the business problem:
  1. Demand Analyst Agent  -> reads LightGBM demand predictions,
                              identifies high-demand / low-supply zones
  2. Dispatch Planner Agent -> takes that analysis and produces a
                              concrete vehicle repositioning plan

Uses a LOCAL LLM (Mistral, via Ollama) for agent reasoning - no
subscription, no API key, fully offline after the model is pulled.

This module can be run directly (python agents/dispatch_crew.py) OR
imported by the FastAPI backend (api/main.py), which calls
run_dispatch_cycle() to trigger the same agent logic on demand.
"""

import pandas as pd
import joblib
import random
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List
import time


class VehicleMove(BaseModel):
    vehicles: int = Field(description="Number of vehicles to move")
    from_zone: str = Field(description="Zone the vehicles move from")
    to_zone: str = Field(description="Zone the vehicles move to")
    reason: str = Field(description="One-line reason for this move")


class DispatchPlan(BaseModel):
    moves: List[VehicleMove]

# Module-level setup (loaded once, reused by every call)

llm = LLM(
    model="ollama/mistral",
    base_url="http://localhost:11434",
)

model = joblib.load("models/demand_model.pkl")

ZONE_MAPPING = {"Zone_A": 0, "Zone_B": 1, "Zone_C": 2,
                "Zone_D": 3, "Zone_E": 4, "Zone_F": 5}

# Baseline vehicle counts per zone. In production this would come from
# a live fleet-tracking database/API instead of being hardcoded here.
BASE_AVAILABILITY = {
    "Zone_A": 15, "Zone_B": 18, "Zone_C": 12,
    "Zone_D": 10, "Zone_E": 14, "Zone_F": 20,
}

_availability_cache = None
_cache_timestamp = None
CACHE_TTL_SECONDS = 60


def get_current_availability() -> dict:
    """
    Returns the current available-vehicle count per zone.

    Simulates "live" data, but caches the result for CACHE_TTL_SECONDS
    so that a dashboard load and a dispatch run happening close together
    see the SAME numbers instead of two different random draws. After
    the TTL expires, a fresh "snapshot" is generated - similar to how
    real vehicle counts would naturally drift over time.

    To connect this to a REAL fleet-tracking API later, replace the
    body inside the `if` block with an API call, e.g.:

        response = requests.get(FLEET_API_URL, headers=AUTH_HEADERS)
        return response.json()
    """
    global _availability_cache, _cache_timestamp

    now = time.time()
    if _availability_cache is None or (now - _cache_timestamp) > CACHE_TTL_SECONDS:
        _availability_cache = {
            zone: max(0, val + random.randint(-8, 8))
            for zone, val in BASE_AVAILABILITY.items()
        }
        _cache_timestamp = now

    return _availability_cache


def build_forecast(day_of_week: int = None) -> pd.DataFrame:
    """Predict demand for every zone, every hour of the day."""
    if day_of_week is None:
        day_of_week = datetime.now().weekday()

    forecast_rows = []
    for zone_name, zone_id in ZONE_MAPPING.items():
        for hour in range(24):
            is_rush_hour = 1 if hour in [8, 9, 17, 18, 19] else 0
            is_night = 1 if (hour >= 21 or hour <= 1) else 0
            features = pd.DataFrame([{
                "day_of_week": day_of_week,
                "is_weekend": 0,
                "hour": hour,
                "zone_id": zone_id,
                "is_rush_hour": is_rush_hour,
                "is_night": is_night,
            }])
            predicted_demand = model.predict(features)[0]
            forecast_rows.append({
                "zone": zone_name,
                "hour": hour,
                "predicted_demand": round(predicted_demand, 1),
            })
    return pd.DataFrame(forecast_rows)

def auto_topup(zone_gap: dict, already_approved_moves: list) -> list:
    """
    Deterministic top-up step: after the LLM's plan has been validated,
    calculate how much surplus and shortage remain uncovered, and
    greedily allocate the largest remaining surplus to the largest
    remaining shortage until one side runs out.

    This never overrides or removes any LLM-suggested move - it only
    fills gaps the LLM's plan left behind, using a clearly-labeled
    reason so it's obvious in the UI which moves came from the AI
    agents and which came from this automatic safety net.
    """

    remaining_surplus = {z: max(0, -g) for z, g in zone_gap.items()}
    remaining_shortage = {z: max(0, g) for z, g in zone_gap.items()}

    for m in already_approved_moves:
        remaining_surplus[m["from_zone"]] = max(
            0, remaining_surplus.get(m["from_zone"], 0) - m["vehicles"]
        )
        remaining_shortage[m["to_zone"]] = max(
            0, remaining_shortage.get(m["to_zone"], 0) - m["vehicles"]
        )

    topup_moves = []

    while True:
        # Pick the zone with the biggest unmet shortage right now
        short_zone = max(remaining_shortage, key=remaining_shortage.get)
        if remaining_shortage[short_zone] <= 0:
            break

        # Pick the zone with the biggest unused surplus right now
        surplus_zone = max(remaining_surplus, key=remaining_surplus.get)
        if remaining_surplus[surplus_zone] <= 0:
            break

        move_amount = int(min(
            remaining_surplus[surplus_zone],
            remaining_shortage[short_zone],
        ))
        if move_amount <= 0:
            break

        topup_moves.append({
            "vehicles": move_amount,
            "from_zone": surplus_zone,
            "to_zone": short_zone,
            "reason": (
                f"Automatic balancing: {surplus_zone} still had unused "
                f"surplus after the agents' plan, redirected to help "
                f"close the remaining gap in {short_zone}."
            ),
        })

        remaining_surplus[surplus_zone] -= move_amount
        remaining_shortage[short_zone] -= move_amount

    return topup_moves


def run_dispatch_cycle(current_hour: int = None) -> dict:
    """
    Runs one full agent cycle: build forecast -> agents reason about
    it -> validate the plan against real numbers.

    Returns a dict with: forecast_summary (str), zone_gap (dict),
    validated_moves (list of dicts), rejected_moves (list of dicts),
    raw_agent_output (str).
    """
    if current_hour is None:
        current_hour = datetime.now().hour

    forecast_df = build_forecast()
    current_availability = get_current_availability()

    upcoming = forecast_df[forecast_df["hour"].isin(
        [current_hour, current_hour + 1, current_hour + 2]
    )]

    zone_gap = {}
    summary_lines = []
    for zone in ZONE_MAPPING:
        zone_forecast = upcoming[upcoming["zone"] == zone]["predicted_demand"].mean()
        available = current_availability[zone]
        gap = round(zone_forecast - available, 1)
        zone_gap[zone] = gap
        summary_lines.append(
            f"{zone}: avg predicted demand next 3 hrs = {zone_forecast:.1f}, "
            f"currently available vehicles = {available}, gap = {gap}"
        )
    forecast_summary = "\n".join(summary_lines)

    # ---- Agents ----
    demand_analyst = Agent(
        role="Demand Analyst",
        goal="Identify which zones will have too much demand and too few "
             "vehicles, and which zones have surplus idle vehicles.",
        backstory=(
            "You are a data-driven mobility analyst at Voss Mobility. "
            "You specialize in reading demand forecasts and translating "
            "them into clear, prioritized insights for the dispatch team."
        ),
        llm=llm,
        verbose=True,
    )

    dispatch_planner = Agent(
        role="Dispatch Planner",
        goal="Turn demand analysis into a concrete, minimal-move vehicle "
             "repositioning plan that reduces customer wait times.",
        backstory=(
            "You are an operations planner at Voss Mobility. You take "
            "analyst insights and decide exactly which vehicles should "
            "move from which zone to which zone, keeping moves practical "
            "and few."
        ),
        llm=llm,
        verbose=True,
    )

    analysis_task = Task(
        description=(
            "Here is the demand forecast vs. current vehicle availability "
            f"for the next 3 hours, by zone:\n\n{forecast_summary}\n\n"
            "IMPORTANT: gap = predicted_demand - available_vehicles. "
            "A POSITIVE gap means that zone is SHORT on vehicles (bad, "
            "needs more vehicles sent in). A NEGATIVE gap means that zone "
            "has MORE vehicles than it needs (a real surplus - safe to "
            "pull vehicles from). Do not label a zone as 'surplus' unless "
            "its gap is negative. Identify: (1) the zones with the most "
            "urgent shortage (highest positive gap, ranked by severity), "
            "and (2) the zones with a genuine surplus (negative gap only)."
        ),
        expected_output=(
            "A short prioritized list of zones with shortages (most "
            "severe first) and a short list of zones with surplus vehicles."
        ),
        agent=demand_analyst,
    )

    planning_task = Task(
        description=(
            "Using the Demand Analyst's findings, create a specific "
            "vehicle repositioning plan. For each recommended move, "
            "state: how many vehicles, from which zone, to which zone, "
            "and a one-line reason. Prefer moving vehicles from the "
            "nearest surplus zone. Keep the plan to a maximum of 5 moves."
        ),
        expected_output=(
            "A numbered list of concrete repositioning actions, e.g. "
            "'Move 3 vehicles from Zone_D to Zone_C - Zone_C has a demand "
            "gap of 12, Zone_D has surplus of 9.'"
        ),
        agent=dispatch_planner,
        context=[analysis_task],
        output_pydantic=DispatchPlan,
    )

    crew = Crew(
        agents=[demand_analyst, dispatch_planner],
        tasks=[analysis_task, planning_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    # ---- Validation guardrail (never trust the LLM's numbers blindly) ----
    remaining_surplus = {zone: max(0, -gap) for zone, gap in zone_gap.items()}

    plan: DispatchPlan = result.pydantic
    validated_moves = []
    rejected_moves = []

    if plan is not None:
        for move in plan.moves:
            from_zone = move.from_zone

            if move.vehicles <= 0:
                rejected_moves.append({
                    "vehicles": move.vehicles, "from_zone": from_zone,
                    "to_zone": move.to_zone,
                    "reason_rejected": f"Invalid vehicle count "
                                       f"({move.vehicles}) - must be a "
                                       f"positive number",
                })
                continue

            if from_zone not in ZONE_MAPPING:
                rejected_moves.append({
                    "vehicles": move.vehicles, "from_zone": from_zone,
                    "to_zone": move.to_zone,
                    "reason_rejected": f"Unknown zone '{from_zone}'",
                })
                continue

            if zone_gap[from_zone] >= 0:
                rejected_moves.append({
                    "vehicles": move.vehicles, "from_zone": from_zone,
                    "to_zone": move.to_zone,
                    "reason_rejected": f"{from_zone} has no surplus "
                                       f"(gap={zone_gap[from_zone]})",
                })
                continue

            available_surplus = round(remaining_surplus.get(from_zone, 0), 1)
            if move.vehicles > available_surplus:
                rejected_moves.append({
                    "vehicles": move.vehicles, "from_zone": from_zone,
                    "to_zone": move.to_zone,
                    "reason_rejected": f"{from_zone} only has "
                                       f"{available_surplus} vehicles of "
                                       f"real surplus left",
                })
                continue

            to_zone = move.to_zone
            if to_zone not in ZONE_MAPPING or zone_gap.get(to_zone, 0) <= 0:
                rejected_moves.append({
                    "vehicles": move.vehicles, "from_zone": from_zone,
                    "to_zone": to_zone,
                    "reason_rejected": f"{to_zone} does not have a real "
                                       f"shortage (gap={zone_gap.get(to_zone)}) "
                                       f"- sending vehicles there would not "
                                       f"help",
                })
                continue

            remaining_surplus[from_zone] -= move.vehicles
            validated_moves.append({
                "vehicles": move.vehicles, "from_zone": from_zone,
                "to_zone": move.to_zone, "reason": move.reason,
            })

    return {
        "forecast_summary": forecast_summary,
        "zone_gap": zone_gap,
        "validated_moves": validated_moves,
        "availability": current_availability,
        "rejected_moves": rejected_moves,
        "raw_agent_output": result.raw,
    }


def save_report(cycle_result: dict, path: str = "models/dispatch_plan.txt"):
    report_lines = [
        "Voss Mobility - Autonomous Dispatch Plan (Validated)",
        "=" * 50, "",
        "Input forecast summary:", cycle_result["forecast_summary"], "",
    ]

    if cycle_result["validated_moves"]:
        report_lines.append("APPROVED MOVES:")
        for i, m in enumerate(cycle_result["validated_moves"], 1):
            report_lines.append(
                f"{i}. Move {m['vehicles']} vehicles from {m['from_zone']} "
                f"to {m['to_zone']} - {m['reason']}"
            )
    else:
        report_lines.append("No valid moves were produced by the agent.")

    if cycle_result["rejected_moves"]:
        report_lines.append("\nREJECTED MOVES (failed validation - not executed):")
        for m in cycle_result["rejected_moves"]:
            report_lines.append(
                f"- Move {m['vehicles']} from {m['from_zone']} to "
                f"{m['to_zone']}: REJECTED ({m['reason_rejected']})"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write("\n\n--- Raw (unvalidated) agent output, for reference ---\n")
        f.write(cycle_result["raw_agent_output"])


if __name__ == "__main__":
    print("=== Running Dispatch Cycle ===\n")
    cycle_result = run_dispatch_cycle()

    print("\n" + "=" * 60)
    print("VALIDATED DISPATCH PLAN (guardrail-checked)")
    print("=" * 60)

    if cycle_result["validated_moves"]:
        for i, m in enumerate(cycle_result["validated_moves"], 1):
            print(f"{i}. Move {m['vehicles']} vehicles from {m['from_zone']} "
                  f"to {m['to_zone']} - {m['reason']}")
    else:
        print("No valid moves were produced by the agent.")

    if cycle_result["rejected_moves"]:
        print("\nREJECTED MOVES (failed validation - not executed):")
        for m in cycle_result["rejected_moves"]:
            print(f"- Move {m['vehicles']} from {m['from_zone']} to "
                  f"{m['to_zone']}: REJECTED ({m['reason_rejected']})")

    save_report(cycle_result)
    print("\nSaved to models/dispatch_plan.txt")
