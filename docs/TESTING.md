# Testing — Autonomous Ride Dispatch Agent (Voss Mobility)

## Overview

The system was tested in two complementary ways:

1. **Automated tests** (`test_system.py`) — fast, repeatable checks on
   the demand model, the live-availability simulation, and the
   guardrail's validation logic. No LLM required, runs in seconds.
2. **Manual end-to-end testing** — real runs of the full pipeline
   (Model → CrewAI Agents → Guardrail → Auto-Balancing → Database →
   API → Dashboard), using the actual local Mistral LLM.

Automated tests catch regressions in the logic quickly. Manual testing
with the real LLM was equally important, since it's what surfaced the
actual inconsistencies an AI agent can produce in practice — and led
directly to the fixes described below.

## 1. Automated Tests (`test_system.py`)

Run with:

```bash
python test_system.py
```

| Area | Checks |
|---|---|
| Demand model | Loads correctly; predictions are numeric, non-negative, and within a plausible range (0–200 rides); rush-hour demand is higher than overnight demand for the same zone |
| Live availability simulation | Returns all 6 zones; values are non-negative; values stay consistent within the caching window (see below) |
| Guardrail validation logic | 6 cases covering every rejection rule: excess vehicles requested, negative/zero vehicle counts, pulling from a non-surplus zone, sending to a non-shortage zone, a valid move being approved, and sequential moves correctly depleting a zone's surplus |

**Result:**
```
RESULTS: 14 passed, 0 failed
```

## 2. What Testing Uncovered — and How It Was Fixed

### Dashboard consistency

Early testing surfaced a real inconsistency: the zone cards and the
dispatch plan could each show different numbers for the same zone at
the same moment — e.g. a zone marked "balanced" on its card while the
dispatch plan simultaneously treated it as short on vehicles. This
happened because each view independently re-simulated vehicle
availability, and could land on two different numbers within the same
refresh cycle.

**Fix:** availability is now cached for a short window (60 seconds), so
the zone cards and the dispatch plan always reason over the same
snapshot of data. The automated test suite was updated accordingly to
verify this consistency directly.

### Guardrail hardening

The LLM does not reliably respect numeric constraints on its own, so
several real mistakes were observed during manual runs and used to
harden the guardrail that checks every move before it's approved:

| # | Mistake observed in a real agent run | Guardrail rule added |
|---|---|---|
| 1 | Proposed moving more vehicles than a zone's actual surplus | Reject if requested vehicles exceed the zone's remaining surplus |
| 2 | Proposed a negative or zero vehicle count | Reject non-positive vehicle counts |
| 3 | Proposed pulling vehicles from a zone that was itself short | Reject if the source zone has no real surplus |
| 4 | Proposed sending vehicles to a zone that already had surplus | Reject if the destination zone has no real shortage |

Every rejected move is logged with a plain-English reason and shown in
the dashboard's "Rejected by guardrail" column, so the reasoning stays
fully auditable.

### Full surplus utilization

Manual testing also showed that the AI agents' plan (capped at 5 moves
per run, to keep suggestions practical) could leave some surplus
unused even when other zones still needed vehicles. A deterministic
top-up step now runs after the agents' plan — pure calculation, no LLM
— to allocate any remaining surplus to any remaining shortage, so
available capacity isn't left idle. These moves are clearly labeled
"Automatic balancing" in the dashboard, separate from the AI agents'
own reasoning.

## 3. End-to-End & Deployment Verification

- All 5 API endpoints exercised directly via the Swagger UI (`/docs`),
  including full dispatch runs persisted to PostgreSQL
- Dashboard verified for: correct data on load/refresh, approved vs.
  rejected moves rendering distinctly, and clear "API Offline"
  handling if the backend is unreachable
- Production deployment (`voss-dispatch`, installed via
  `pip install -e .`) verified to start cleanly, check all
  dependencies (model, database, Ollama, frontend build), and serve
  the full dashboard from a single process — no Docker required

## 4. Known Limitations (documented, not defects)

- **Synthetic data:** demand data is generated for this demo, not
  drawn from real Voss Mobility operations.
- **On-demand:** the system recalculates on request
  (button click / API call), rather than streaming continuously.
- **Simulated fleet data:** vehicle availability is simulated around
  fixed baselines rather than read from a live GPS fleet system.
