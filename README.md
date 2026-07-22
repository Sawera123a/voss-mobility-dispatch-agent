# Voss Mobility — Autonomous Ride Dispatch Agent

An AI agent system that predicts ride demand and autonomously plans vehicle repositioning for a ride-sharing fleet — replacing manual, error-prone dispatch decisions with a validated, explainable AI pipeline.

Built as a self-hosted, subscription-free project: a local LLM (Mistral via Ollama), a locally-trained ML model (LightGBM), and a Python-native deployment with no Docker required.

## The Problem

Voss Mobility's ride-sharing dispatch is rebalanced manually by ops staff, causing longer customer wait times in high-demand areas and idle, unproductive vehicles in low-demand areas.

## The Solution

- A LightGBM model predicts ride demand per zone, per hour, from historical patterns.
- Two CrewAI agents (running a local Mistral LLM) reason about that forecast: a Demand Analyst identifies shortages and surpluses, and a Dispatch Planner proposes specific vehicle moves.
- A Python validation guardrail independently re-checks every proposed move against the real numbers before it's accepted — because LLMs don't reliably do exact arithmetic, and a repositioning system that can be wrong is worse than no automation at all.
- Results are served through a FastAPI backend + PostgreSQL, and visualized on a React dashboard styled as a live dispatch control console.

## Architecture

```
Synthetic ride-demand data
        │
        ▼
LightGBM demand model  ───────────────┐
        │                             │
        ▼                             │
Demand Analyst Agent (CrewAI + Mistral)
        │
        ▼
Dispatch Planner Agent (CrewAI + Mistral)
        │
        ▼
Python Validation Guardrail  ◄─────── (re-checks against real numbers)
        │
        ▼
Automatic Top-Up  ◄─────── (fills remaining gaps, deterministic, no LLM)
        │
        ▼
PostgreSQL  ──────►  FastAPI  ──────►  React Dashboard
```

## Why a Guardrail?

Early testing showed the local LLM would sometimes propose moves that looked reasonable in prose but were numerically wrong — requesting more vehicles than a zone actually had spare, using negative vehicle counts, pulling from a zone that was itself short, or sending vehicles to a zone that didn't need them. The guardrail independently recalculates each zone's real surplus/shortage from the underlying data and rejects any move that doesn't hold up — with the reason logged and shown transparently in the dashboard, never silently discarded. See [`docs/testing.md`](./docs/testing.md) for the specific cases discovered.

## Why Automatic Balancing?

The Dispatch Planner agent limits itself to a small number of moves per run, to keep suggestions practical rather than overwhelming. Testing showed this could leave some surplus unused even when other zones still had an unmet shortage — the agent would "spread out" its limited move budget across several zones rather than fully resolving any one of them.

After the guardrail approves the agents' moves, a deterministic top-up step runs — plain calculation, no LLM involved — that allocates any remaining surplus to any remaining shortage until one side is exhausted. These moves are labeled "Automatic balancing" in the dashboard, kept clearly distinct from the AI agents' own reasoning, so it's always visible which decisions came from the AI and which came from this deterministic safety net.

## Why Cached Availability?

Early testing surfaced a real inconsistency: the zone status cards and the dispatch plan could each independently re-simulate vehicle availability, and land on two different numbers for the same zone within the same refresh cycle — e.g. a zone shown as "balanced" on its card while the dispatch plan simultaneously treated it as short. Availability is now cached for a short window (60 seconds) so every view in a given cycle reasons over the exact same snapshot of data.

## Tech Stack

| Layer | Technology |
|---|---|
| Demand prediction | LightGBM |
| Agent orchestration | CrewAI |
| Agent reasoning (LLM) | Mistral, via Ollama (local, free) |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Frontend | React (Vite) |
| Deployment | Plain installable Python package (pip install -e .) — no Docker |

## Project Structure

```
voss-mobility-agent/
├── agents/
│   └── dispatch_crew.py       # CrewAI agents + validation guardrail
├── api/
│   └── main.py                # FastAPI backend (5 endpoints) + serves built frontend
├── data/                       # Synthetic dataset + train/test splits
├── models/                     # Trained model, evaluation report, latest dispatch plan
├── frontend/                   # React dashboard (Vite)
├── docs/
│   ├── research.md
│   └── testing.md
├── database.py                 # SQLAlchemy connection setup
├── db_models.py                 # ORM models (dispatch_runs, dispatch_moves, forecast_log)
├── generate_data.py            # Phase 2: synthetic data generation
├── clean_data.py                # Phase 3: feature engineering + train/test split
├── train_model.py               # Phase 4: LightGBM training
├── evaluate_model.py            # Phase 5: model evaluation
├── test_system.py               # Phase 10: automated tests
├── run_service.py               # Deployment entry point (no Docker)
├── requirements.txt
├── pyproject.toml
└── .env                         # Database credentials (not committed)
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (running locally)
- Ollama with the mistral model pulled:
  ```
  ollama run mistral
  ```

### 1. Clone and set up the Python environment

```bash
git clone <this-repo-url>
cd voss-mobility-agent
python -m venv venv
venv\Scripts\activate          # Windows
```

> Dependencies are installed in Step 5 via `pip install -e .`, which
> reads `pyproject.toml`. `requirements.txt` is kept in sync for
> reference / alternate tooling, but you only need to run one of the
> two install commands — not both.

### 2. Set up the database

```sql
psql -U postgres
CREATE DATABASE voss_mobility;
```

Create a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=voss_mobility
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 3. Build the ML pipeline (one-time)

```bash
python generate_data.py
python clean_data.py
python train_model.py
python evaluate_model.py
```

### 4. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 5. Run everything

```bash
pip install -e .
voss-dispatch
```

Then open:

- Dashboard: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs

No Docker, no containers — just a locally installed Python command.

## Model Performance

Trained on synthetic ride-demand data (4,320 rows: 6 zones × 24 hours × 30 days) with LightGBM:

| Metric | Training | Test (unseen data) |
|---|---|---|
| R² | 0.9066 | 0.8883 |
| MAE | — | 3.32 rides |
| RMSE | — | 4.11 rides |

The small train/test gap (0.018) indicates the model generalizes well and is not overfitting. See [`docs/testing.md`](./docs/testing.md) for full evaluation details.

## Known Limitations

- All data is synthetic — generated to have realistic patterns (rush hours, weekend nightlife demand), not real Voss Mobility data.
- Vehicle availability is simulated with randomized live-feeling variation around fixed baselines, not read from a real GPS fleet-tracking system.
- The system is on-demand, not a continuously streaming real-time service — a natural next step, not part of this project's scope.

See [`docs/testing.md`](./docs/testing.md) for the full list.

## License

This is a portfolio/demo project built for learning purposes.
