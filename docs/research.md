# Research — Autonomous Ride Dispatch Agent (Voss Mobility)

## 1. Business Problem

Voss Mobility (Sweden) currently rebalances its ride-sharing fleet **manually**. Ops staff
watch demand and decide where to move idle vehicles. This causes:

- Longer customer wait times in high-demand areas
- Idle vehicles sitting in low-demand areas
- Lost revenue due to poor vehicle distribution

**Goal:** Build an AI agent that autonomously predicts demand and plans vehicle
repositioning — removing the manual decision-making step.

---

## 2. What is an AI Agent?

An AI agent is an autonomous entity that can:
- **Perceive** its environment (read data/context)
- **Decide** based on that data (reasoning, not just responding)
- **Act** to achieve a goal (produce a decision, call a tool, trigger a process)
- Maintain context and take initiative, unlike a simple chatbot that only reacts to prompts

This is different from a normal LLM chatbot, which only answers a single prompt with no
persistence, memory, or ability to independently pursue a goal.

---

## 3. What is CrewAI?

CrewAI is a lightweight Python framework used to orchestrate multiple AI agents working
together as a **crew** (team) to complete a shared goal.

### Core building blocks

| Concept | Meaning |
|---|---|
| **Agent** | An AI worker with a defined role, goal, and backstory (e.g. "Demand Analyst") |
| **Task** | A specific job assigned to an agent |
| **Tool** | Something an agent can use (API call, database query, calculation, etc.) |
| **Crew** | The full team of agents + tasks working together toward one outcome |

CrewAI supports **sequential** or **hierarchical** processes — agents can run one after
another, or a manager agent can delegate to others.

### Requirements
- Python **3.10 – 3.13** (project uses 3.12.8 ✅ confirmed via `python --version`)
- Uses **uv** as the recommended package manager (faster than pip)
- Install via: `pip install crewai`

### LLM Flexibility
CrewAI defaults to OpenAI's GPT-4, but can connect to **local, free LLMs** (e.g. Llama 3,
Mistral) through **Ollama** — meaning no paid API subscription is required for this
project, consistent with the "no subscription" constraint used elsewhere in this project
(similar to the local n8n setup).

---

## 4. Planned Agent Architecture for This Project

Two agents working in a sequential crew:

1. **Demand Predictor Agent**
   - Reads output from the LightGBM demand-prediction model
   - Summarizes where/when demand is expected to spike

2. **Dispatch Planner Agent**
   - Takes the Demand Predictor's output
   - Reasons about current vehicle locations (idle vs. busy)
   - Produces a repositioning plan (which vehicles should move where)

```
Historical Ride Data
        │
        ▼
LightGBM Model (demand prediction)
        │
        ▼
Demand Predictor Agent (CrewAI)
        │
        ▼
Dispatch Planner Agent (CrewAI)
        │
        ▼
FastAPI endpoint → React frontend (shows repositioning plan)
```

---

## 5. Tech Stack Confirmed

| Layer | Tool |
|---|---|
| Agent orchestration | CrewAI |
| Agent reasoning (LLM) | Llama 3 / Mistral via Ollama (local, free) |
| Demand prediction model | LightGBM |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Frontend | React |
| Deployment | **No Docker** — packaged as an installable Python service (engineering challenge) |

---

## 6. Environment Check (Completed)

- ✅ Project folder created: `C:\voss-mobility-agent`
- ✅ Sub-folders created: `data/`, `models/`, `api/`, `frontend/`, `docs/`
- ✅ Python confirmed installed: **Python 3.12.8**

---

## 7. Research Sources

- CrewAI official documentation (docs.crewai.com)
- CrewAI GitHub repository and examples
- IBM "What is CrewAI?" overview
- DigitalOcean CrewAI practical guide

---

## 8. Next Steps (Phase 2)

Collect a ride-share dataset (NYC/Chicago public data from Kaggle, or synthetic demand
data) to use for training the LightGBM demand-prediction model.
