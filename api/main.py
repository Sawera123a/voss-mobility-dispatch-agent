import sys
import os

# Allow importing agents/dispatch_crew.py from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import engine, Base, get_db
import db_models as models
from agents.dispatch_crew import build_forecast, run_dispatch_cycle, save_report, ZONE_MAPPING, get_current_availability

import httpx

# Create tables in PostgreSQL if they don't exist yet
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Voss Mobility Dispatch API",
    description="AI-powered demand prediction and vehicle repositioning for Voss Mobility.",
    version="1.0.0",
)

# Allow the React frontend to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAFIKLAB_API_KEY = os.getenv("TRAFIKLAB_API_KEY")
STOCKHOLM_CENTRAL_STOP_ID = "740000001" 

FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)


@app.get("/health")
def health_check():
    """Simple check to confirm the API is running."""
    return {"status": "ok", "service": "Voss Mobility Dispatch API"}


@app.get("/demand-forecast")
def get_demand_forecast(db: Session = Depends(get_db)):
    """
    Returns predicted ride demand for every zone, every hour of the
    day (using the trained LightGBM model from Phase 4), and logs
    each prediction to the database.
    """
    forecast_df = build_forecast()

    # Log this run to the database
    for _, row in forecast_df.iterrows():
        db.add(models.DemandForecastLog(
            zone=row["zone"], hour=int(row["hour"]),
            predicted_demand=float(row["predicted_demand"]),
        ))
    db.commit()

    return {
        "zones": list(ZONE_MAPPING.keys()),
        "forecast": forecast_df.to_dict(orient="records"),
        "availability": get_current_availability(),
    }


@app.get("/transit-status")
def get_transit_status():
    """
    Returns upcoming public transit departures from Stockholm Central,
    via Trafiklab's ResRobot API. Shown on the dashboard purely as
    real-world context alongside dispatch decisions - it does NOT feed
    into the demand model, the AI agents, or the guardrail. If no API
    key is configured or the request fails, returns available=False
    instead of breaking the dashboard.
    """
    if not TRAFIKLAB_API_KEY:
        return {"available": False, "departures": []}
    try:
        response = httpx.get(
            "https://api.resrobot.se/v2.1/departureBoard",
            params={
                "id": STOCKHOLM_CENTRAL_STOP_ID,
                "format": "json",
                "accessId": TRAFIKLAB_API_KEY,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return {"available": False, "departures": []}
    departures = []
    for d in data.get("Departure", [])[:8]:
        scheduled_time = d.get("time", "")
        rt_time = d.get("rtTime")
        delayed = bool(rt_time and rt_time != scheduled_time)
        departures.append({
            "line": d.get("name", "Unknown"),
            "stop": d.get("stop", ""),
            "direction": d.get("direction", ""),
            "time": scheduled_time,
            "rt_time": rt_time,
            "delayed": delayed,
        })
    return {"available": True, "departures": departures}


@app.post("/dispatch-plan/run")
def trigger_dispatch_plan(current_hour: int = None, db: Session = Depends(get_db)):
    """
    Triggers the two CrewAI agents (Demand Analyst + Dispatch Planner)
    to analyze current conditions and produce a validated vehicle
    repositioning plan. This can take 30 seconds - 2 minutes since it
    runs a local LLM (Mistral via Ollama).

    Saves the full result (approved + rejected moves) to PostgreSQL
    and to models/dispatch_plan.txt.
    """
    cycle_result = run_dispatch_cycle(current_hour=current_hour)
    save_report(cycle_result)

    # Save to database
    run = models.DispatchRun(
        forecast_summary=cycle_result["forecast_summary"],
        raw_agent_output=cycle_result["raw_agent_output"],
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    for m in cycle_result["validated_moves"]:
        db.add(models.DispatchMove(
            run_id=run.id, vehicles=m["vehicles"], from_zone=m["from_zone"],
            to_zone=m["to_zone"], reason=m["reason"], approved=True,
        ))
    for m in cycle_result["rejected_moves"]:
        db.add(models.DispatchMove(
            run_id=run.id, vehicles=m["vehicles"], from_zone=m["from_zone"],
            to_zone=m["to_zone"], reason=None, approved=False,
            rejection_reason=m["reason_rejected"],
        ))
    db.commit()

    return {
        "run_id": run.id,
        "forecast_summary": cycle_result["forecast_summary"],
        "availability": cycle_result["availability"],
        "approved_moves": cycle_result["validated_moves"],
        "rejected_moves": cycle_result["rejected_moves"],
    }


@app.get("/dispatch-plan/history")
def get_dispatch_history(limit: int = 10, db: Session = Depends(get_db)):
    """Returns the most recent dispatch runs, newest first."""
    runs = (
        db.query(models.DispatchRun)
        .order_by(models.DispatchRun.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "run_id": r.id,
            "created_at": r.created_at.isoformat(),
            "approved_moves": [
                {"vehicles": m.vehicles, "from_zone": m.from_zone,
                 "to_zone": m.to_zone, "reason": m.reason}
                for m in r.moves if m.approved
            ],
            "rejected_count": sum(1 for m in r.moves if not m.approved),
        }
        for r in runs
    ]


@app.get("/dispatch-plan/latest")
def get_latest_dispatch_plan(db: Session = Depends(get_db)):
    """Returns the most recent dispatch run in full detail."""
    run = (
        db.query(models.DispatchRun)
        .order_by(models.DispatchRun.created_at.desc())
        .first()
    )
    if not run:
        return {"message": "No dispatch runs yet. POST to /dispatch-plan/run first."}

    return {
        "run_id": run.id,
        "created_at": run.created_at.isoformat(),
        "forecast_summary": run.forecast_summary,
        "approved_moves": [
            {"vehicles": m.vehicles, "from_zone": m.from_zone,
             "to_zone": m.to_zone, "reason": m.reason}
            for m in run.moves if m.approved
        ],
        "rejected_moves": [
            {"vehicles": m.vehicles, "from_zone": m.from_zone,
             "to_zone": m.to_zone, "rejection_reason": m.rejection_reason}
            for m in run.moves if not m.approved
        ],
    }


# Serve the built React frontend (production deployment, no Docker)

if os.path.isdir(FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        """
        Serves the built frontend. If the requested path matches a real
        file inside dist/ (favicon.svg, robots.txt, etc.), that exact
        file is returned. Otherwise falls back to index.html, so React
        Router-style client-side routes still load the app correctly.
        """
        requested_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(requested_path):
            return FileResponse(requested_path)

        index_path = os.path.join(FRONTEND_DIST, "index.html")
        return FileResponse(index_path)
