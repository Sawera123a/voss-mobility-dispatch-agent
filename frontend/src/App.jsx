import { useState, useEffect, useCallback } from "react";
import {
  ArrowRight,
  Activity,
  RadioTower,
  TriangleAlert,
  CircleCheck,
  CircleX,
  RefreshCw,
  Info,
  X,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

const ZONES = ["Zone_A", "Zone_B", "Zone_C", "Zone_D", "Zone_E", "Zone_F"];

const ZONE_DISPLAY_NAMES = {
  Zone_A: "Södermalm",
  Zone_B: "Östermalm",
  Zone_C: "Norrmalm (City Center)",
  Zone_D: "Kungsholmen",
  Zone_E: "Vasastan",
  Zone_F: "Gamla Stan",
};

function useClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

function ZoneGauge({ zone, demand, available }) {
  const gap = +(demand - available).toFixed(1);
  const status = gap > 5 ? "shortage" : gap < -1 ? "surplus" : "balanced";
  const max = Math.max(demand, available, 1) * 1.15;
  const fillPct = Math.min(100, (available / max) * 100);
  const demandLinePct = Math.min(100, (demand / max) * 100);

  const plainText =
    status === "shortage"
      ? `Needs about ${Math.round(gap)} more vehicles`
      : status === "surplus"
        ? `Has about ${Math.round(-gap)} vehicles to spare`
        : "Roughly balanced";

  return (
    <div className={`zone-card zone-card--${status}`}>
      <div className="zone-card__head">
        <span className="zone-card__name">
          {ZONE_DISPLAY_NAMES[zone] || zone.replace("Zone_", "Zone ")}
        </span>
        <span className={`zone-card__badge zone-card__badge--${status}`}>
          {status === "shortage"
            ? "SHORT"
            : status === "surplus"
              ? "SURPLUS"
              : "STABLE"}
        </span>
      </div>

      <div className="zone-gauge">
        <div className="zone-gauge__track">
          <div className="zone-gauge__fill" style={{ height: `${fillPct}%` }} />
          <div
            className="zone-gauge__demand-line"
            style={{ bottom: `${demandLinePct}%` }}
          />
        </div>
        <div className="zone-gauge__readout">
          <div className="zone-gauge__stat">
            <span className="zone-gauge__stat-label">Cars here now</span>
            <span className="zone-gauge__stat-value">{available}</span>
          </div>
          <div className="zone-gauge__stat">
            <span className="zone-gauge__stat-label">Expected demand</span>
            <span className="zone-gauge__stat-value">{demand.toFixed(1)}</span>
          </div>
        </div>
      </div>

      <div className={`zone-card__plain zone-card__plain--${status}`}>
        {plainText}
      </div>
    </div>
  );
}

function MoveRow({ move, approved }) {
  return (
    <div
      className={`move-row ${approved ? "move-row--approved" : "move-row--rejected"}`}
    >
      <div className="move-row__icon">
        {approved ? <CircleCheck size={16} /> : <CircleX size={16} />}
      </div>
      <div className="move-row__body">
        <div className="move-row__path">
          <span className="move-row__vehicles">{move.vehicles}</span>
          <span className="move-row__from">
            {move.from_zone.replace("Zone_", "")}
          </span>
          <ArrowRight size={14} className="move-row__arrow" />
          <span className="move-row__to">
            {approved ? move.to_zone.replace("Zone_", "") : "\u2014"}
          </span>
        </div>
        <div className="move-row__reason">
          {approved ? move.reason : move.reason_rejected}
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const clock = useClock();
  const [forecast, setForecast] = useState(null);
  const [availability, setAvailability] = useState({});
  const [plan, setPlan] = useState(null);
  const [loadingForecast, setLoadingForecast] = useState(false);
  const [runningPlan, setRunningPlan] = useState(false);
  const [apiHealthy, setApiHealthy] = useState(null);
  const [error, setError] = useState(null);
  const [showHelp, setShowHelp] = useState(false);
  const [transitStatus, setTransitStatus] = useState(null);

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setApiHealthy(res.ok);
    } catch {
      setApiHealthy(false);
    }
  }, []);

  const loadForecast = useCallback(async () => {
    setLoadingForecast(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/demand-forecast`);
      const data = await res.json();
      setForecast(data.forecast);
      setAvailability(data.availability || {});
    } catch {
      setError("Could not reach the forecast endpoint. Is the API running?");
    } finally {
      setLoadingForecast(false);
    }
  }, []);

  const loadLatestPlan = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/dispatch-plan/latest`);
      const data = await res.json();
      if (data.run_id) setPlan(data);
    } catch {
      /* silent - no plan yet is a normal state */
    }
  }, []);

  const runDispatchPlan = async () => {
    setRunningPlan(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/dispatch-plan/run`, {
        method: "POST",
      });
      const data = await res.json();
      setPlan(data);
      // Use the SAME availability snapshot the agents just reasoned
      // over, instead of re-fetching (which could pull a newer,
      // different snapshot and cause mismatched numbers on screen).
      if (data.availability) {
        setAvailability(data.availability);
      }
    } catch {
      setError(
        "Dispatch run failed. Check that the API and Ollama are running.",
      );
    } finally {
      setRunningPlan(false);
    }
  };

  useEffect(() => {
    checkHealth();
    loadForecast();
    loadLatestPlan();
  }, [checkHealth, loadForecast, loadLatestPlan]);

  useEffect(() => {
    fetch(`${API_BASE}/transit-status`)
      .then((res) => res.json())
      .then(setTransitStatus)
      .catch(() => setTransitStatus({ available: false, departures: [] }));
  }, []);

  // Average next-3-hours demand per zone for the gauges (mirrors backend logic)
  const currentHour = new Date().getHours();
  const zoneAverages = ZONES.reduce((acc, zone) => {
    if (!forecast) {
      acc[zone] = 0;
      return acc;
    }
    const hours = [currentHour, (currentHour + 1) % 24, (currentHour + 2) % 24];
    const rows = forecast.filter(
      (r) => r.zone === zone && hours.includes(r.hour),
    );
    const avg =
      rows.reduce((s, r) => s + r.predicted_demand, 0) / (rows.length || 1);
    acc[zone] = avg;
    return acc;
  }, {});

  return (
    <div className="console">
      <header className="console__topbar">
        <div className="console__brand">
          <RadioTower size={18} className="console__brand-icon" />
          <div>
            <div className="console__brand-name">VOSS MOBILITY</div>
            <div className="console__brand-sub">DISPATCH CONSOLE</div>
          </div>
        </div>

        <div className="console__status">
          <button
            className="btn btn--icon"
            onClick={() => setShowHelp(true)}
            title="What does this do?"
          >
            <Info size={16} />
          </button>
          <span
            className={`status-pill ${apiHealthy ? "status-pill--ok" : "status-pill--down"}`}
          >
            <Activity size={13} />
            {apiHealthy === null
              ? "Checking..."
              : apiHealthy
                ? "API Online"
                : "API Offline"}
          </span>
          <span className="console__clock">
            {clock.toLocaleTimeString("en-GB", { hour12: false })}
          </span>
        </div>
      </header>

      {showHelp && (
        <div className="help-overlay" onClick={() => setShowHelp(false)}>
          <div className="help-card" onClick={(e) => e.stopPropagation()}>
            <div className="help-card__head">
              <h3>What am I looking at?</h3>
              <button
                className="btn btn--icon"
                onClick={() => setShowHelp(false)}
              >
                <X size={16} />
              </button>
            </div>
            <p>
              This screen predicts, for each area of the city, whether there
              will be
              <b> too few cars</b> for the number of rides expected, or
              <b> too many idle cars</b> sitting around.
            </p>
            <p>
              Press <b>"Run Dispatch Agents"</b> and an AI planner decides which
              idle cars should move to the areas that need them - so customers
              wait less, and cars don't sit empty.
            </p>
            <p>
              Every suggested move is double-checked by a safety rule before
              it's shown as approved, so a car can never be sent from an area
              that doesn't actually have spare cars.
            </p>
          </div>
        </div>
      )}

      {forecast && (
        <div className="summary-strip">
          {(() => {
            const shortCount = ZONES.filter(
              (z) => (zoneAverages[z] || 0) - (availability[z] ?? 0) > 5,
            ).length;
            const surplusCount = ZONES.filter(
              (z) => (zoneAverages[z] || 0) - (availability[z] ?? 0) < -1,
            ).length;
            return (
              <>
                <span className="summary-strip__item summary-strip__item--short">
                  {shortCount} area{shortCount === 1 ? "" : "s"} need more cars
                </span>
                <span className="summary-strip__divider">&middot;</span>
                <span className="summary-strip__item summary-strip__item--surplus">
                  {surplusCount} area{surplusCount === 1 ? "" : "s"} have spare
                  cars
                </span>
              </>
            );
          })()}
        </div>
      )}

      <main className="console__main">
        <section className="console__section">
          <div className="console__section-head">
            <h2>
              Stockholm Zone Status{" "}
              <span className="console__section-sub">
                next 3 hours, predicted
              </span>
            </h2>
            <button
              className="btn btn--ghost"
              onClick={loadForecast}
              disabled={loadingForecast}
            >
              <RefreshCw size={14} className={loadingForecast ? "spin" : ""} />
              Refresh
            </button>
          </div>

          <div className="zone-grid">
            {ZONES.map((zone) => (
              <ZoneGauge
                key={zone}
                zone={zone}
                demand={zoneAverages[zone] || 0}
                available={availability[zone] ?? 0}
              />
            ))}
          </div>
        </section>

        {transitStatus?.available && transitStatus.departures.length > 0 && (
          <section className="console__section">
            <div className="console__section-head">
              <h2>
                Live Transit Context{" "}
                <span className="console__section-sub">
                  Stockholm Central, via Trafiklab
                </span>
              </h2>
            </div>
            <div className="transit-list">
              {transitStatus.departures.map((dep, i) => (
                <div key={i} className="transit-item">
                  <span className="transit-item__line">{dep.line}</span>
                  <span className="transit-item__direction">
                    → {dep.direction}
                  </span>
                  <span
                    className={`transit-item__time ${dep.delayed ? "transit-item__time--delayed" : ""}`}
                  >
                    {dep.time}
                    {dep.delayed ? " (delayed)" : ""}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        <section className="console__section console__section--panel">
          <div className="console__section-head">
            <h2>
              Dispatch Plan{" "}
              <span className="console__section-sub">
                agent-generated, guardrail-checked
              </span>
            </h2>
            <button
              className="btn btn--primary"
              onClick={runDispatchPlan}
              disabled={runningPlan}
            >
              {runningPlan ? "Running agents..." : "Run Dispatch Agents"}
            </button>
          </div>

          {error && (
            <div className="alert-banner">
              <TriangleAlert size={15} />
              {error}
            </div>
          )}

          {runningPlan && (
            <div className="thinking-banner">
              <span className="thinking-dot" />
              Demand Analyst and Dispatch Planner are reasoning locally via
              Mistral... this can take up to 2 minutes.
            </div>
          )}

          {!plan && !runningPlan && (
            <div className="empty-state">
              No dispatch plan yet. Run the agents to generate one.
            </div>
          )}

          {plan && (
            <div className="plan-body">
              <div className="plan-meta">
                Run #{plan.run_id}
                {plan.created_at
                  ? ` \u2014 ${new Date(plan.created_at).toLocaleString()}`
                  : ""}
              </div>

              <div className="plan-columns">
                <div className="plan-column">
                  <div className="plan-column__label plan-column__label--approved">
                    Approved moves ({plan.approved_moves.length})
                  </div>
                  {plan.approved_moves.length === 0 && (
                    <div className="empty-state empty-state--small">
                      No moves approved this cycle.
                    </div>
                  )}
                  {plan.approved_moves.map((m, i) => (
                    <MoveRow key={i} move={m} approved />
                  ))}
                </div>

                <div className="plan-column">
                  <div className="plan-column__label plan-column__label--rejected">
                    Rejected by guardrail ({plan.rejected_moves.length})
                  </div>
                  {plan.rejected_moves.length === 0 && (
                    <div className="empty-state empty-state--small">
                      No rejections this cycle.
                    </div>
                  )}
                  {plan.rejected_moves.map((m, i) => (
                    <MoveRow key={i} move={m} approved={false} />
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="console__footer">
        Autonomous Ride Dispatch Agent &middot; LightGBM demand model + CrewAI
        agents (local Mistral) &middot; Sweden
      </footer>
    </div>
  );
}
