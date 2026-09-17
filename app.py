"""
Flask app: HTTP routes only. All fire-reasoning logic lives in:
  bn_model.py    - Bayesian Network (ignition causes, exact inference)
  geometry.py    - segment-intersection helper
  spread.py      - region-to-region spread model (MRF-style) + rock_factor
  simulation.py  - single-run time-stepped simulation (noisy-OR combination)
  montecarlo.py  - aggregation over many runs + what-if comparison
"""

from flask import Flask, render_template, request, jsonify
from bn_model import STATES_3, STATES_2
from simulation import compute_priors, run_one_simulation
from montecarlo import analyze as mc_analyze, compare_wall as mc_compare_wall
import random

app = Flask(__name__)

STATE = {
    "regions": [],   # [{id, x, y, radius, density, human_proximity}]
    "wall": None,
    "wind": {"angle": 0, "strength": 0.5},
    "env": {"temperature": "medium", "rainfall": "medium", "lightning": "no"},
    "next_id": 1
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(STATE)


# ---------- Regions ----------

@app.route("/api/region", methods=["POST"])
def add_region():
    data = request.get_json(force=True)
    x, y = data.get("x"), data.get("y")
    if x is None or y is None:
        return jsonify({"error": "x and y required"}), 400

    region = {
        "id": STATE["next_id"],
        "x": x,
        "y": y,
        "radius": float(data.get("radius", 30)),
        "density": data.get("density", "medium") if data.get("density") in STATES_3 else "medium",
        "human_proximity": data.get("human_proximity", "no") if data.get("human_proximity") in STATES_2 else "no",
    }
    STATE["regions"].append(region)
    STATE["next_id"] += 1
    return jsonify(region)


@app.route("/api/region/<int:region_id>", methods=["PATCH"])
def update_region(region_id):
    data = request.get_json(force=True)
    for r in STATE["regions"]:
        if r["id"] == region_id:
            if "radius" in data:
                r["radius"] = float(data["radius"])
            if "density" in data and data["density"] in STATES_3:
                r["density"] = data["density"]
            if "human_proximity" in data and data["human_proximity"] in STATES_2:
                r["human_proximity"] = data["human_proximity"]
            return jsonify(r)
    return jsonify({"error": "not found"}), 404


@app.route("/api/region/<int:region_id>", methods=["DELETE"])
def delete_region(region_id):
    STATE["regions"] = [r for r in STATE["regions"] if r["id"] != region_id]
    return jsonify({"ok": True})


@app.route("/api/clear", methods=["POST"])
def clear_all():
    STATE["regions"] = []
    STATE["wall"] = None
    STATE["next_id"] = 1
    return jsonify({"ok": True})


# ---------- Wall / wind / env ----------

@app.route("/api/wall", methods=["POST"])
def set_wall():
    data = request.get_json(force=True)
    STATE["wall"] = {"x1": data["x1"], "y1": data["y1"], "x2": data["x2"], "y2": data["y2"]}
    return jsonify(STATE["wall"])


@app.route("/api/wall", methods=["DELETE"])
def clear_wall():
    STATE["wall"] = None
    return jsonify({"ok": True})


@app.route("/api/wind", methods=["POST"])
def set_wind():
    data = request.get_json(force=True)
    STATE["wind"]["angle"] = float(data.get("angle", STATE["wind"]["angle"]))
    STATE["wind"]["strength"] = float(data.get("strength", STATE["wind"]["strength"]))
    return jsonify(STATE["wind"])


@app.route("/api/env", methods=["POST"])
def set_env():
    data = request.get_json(force=True)
    if data.get("temperature") in STATES_3:
        STATE["env"]["temperature"] = data["temperature"]
    if data.get("rainfall") in STATES_3:
        STATE["env"]["rainfall"] = data["rainfall"]
    if data.get("lightning") in STATES_2:
        STATE["env"]["lightning"] = data["lightning"]
    return jsonify(STATE["env"])


# ---------- Single-run simulation ----------

@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Runs ONE stochastic timeline over `seconds` steps, reporting per-second ignition
    events with attributed cause (own BN-driven cause vs. spread from a neighbor).
    body: { seconds: int (default 50), seed: optional int }
    """
    data = request.get_json(force=True) if request.data else {}
    seconds = int(data.get("seconds", 50))
    seed = data.get("seed")
    if seed is not None:
        random.seed(int(seed))

    regions = STATE["regions"]
    if not regions:
        return jsonify({"error": "no regions placed"}), 400

    wall, wind, env = STATE["wall"], STATE["wind"], STATE["env"]
    priors = compute_priors(regions, env)

    timeline, ignited_at, cause = run_one_simulation(
        regions, wall, wind, env, priors, seconds=seconds, track_timeline=True
    )

    summary = []
    for r in regions:
        rid = r["id"]
        summary.append({
            "id": rid,
            "ignited": rid in ignited_at,
            "ignited_at": ignited_at.get(rid),
            "cause_type": cause.get(rid, {}).get("type"),
            "cause_detail": cause.get(rid, {}).get("detail"),
            "base_prior": round(priors[rid], 4)
        })

    return jsonify({"timeline": timeline, "summary": summary, "seconds_run": len(timeline)})


# ---------- Monte Carlo aggregation ----------

@app.route("/api/analyze", methods=["POST"])
def analyze_route():
    """
    Monte Carlo analysis: runs the simulation N times and aggregates into
    probability estimates (P(ignite), P(last), cause split, expected time, etc.)
    body: { runs: int (default 300), seconds: int (default 50), seed: optional int }
    """
    data = request.get_json(force=True) if request.data else {}
    runs = max(1, min(5000, int(data.get("runs", 300))))
    seconds = int(data.get("seconds", 50))
    seed = data.get("seed")
    if seed is not None:
        random.seed(int(seed))

    regions = STATE["regions"]
    if not regions:
        return jsonify({"error": "no regions placed"}), 400

    landscape, results = mc_analyze(
        regions, STATE["wall"], STATE["wind"], STATE["env"], runs=runs, seconds=seconds
    )
    return jsonify({"landscape": landscape, "results": results})


# ---------- What-if: firebreak comparison ----------

@app.route("/api/compare_wall", methods=["POST"])
def compare_wall_route():
    """
    Runs Monte Carlo analysis with and without the current firebreak and reports
    the difference in expected damage (decision-network utility comparison).
    body: { runs: int (default 300), seconds: int (default 50) }
    """
    data = request.get_json(force=True) if request.data else {}
    runs = max(1, min(5000, int(data.get("runs", 300))))
    seconds = int(data.get("seconds", 50))

    regions = STATE["regions"]
    if not regions:
        return jsonify({"error": "no regions placed"}), 400
    if STATE["wall"] is None:
        return jsonify({"error": "no firebreak drawn - draw one first to compare"}), 400

    result = mc_compare_wall(
        regions, STATE["wall"], STATE["wind"], STATE["env"], runs=runs, seconds=seconds
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)