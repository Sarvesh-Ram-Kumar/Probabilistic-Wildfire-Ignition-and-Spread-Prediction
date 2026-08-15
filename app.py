from flask import Flask, render_template, request, jsonify
import math
import random
from bn_model import ignition_prior, STATES_3, STATES_2

app = Flask(__name__)

STATE = {
    "regions": [],   # [{id, x, y, radius, density, human_proximity}]
    "wall": None,
    "wind": {"angle": 0, "strength": 0.5},
    "env": {"temperature": "medium", "rainfall": "medium", "lightning": "no"},  # shared weather
    "next_id": 1
}

DENSITY_TO_FUEL = {"low": 0.3, "medium": 0.6, "high": 1.0}


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


# ---------- Geometry / spread ----------

def segments_intersect(p1, p2, p3, p4):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def wall_factor(i, j, wall, wind_strength):
    if wall is None:
        return 1.0
    p1, p2 = (i["x"], i["y"]), (j["x"], j["y"])
    p3, p4 = (wall["x1"], wall["y1"]), (wall["x2"], wall["y2"])
    if segments_intersect(p1, p2, p3, p4):
        base_block, wind_penetration = 0.85, 0.5
        return max(0.05, 1 - base_block * (1 - wind_penetration * wind_strength))
    return 1.0


def spread_probability(i, j, wind, wall, lam=140.0, base_rate=0.5):
    """Per-second probability fire spreads from burning region i to unburned region j."""
    dx, dy = j["x"] - i["x"], j["y"] - i["y"]
    center_dist = math.hypot(dx, dy)
    edge_dist = max(1.0, center_dist - i["radius"] - j["radius"])
    if center_dist == 0:
        return 0.0

    distance_decay = math.exp(-edge_dist / lam)

    wind_rad = math.radians(wind["angle"])
    wind_vec = (math.cos(wind_rad), math.sin(wind_rad))
    to_j = (dx / center_dist, dy / center_dist)
    cos_theta = wind_vec[0] * to_j[0] + wind_vec[1] * to_j[1]
    wind_factor = max(0.05, 1 + wind["strength"] * cos_theta)

    fuel_j = DENSITY_TO_FUEL.get(j["density"], 0.6)
    wf = wall_factor(i, j, wall, wind["strength"])

    p = base_rate * distance_decay * wind_factor * fuel_j * wf
    return min(0.95, max(0.0, p))


# ---------- Simulation (time-stepped, BN + spread, cause attribution) ----------

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
    wall = STATE["wall"]
    wind = STATE["wind"]
    env = STATE["env"]

    if not regions:
        return jsonify({"error": "no regions placed"}), 400

    # precompute each region's BN-derived per-second base ignition hazard
    priors = {}
    for r in regions:
        p = ignition_prior(
            temperature=env["temperature"],
            rainfall=env["rainfall"],
            lightning=env["lightning"],
            human_proximity=r["human_proximity"],
            tree_density=r["density"],
        )
        # BN gives an overall likelihood; treat a scaled-down fraction as the per-second
        # hazard so a 50s simulation doesn't trivially ignite every region instantly
        priors[r["id"]] = p * 0.04

    by_id = {r["id"]: r for r in regions}
    ids = [r["id"] for r in regions]

    on_fire = set()
    ignited_at = {}
    cause = {}

    timeline = []

    for t in range(1, seconds + 1):
        events = []
        currently_burning = list(on_fire)

        for rid in ids:
            if rid in on_fire:
                continue
            r = by_id[rid]

            own_ignite = random.random() < priors[rid]

            spread_ignite = False
            triggering_neighbor = None
            best_p = -1
            for bid in currently_burning:
                p_sp = spread_probability(by_id[bid], r, wind, wall)
                if random.random() < p_sp:
                    spread_ignite = True
                    if p_sp > best_p:
                        best_p = p_sp
                        triggering_neighbor = bid

            if own_ignite or spread_ignite:
                on_fire.add(rid)
                ignited_at[rid] = t
                if own_ignite and not spread_ignite:
                    cause[rid] = {"type": "natural/human cause", "detail": describe_cause(env, r)}
                elif spread_ignite and not own_ignite:
                    cause[rid] = {"type": "spread", "detail": f"spread from region #{triggering_neighbor}"}
                else:
                    cause[rid] = {"type": "both", "detail": f"own cause AND spread from region #{triggering_neighbor}"}
                events.append({"id": rid, "cause_type": cause[rid]["type"], "cause_detail": cause[rid]["detail"]})

        timeline.append({"second": t, "events": events})
        if len(on_fire) == len(ids):
            break

    summary = []
    for rid in ids:
        summary.append({
            "id": rid,
            "ignited": rid in ignited_at,
            "ignited_at": ignited_at.get(rid),
            "cause_type": cause.get(rid, {}).get("type"),
            "cause_detail": cause.get(rid, {}).get("detail"),
            "base_prior": round(priors[rid], 4)
        })

    return jsonify({"timeline": timeline, "summary": summary, "seconds_run": len(timeline)})


def describe_cause(env, region):
    reasons = []
    if env["lightning"] == "yes":
        reasons.append("lightning")
    if region["human_proximity"] == "yes":
        reasons.append("human activity")
    if env["temperature"] == "high" and env["rainfall"] == "low":
        reasons.append("hot/dry conditions")
    if region["density"] == "high":
        reasons.append("dense fuel load")
    if not reasons:
        reasons.append("background risk")
    return " + ".join(reasons)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
