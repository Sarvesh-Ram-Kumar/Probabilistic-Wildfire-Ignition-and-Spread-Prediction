"""
Region-to-region spread model (the MRF-style pairwise component).
spread_prob(i->j) = base_rate x distance_decay x wind_factor x fuel(j) x rock_factor
"""

import math
from geometry import segments_intersect

DENSITY_TO_FUEL = {"low": 0.3, "medium": 0.6, "high": 1.0}


def rock_factor(i, j, wall, wind_strength):
    """
    Barrier (rock wall / firebreak) blocking factor for the i->j spread path.
    1.0 if no barrier intersects the path; reduced otherwise.
    Strong wind partially defeats the barrier (embers carry over).
    """
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
    if center_dist == 0:
        return 0.0
    edge_dist = max(1.0, center_dist - i["radius"] - j["radius"])

    distance_decay = math.exp(-edge_dist / lam)

    wind_rad = math.radians(wind["angle"])
    wind_vec = (math.cos(wind_rad), math.sin(wind_rad))
    to_j = (dx / center_dist, dy / center_dist)
    cos_theta = wind_vec[0] * to_j[0] + wind_vec[1] * to_j[1]
    wind_factor = max(0.05, 1 + wind["strength"] * cos_theta)

    fuel_j = DENSITY_TO_FUEL.get(j["density"], 0.6)
    rf = rock_factor(i, j, wall, wind["strength"])

    p = base_rate * distance_decay * wind_factor * fuel_j * rf
    return min(0.95, max(0.0, p))


def describe_cause(env, region):
    """Human-readable explanation for an own-cause ignition (used in single-run timeline)."""
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
