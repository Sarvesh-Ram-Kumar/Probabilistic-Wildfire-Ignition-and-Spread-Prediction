"""
Core stochastic simulation: combines BN ignition priors with the spread model
via noisy-OR, one second at a time.
"""

import random
from bn_model import ignition_prior
from spread import spread_probability, describe_cause


def compute_priors(regions, env, hazard_scale=0.04):
    """BN-derived per-second base ignition hazard for each region (own-cause)."""
    priors = {}
    for r in regions:
        p = ignition_prior(
            temperature=env["temperature"],
            rainfall=env["rainfall"],
            lightning=env["lightning"],
            human_proximity=r["human_proximity"],
            tree_density=r["density"],
        )
        # BN gives an overall likelihood; scale down to a per-second hazard so a
        # 50s simulation doesn't trivially ignite every region instantly.
        priors[r["id"]] = p * hazard_scale
    return priors


def run_one_simulation(regions, wall, wind, env, priors, seconds=50, track_timeline=True):
    """
    Runs ONE stochastic timeline over `seconds` steps.
    Each unburned region rolls independently against its own-cause prior and
    against spread from every currently-burning neighbour (noisy-OR combination).

    Returns (timeline, ignited_at, cause).
    Set track_timeline=False for Monte Carlo runs where only final outcomes matter.
    """
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
                if track_timeline:
                    events.append({
                        "id": rid, "cause_type": cause[rid]["type"], "cause_detail": cause[rid]["detail"]
                    })

        if track_timeline:
            timeline.append({"second": t, "events": events})
        if len(on_fire) == len(ids):
            break

    return timeline, ignited_at, cause
