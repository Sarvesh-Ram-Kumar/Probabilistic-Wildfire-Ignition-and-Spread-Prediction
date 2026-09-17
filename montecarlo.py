"""
Monte Carlo aggregation layer: runs the single-run simulation many times and
turns raw outcomes into probability estimates and decision-network style
comparisons (Objectives 2, 3 and 4).
"""

import statistics
from simulation import compute_priors, run_one_simulation


def analyze(regions, wall, wind, env, runs=300, seconds=50):
    """
    Runs `runs` independent simulations and aggregates outcomes.
    Returns (landscape_dict, per_region_results_list).
    """
    priors = compute_priors(regions, env)
    ids = [r["id"] for r in regions]

    ignite_count = {rid: 0 for rid in ids}
    last_count = {rid: 0 for rid in ids}
    cause_count = {rid: {"spread": 0, "natural/human cause": 0, "both": 0} for rid in ids}
    ignite_times = {rid: [] for rid in ids}

    burned_per_run = []
    all_ignite_runs = 0
    none_ignite_runs = 0

    for _ in range(runs):
        _, ignited_at, cause = run_one_simulation(
            regions, wall, wind, env, priors, seconds=seconds, track_timeline=False
        )

        burned_per_run.append(len(ignited_at))
        if len(ignited_at) == len(ids):
            all_ignite_runs += 1
        if len(ignited_at) == 0:
            none_ignite_runs += 1

        for rid, t in ignited_at.items():
            ignite_count[rid] += 1
            ignite_times[rid].append(t)
            ctype = cause[rid]["type"]
            if ctype in cause_count[rid]:
                cause_count[rid][ctype] += 1

        if ignited_at:
            last_t = max(ignited_at.values())
            for rid, t in ignited_at.items():
                if t == last_t:
                    last_count[rid] += 1

    results = []
    for r in regions:
        rid = r["id"]
        n_ig = ignite_count[rid]
        times = ignite_times[rid]
        cc = cause_count[rid]
        results.append({
            "id": rid,
            "p_ignite": round(n_ig / runs, 4),
            "p_last": round(last_count[rid] / runs, 4),
            "p_cause_spread": round(cc["spread"] / n_ig, 4) if n_ig else None,
            "p_cause_own": round(cc["natural/human cause"] / n_ig, 4) if n_ig else None,
            "p_cause_both": round(cc["both"] / n_ig, 4) if n_ig else None,
            "mean_ignite_time": round(statistics.mean(times), 2) if times else None,
            "std_ignite_time": round(statistics.pstdev(times), 2) if len(times) > 1 else 0.0,
            "base_prior": round(priors[rid], 4),
            "density": r["density"],
            "human_proximity": r["human_proximity"],
        })

    expected_burned = statistics.mean(burned_per_run)
    landscape = {
        "runs": runs,
        "seconds": seconds,
        "n_regions": len(ids),
        "expected_regions_burned": round(expected_burned, 3),
        "std_regions_burned": round(statistics.pstdev(burned_per_run), 3) if runs > 1 else 0.0,
        "p_all_ignite": round(all_ignite_runs / runs, 4),
        "p_none_ignite": round(none_ignite_runs / runs, 4),
        "utility": round(-expected_burned, 3),  # decision-network utility: higher (less negative) is better
        "wall_present": wall is not None,
        "wind": dict(wind),
        "env": dict(env),
    }
    return landscape, results


def compare_wall(regions, wall, wind, env, runs=300, seconds=50):
    """
    Runs the Monte Carlo analysis twice - with and without the given wall - and
    reports the difference in expected damage (What-If / Decision objective).
    """
    priors = compute_priors(regions, env)

    def expected_burned(w):
        burned = []
        for _ in range(runs):
            _, ignited_at, _ = run_one_simulation(
                regions, w, wind, env, priors, seconds=seconds, track_timeline=False
            )
            burned.append(len(ignited_at))
        return statistics.mean(burned)

    with_wall = expected_burned(wall)
    without_wall = expected_burned(None)

    return {
        "runs": runs,
        "seconds": seconds,
        "n_regions": len(regions),
        "expected_burned_with_wall": round(with_wall, 3),
        "expected_burned_without_wall": round(without_wall, 3),
        "regions_saved": round(without_wall - with_wall, 3),
        "utility_with_wall": round(-with_wall, 3),
        "utility_without_wall": round(-without_wall, 3),
        "utility_gain": round(without_wall - with_wall, 3),
    }
