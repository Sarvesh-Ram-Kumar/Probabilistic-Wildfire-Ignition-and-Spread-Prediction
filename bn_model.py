"""
Bayesian Network for per-region ignition priors.

Structure (3 levels):
LEVEL 0 (roots):    Temperature, Rainfall, Lightning, HumanProximity, TreeDensity
LEVEL 1 (derived):  DryVegetation (<- Temperature, Rainfall)
                     HumanActivity (<- HumanProximity)
LEVEL 2 (output):   Ignition (<- DryVegetation, HumanActivity, Lightning, TreeDensity)
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools

STATES_3 = ["low", "medium", "high"]
STATES_2 = ["no", "yes"]


def _logistic(x):
    import math
    return 1 / (1 + math.exp(-x))


def build_dryveg_cpd():
    temp_score = {"low": 0.0, "medium": 1.0, "high": 2.0}
    rain_score = {"low": 2.0, "medium": 1.0, "high": 0.0}

    values = [[], [], []]
    for temp in STATES_3:
        for rain in STATES_3:
            score = temp_score[temp] + rain_score[rain]
            high_p = _logistic((score - 2.0) * 1.5)
            low_p = _logistic(-(score - 2.0) * 1.5)
            med_p = max(0.05, 1 - high_p - low_p)
            total = high_p + med_p + low_p
            values[0].append(low_p / total)
            values[1].append(med_p / total)
            values[2].append(high_p / total)

    return TabularCPD(
        variable="DryVegetation", variable_card=3, values=values,
        evidence=["Temperature", "Rainfall"], evidence_card=[3, 3],
        state_names={"DryVegetation": STATES_3, "Temperature": STATES_3, "Rainfall": STATES_3}
    )


def build_humanactivity_cpd():
    return TabularCPD(
        variable="HumanActivity", variable_card=2,
        values=[[0.9, 0.35], [0.1, 0.65]],
        evidence=["HumanProximity"], evidence_card=[2],
        state_names={"HumanActivity": ["low", "high"], "HumanProximity": STATES_2}
    )


def build_ignition_cpd():
    dryveg_w = {"low": 0.0, "medium": 1.0, "high": 2.2}
    human_w = {"low": 0.0, "high": 1.4}
    lightning_w = {"no": 0.0, "yes": 1.8}
    density_w = {"low": 0.0, "medium": 0.6, "high": 1.2}

    combos = list(itertools.product(STATES_3, ["low", "high"], STATES_2, STATES_3))
    no_row, yes_row = [], []
    for dv, ha, lt, td in combos:
        score = dryveg_w[dv] + human_w[ha] + lightning_w[lt] + density_w[td]
        p_yes = _logistic((score - 3.0) * 1.1)
        p_yes = min(0.97, max(0.01, p_yes))
        yes_row.append(p_yes)
        no_row.append(1 - p_yes)

    return TabularCPD(
        variable="Ignition", variable_card=2,
        values=[no_row, yes_row],
        evidence=["DryVegetation", "HumanActivity", "Lightning", "TreeDensity"],
        evidence_card=[3, 2, 2, 3],
        state_names={
            "Ignition": STATES_2, "DryVegetation": STATES_3,
            "HumanActivity": ["low", "high"], "Lightning": STATES_2, "TreeDensity": STATES_3
        }
    )


def build_network():
    model = DiscreteBayesianNetwork([
        ("Temperature", "DryVegetation"),
        ("Rainfall", "DryVegetation"),
        ("HumanProximity", "HumanActivity"),
        ("DryVegetation", "Ignition"),
        ("HumanActivity", "Ignition"),
        ("Lightning", "Ignition"),
        ("TreeDensity", "Ignition"),
    ])

    uniform3 = TabularCPD("Temperature", 3, [[1/3], [1/3], [1/3]], state_names={"Temperature": STATES_3})
    rainfall_cpd = TabularCPD("Rainfall", 3, [[1/3], [1/3], [1/3]], state_names={"Rainfall": STATES_3})
    lightning_cpd = TabularCPD("Lightning", 2, [[0.85], [0.15]], state_names={"Lightning": STATES_2})
    human_prox_cpd = TabularCPD("HumanProximity", 2, [[0.5], [0.5]], state_names={"HumanProximity": STATES_2})
    tree_density_cpd = TabularCPD("TreeDensity", 3, [[1/3], [1/3], [1/3]], state_names={"TreeDensity": STATES_3})

    model.add_cpds(
        uniform3, rainfall_cpd, lightning_cpd, human_prox_cpd, tree_density_cpd,
        build_dryveg_cpd(), build_humanactivity_cpd(), build_ignition_cpd()
    )
    assert model.check_model()
    return model


_MODEL = build_network()
_INFER = VariableElimination(_MODEL)
_PRIOR_CACHE = {}


def ignition_prior(temperature, rainfall, lightning, human_proximity, tree_density):
    """Exact inference (variable elimination) for P(Ignition=yes | evidence)."""
    key = (temperature, rainfall, lightning, human_proximity, tree_density)
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]

    result = _INFER.query(
        variables=["Ignition"],
        evidence={
            "Temperature": temperature, "Rainfall": rainfall, "Lightning": lightning,
            "HumanProximity": human_proximity, "TreeDensity": tree_density,
        },
        show_progress=False
    )
    value = float(result.values[result.state_names["Ignition"].index("yes")])
    _PRIOR_CACHE[key] = value
    return value


if __name__ == "__main__":
    print("dry+hot+lightning+dense:", ignition_prior("high", "low", "yes", "yes", "high"))
    print("wet+cool+no lightning+sparse:", ignition_prior("low", "high", "no", "no", "low"))
    print("medium everything:", ignition_prior("medium", "medium", "no", "no", "medium"))