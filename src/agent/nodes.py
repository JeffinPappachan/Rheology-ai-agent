import yaml
from src.surrogate.inference import predict
from src.control.stability import is_stable

config = yaml.safe_load(open("config/config.yaml"))

def predict_node(state):
    result = predict(state["temperature"], state["flow_rate"])
    state["predicted_viscosity"] = result["viscosity"]
    return state


def check_node(state):
    stable = is_stable(
        state["predicted_viscosity"],
        config["target_viscosity"],
        config["tolerance"]
    )

    state["deviation"] = not stable
    return state


def decision_node(state):
    """
    Correct proportional control for shear-thinning fluid.
    """

    error = state["predicted_viscosity"] - config["target_viscosity"]

    Kp = 0.3  # tuning gain

    # Since d(viscosity)/d(flow) < 0
    # Control law must compensate sign inversion

    adjustment = Kp * error

    state["control_action"] = adjustment
    return state


def apply_node(state):
    state["flow_rate"] += state["control_action"]

    # Optional safety bounds (recommended)
    state["flow_rate"] = max(0.5, min(3.0, state["flow_rate"]))

    state["iteration"] += 1
    return state