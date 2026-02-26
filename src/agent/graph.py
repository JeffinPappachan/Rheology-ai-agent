from langgraph.graph import StateGraph
from src.agent.state import PlantState
from src.agent.nodes import (
    predict_node,
    check_node,
    decision_node,
    apply_node
)
import yaml

config = yaml.safe_load(open("config/config.yaml"))

graph = StateGraph(PlantState)

graph.add_node("predict", predict_node)
graph.add_node("check", check_node)
graph.add_node("decide", decision_node)
graph.add_node("apply", apply_node)

graph.set_entry_point("predict")

graph.add_edge("predict", "check")

def route(state):
    if not state["deviation"]:
        return "__end__"

    if state["iteration"] >= config["max_iterations"]:
        return "__end__"

    return "decide"

graph.add_conditional_edges(
    "check",
    route,
    {
        "decide": "decide",
        "__end__": "__end__"
    }
)

graph.add_edge("decide", "apply")
graph.add_edge("apply", "predict")

app = graph.compile()