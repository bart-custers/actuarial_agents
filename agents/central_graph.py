from typing import TypedDict, List, Dict, Any
from utils.message_types import Message 
from langgraph.graph import StateGraph, END


class WorkflowState(TypedDict):
    dataset_path: str
    iteration: int
    phase: str
    metrics: Dict[str, Any]
    review_notes: List[str]
    status: str
    action: str
    memory_log: Dict[str, Any]
    hub: Any

def merge_state(old_state: WorkflowState, new_metadata: Dict[str, Any]) -> WorkflowState:
    """Safely merge new metadata without losing persistent keys."""
    new_state = {**old_state, **(new_metadata or {})}
    new_state["hub"] = old_state["hub"]          # ensure hub is preserved
    new_state["iteration"] = old_state.get("iteration", 0)
    new_state["memory_log"] = old_state.get("memory_log", {})
    return new_state

def safe_update_state(state, new_data):
    """Safely merge new_data into the workflow state."""
    if not new_data:
        return state
    for k, v in new_data.items():
        if isinstance(v, dict) and isinstance(state.get(k), dict):
            state[k].update(v)
        else:
            state[k] = v
    return state

def dataprep_node(state: WorkflowState) -> WorkflowState:
    hub = state["hub"]
    msg = Message(
        sender="hub",
        recipient="dataprep",
        type="task",
        content="Clean dataset and summarize.",
        metadata=state,
    )

    response = hub.send(msg)
    state = safe_update_state(state, response.metadata)

    state["phase"] = "modelling"
    print("[dataprep_node] completed → next phase: modelling")
    return state


# ------------------------------
#  Modelling Node
# ------------------------------
def modelling_node(state: WorkflowState) -> WorkflowState:
    hub = state["hub"]
    msg = Message(
        sender="hub",
        recipient="modelling",
        type="task",
        content=f"Train predictive model (iteration {state.get('iteration', 0)}).",
        metadata=state,
    )

    response = hub.send(msg)
    state = safe_update_state(state, response.metadata)

    # Ensure metrics are recorded in memory for downstream review
    if "metrics" not in state and hasattr(response, "metadata"):
        state["metrics"] = response.metadata.get("metrics", {})

    state["phase"] = "reviewing"
    print("[modelling_node] completed → next phase: reviewing")
    return state


# ------------------------------
#  Reviewing Node
# ------------------------------
def reviewing_node(state: WorkflowState) -> WorkflowState:
    hub = state["hub"]
    msg = Message(
        sender="hub",
        recipient="reviewing",
        type="task",
        content="Review model outputs and consistency.",
        metadata=state,
    )

    response = hub.send(msg)
    state = safe_update_state(state, response.metadata)

    action = state.get("action", "proceed_to_explanation")
    if action == "retrain_model":
        next_phase = "modelling"
    elif action == "reclean_data":
        next_phase = "dataprep"
    elif action == "abort_workflow":
        next_phase = "end"
    else:
        next_phase = "explanation"

    state["phase"] = next_phase
    print(f"[reviewing_node] completed → next phase: {next_phase}")
    return state


# ------------------------------
#  Explanation Node
# ------------------------------
def explanation_node(state: WorkflowState) -> WorkflowState:
    hub = state["hub"]
    msg = Message(
        sender="hub",
        recipient="explanation",
        type="task",
        content="Generate explanations and belief-revision report.",
        metadata=state,
    )

    response = hub.send(msg)
    state = safe_update_state(state, response.metadata)
    state["phase"] = "end"

    print("[explanation_node] completed → workflow end.")
    return state

def build_actuarial_graph(hub):
    graph = StateGraph(WorkflowState)

    def dataprep_node(state):
        msg = {"sender": "hub", "recipient": "dataprep", "type": "task", "content": "Clean dataset and summarize.", "metadata": state}
        response = hub.send(msg)
        state.update(response.metadata or {})
        state["phase"] = "modelling"
        return state

    def modelling_node(state):
        msg = {"sender": "hub", "recipient": "modelling", "type": "task", "content": f"Train predictive model (iteration {state['iteration']}).", "metadata": state}
        response = hub.send(msg)
        state.update(response.metadata or {})
        state["phase"] = "reviewing"
        return state

    def reviewing_node(state):
        msg = {"sender": "hub", "recipient": "reviewing", "type": "task", "content": "Review model outputs and consistency.", "metadata": state}
        response = hub.send(msg)
        state.update(response.metadata or {})
        return state

    def explanation_node(state):
        msg = {"sender": "hub", "recipient": "explanation", "type": "task", "content": "Generate explanations and belief-revision report.", "metadata": state}
        response = hub.send(msg)
        state.update(response.metadata or {})
        state["phase"] = "end"
        return state

    graph.add_node("dataprep", dataprep_node)
    graph.add_node("modelling", modelling_node)
    graph.add_node("reviewing", reviewing_node)
    graph.add_node("explanation", explanation_node)

    # transitions
    graph.add_edge("dataprep", "modelling")
    graph.add_edge("modelling", "reviewing")

    def review_decision(state):
        return {
            "retrain_model": "modelling",
            "reclean_data": "dataprep",
            "abort_workflow": END,
            "proceed_to_explanation": "explanation",
        }.get(state.get("action", "proceed_to_explanation"), "explanation")

    graph.add_conditional_edges("reviewing", review_decision)
    graph.add_edge("explanation", END)
    graph.set_entry_point("dataprep")
    return graph

# Wrap agents as Langgraph nodes
# def dataprep_node(state: WorkflowState) -> WorkflowState:
#     hub = state["hub"]
#     msg = Message(
#     sender="hub",
#     recipient="dataprep",
#     type="task",
#     content="Clean dataset and summarize.",
#     metadata=state
#     )
#     response = hub.send(msg)
#     #state = merge_state(state, response.metadata)
#     state = safe_update_state(state, response.metadata)
#     state["phase"] = "modelling"
#     print("[dataprep_node] completed → next phase: modelling")
#     return state

# def modelling_node(state: WorkflowState) -> WorkflowState:
#     hub = state["hub"]
#     msg = Message(
#     sender="hub",
#     recipient="modelling",
#     type="task",
#     content=f"Train predictive model (iteration {state['iteration']}).",
#     metadata=state
#     )
#     response = hub.send(msg)
#     #state = merge_state(state, response.metadata)
#     state = safe_update_state(state, response.metadata)
#     state["phase"] = "reviewing"
#     print("[modelling_node] completed → next phase: reviewing")
#     return state

# def reviewing_node(state: WorkflowState) -> WorkflowState:
#     hub = state["hub"]
#     msg = Message(
#     sender="hub",
#     recipient="reviewing",
#     type="task",
#     content="Review model outputs and consistency.",
#     metadata=state
#     )
#     response = hub.send(msg)
#     #state = merge_state(state, response.metadata)
#     state = safe_update_state(state, response.metadata)

#     action = state.get("action", "proceed_to_explanation")
#     if action == "retrain_model":
#         state["phase"] = "modelling"
#     elif action == "reclean_data":
#         state["phase"] = "dataprep"
#     elif action == "abort_workflow":
#         state["phase"] = "end"
#     else:
#         state["phase"] = "explanation"

#     print(f"[reviewing_node] completed → next phase: {state['phase']}")
#     return state

# def explanation_node(state: WorkflowState) -> WorkflowState:
#     hub = state["hub"]
#     msg = Message(
#     sender="hub",
#     recipient="explanation",
#     type="task",
#     content="Generate explanations and belief-revision report.",
#     metadata=state
#     )
#     response = hub.send(msg)
#     #state = merge_state(state, response.metadata)
#     state = safe_update_state(state, response.metadata)
#     state["phase"] = "end"
#     print("[explanation_node] completed → workflow end.")
#     return state

# # Define the graph structure
# def build_actuarial_graph():
#     graph = StateGraph(WorkflowState)

#     graph.add_node("dataprep", dataprep_node)
#     graph.add_node("modelling", modelling_node)
#     graph.add_node("reviewing", reviewing_node)
#     graph.add_node("explanation", explanation_node)

#     # Define linear transitions
#     graph.add_edge("dataprep", "modelling")
#     graph.add_edge("modelling", "reviewing")

#     # Conditional transitions from reviewing
#     def review_decision(state: WorkflowState):
#         action = state.get("action", "proceed_to_explanation")
#         return {
#             "retrain_model": "modelling",
#             "reclean_data": "dataprep",
#             "abort_workflow": END,
#             "proceed_to_explanation": "explanation",
#         }.get(action, "explanation")

#     graph.add_conditional_edges("reviewing", review_decision)

#     # Final step
#     graph.add_edge("explanation", END)
#     graph.set_entry_point("dataprep")

#     return graph
