from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from app.graph.nodes import AVAILABLE_TOOLS, assistant_node
from app.graph.state import AgentState


def should_continue(state: AgentState):
    """Check if the assistant called a tool or is done."""
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Construct the StateGraph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("assistant", assistant_node)
workflow.add_node("tools", ToolNode(AVAILABLE_TOOLS))

# Add Edges
workflow.set_entry_point("assistant")
workflow.add_conditional_edges("assistant", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "assistant")

# Compile graph
graph = workflow.compile()

