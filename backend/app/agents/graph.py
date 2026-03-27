from langgraph.graph import StateGraph, END
from backend.app.agents.state import AgentState
from backend.app.agents.nodes import retrieve_node, generate_node

# 1. Create Graph
workflow = StateGraph(AgentState)

# 2. Add Nodes (The Agents)
workflow.add_node("researcher", retrieve_node)
workflow.add_node("generator", generate_node)

# 3. Define Edges (The Flow)
# Start -> Researcher -> Generator -> End
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "generator")
workflow.add_edge("generator", END)

# 4. Compile
app = workflow.compile()