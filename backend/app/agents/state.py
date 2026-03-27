from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """
    The shared memory of the agent workflow.
    """
    question: str                   # The user's input
    generation: Optional[str]       # The final answer
    documents: List[str]            # Context retrieved from Vector DB or Web
    source: Optional[str]           # 'vector' or 'web' (to show citations)