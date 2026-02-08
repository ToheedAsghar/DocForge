"""
AGENT STATE DEFINITIONS
Defines the shared memory all agents use.
"""

from operator import add
from typing import TypedDict, List, Optional, Annotated

class DocumentChunk(TypedDict):
    """
    Represents a chunk of a document.
    """

    id: str
    text: str
    score: float
    metadata: dict

class AgentStep(TypedDict):
    """
    Represents a single step in the agent's execution.
    """

    agent_name: str
    action: str
    reasoning: str
    timestamp: float

# MAIN STATE
class GraphState(TypedDict):
    # Input Section
    query: str

    # Routing Section
    search_query: str  # Optimized query for retrieval
    query_type: str
    confidence: float

    # Retrieval Section
    retrieved_chunks: List[DocumentChunk]
    retrieval_strategy: str
    
    # Analysis Section
    synthesized_answer: str
    information_gaps: List[str]

    # Validation Section
    validation_passed: bool
    validation_issues: List[str]
    fact_checked_answer: str 

    # Citations Section
    citations: List[str]
    
    # MetaData Section
    agent_steps: Annotated[List[AgentStep], add]
    retry_cnt: int
    total_tokens_used: int
    latency_ms: float