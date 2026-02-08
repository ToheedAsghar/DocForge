"""
LANGGRAPH ORCHESTRATION - THE CONTROL FLOW

JOB: Define how agents are connected and how data flows between them.
"""

import time
from typing import Literal
from backend.config import settings
from backend.services.llm_client import get_llm_client
from langgraph.graph import StateGraph, END
from backend.agents.routing_agent import router_query
from backend.agents.state import GraphState, AgentStep
from backend.agents.validation_agent import validate_answer
from backend.agents.retrieval_agent import retrieve_documents
from backend.agents.analysis_agent import analyze_and_synthesize

def should_validate(state: GraphState) -> Literal["VALIDATION_NODE", "SKIP_VALIDATION_NODE"]:
    """
    Decide whether to run validation or skip it based on confidence metrics.
    
    Skip validation when:
    1. Retrieved chunks have high average scores (> 0.85)
    2. We have enough supporting documents (>= 3)
    3. No information gaps were identified
    """

    chunks = state.get('retrieved_chunks', [])
    info_gaps = state.get('information_gaps', [])

    if not chunks:
        return "VALIDATION_NODE"

    avg_score = sum(c['score'] for c in chunks) / len(chunks)

    high_retrieval_score = avg_score > 0.85
    enough_sources = len(chunks) >= 3
    no_critical_gaps = len(info_gaps) == 0 or all('incomplete' not in gap.lower() and 'missing' not in gap.lower() for gap in info_gaps)

    if high_retrieval_score and enough_sources and no_critical_gaps:
        print(f"[INFO]\tHigh confidence (avg_score={avg_score:.2f}, sources={len(chunks)}) - SKIPPING validation")
        return 'SKIP_VALIDATION_NODE'
    else:
        print(f"[INFO]\tRunning validation (avg_score={avg_score:.2f}, sources={len(chunks)}, gaps={len(info_gaps)})")
        return "VALIDATION_NODE"


def validation_routing(state: GraphState) -> Literal["END", "RETRIEVER_NODE"]:
    """
    Decide whether to end the workflow or move back to the retriever
    """

    retry_cnt = state['retry_cnt']
    validation_passed = state['validation_passed']

    """
    DECISION LOGIC:
    1. 
    """

    if validation_passed:
        print(f"[INFO]\tValidation Passed. Ending Workflow.")
        return "END"
    
    if retry_cnt >= settings.MAX_RETRIES:
        print(f"[INFO]\tMAX retries reached. Returning to corrected answer.")
        return "END"
    else:
        print(f"[INFO]\tWill Retry Retrieval (attempt {retry_cnt+1})")
        return "RETRIEVER_NODE"

def skip_validation_node(state: GraphState) -> dict:
    step = AgentStep(
        agent_name="validation_agent",
        action="skipped (high confidence)",
        reasoning=f"Retrieval scores high, sufficient sources available",
        timestamp=time.time()
    )

    return {
        "validation_passed": True,
        "validation_issues": [],
        "fact_checked_answer": state.get("synthesized_answer", ""),
        "agent_steps": [step]
    }

def create_graph():
    graph = StateGraph(GraphState)

    # add nodes
    graph.add_node("ROUTER_NODE", router_query)
    graph.add_node("RETRIEVER_NODE", retrieve_documents)
    graph.add_node("ANALYSIS_NODE", analyze_and_synthesize)
    graph.add_node("VALIDATION_NODE", validate_answer)
    graph.add_node("SKIP_VALIDATION_NODE", skip_validation_node)

    # set entry point
    graph.set_entry_point("ROUTER_NODE")
    
    # add edges
    graph.add_edge("ROUTER_NODE", "RETRIEVER_NODE")
    graph.add_edge("RETRIEVER_NODE", "ANALYSIS_NODE")

    graph.add_conditional_edges('ANALYSIS_NODE', should_validate, 
    {
        "VALIDATION_NODE": "VALIDATION_NODE",
        "SKIP_VALIDATION_NODE": "SKIP_VALIDATION_NODE"
    })

    graph.add_edge("SKIP_VALIDATION_NODE", END)
    
    graph.add_conditional_edges("VALIDATION_NODE", validation_routing, {
        "END": END,
        "RETRIEVER_NODE": "RETRIEVER_NODE"
    })

    compiled_graph = graph.compile()
    return compiled_graph

def run_graph(query: str, user_id: str = None, use_cache: bool = True) -> GraphState:
    """
    EXECUTE THE COMPLETE WORKFLOW FOR THE QUERY WITH CACHE
    """

    if use_cache:
        from backend.services.cache import get_cache_service

        cache = get_cache_service()
        result = cache.get(query)

        if result:
            print(f"[INFO]\tReturning cached result for: {query[:50]}...")
            print(f"[INFO]\tSkipping workflow execution.")

            result['_from_cache'] = True
            result['query'] = query
            return result

    print(f"[INFO]\tNo cached result found. Running workflow...")

    INITIAL_STATE = GraphState(
        query=query,
        retry_cnt=0,
        agent_steps=[],
        total_tokens_used=0,
        latency_ms=0.0
    )

    print(f"[INFO]\tStarting RAG Workflow for Query: {query[:50]}...")

    start_time = time.time()
    graph = create_graph()
    final_state = graph.invoke(INITIAL_STATE)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    llm = get_llm_client()

    final_state['latency_ms'] = latency_ms
    final_state['total_tokens_used'] = llm.get_token_usage()

    if use_cache:
        cache = get_cache_service()
        print(f"[INFO]\tCaching result...")

        if final_state.get('validation_passed', False) or final_state.get('retry_cnt', 0) >= settings.MAX_RETRIES - 1:
            print(f"[INFO]\tValidation passed. Caching result...")
            cache.set(query, dict(final_state))
        else:
            print(f"[INFO]\tValidation failed. Not caching result.")

    print(f"[INFO]\tWorkflow Completed in {latency_ms:.2f}ms")
    print(f"[INFO]\tTotal Tokens Used: {final_state['total_tokens_used']}")
    print(f"[INFO]\tRetry Count: {final_state['retry_cnt']}")
    print(f"[INFO]\tDocuments Retrieved: {len(final_state['retrieved_chunks'])}")
    print(f"[INFO]\tValidation Passed: {final_state['validation_passed']}")
    print(f"[INFO]\tAgent Steps: {len(final_state['agent_steps'])}")
    print(f"[INFO]\tFinal Answer: {final_state['synthesized_answer'][:100]}...")

    return final_state
