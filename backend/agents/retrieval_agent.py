"""
JOB: Search the vector database for relevant documents.
WHY: The quality of retrieval directly impacts the answer quality.
"""

import time
from typing import Dict
from backend.config import settings
from backend.services.vector_store import get_vector_store
from backend.agents.state import AgentStep, DocumentChunk, GraphState

def retrieve_documents(state: GraphState) -> Dict:
    """
    Get Relevant Documents from PINECONE Vector Database
    
    PROCESS:
    1. Determine how many documents to retreive based on the query type
    2. Optionally Adjust Strategy if this is a retry
    3. Search Vector Database
    4. Filter by relevance score
    5. Return top results
    """

    query = state["query"]
    search_query = state.get("search_query", query) # Use optimized query if available
    query_type = state["query_type"]
    retry_cnt = state["retry_cnt"]

    print(f"\n{'='*60}")
    print(f"Retrieval Agent")
    print(f"{'='*60}")

    top_k_map = {
        "simple_lookup": settings.TOP_K_SIMPLE,
        "complex_reasoning": settings.TOP_K_COMPLEX,
        "multi_hop": settings.TOP_K_MULTIHOP
    }

    top_k = top_k_map.get(query_type.lower(), settings.TOP_K_COMPLEX)


    # Adaptive Retrieval Strategy for Retries
    retrieval_strategy = "semantic" # Default: Vector Similarity Search
    min_score = settings.RELEVANCE_THRESHOLD

    if retry_cnt > 0:
        print(f"[INFO]\tRetry# {retry_cnt}, adapting strategy...")

        # increase documents by 50%
        top_k = int(top_k * 1.5)

        # lower threshold to get more diverse results
        min_score = min_score * 0.85

        retrieval_strategy = "semantic_relaxed"

        print(f"[INFO]\tRetrieval Strategy: {retrieval_strategy}")
        print(f"[INFO]\tMin Score: {min_score}")
        print(f"[INFO]\tTop K: {top_k}")

    print(f"[INFO]\tRetrieving {top_k} documents for query: {search_query[:50]}...")
    if search_query != query:
        print(f"[INFO]\t(Optimized from original: {query[:50]}...)")

    vector_store = get_vector_store()
    try:
        raw_results = vector_store.search(
            query=search_query,
            top_k=top_k,
            min_score=min_score
        )
    except Exception as e:
        print(f"[ERROR]\tFailed to retrieve documents: {str(e)}")
        raw_results = []

    print(f"[INFO]\tRetrieved {len(raw_results)} documents")

    retrieved_chunks = []
    for i, result in enumerate(raw_results):
        chunk = DocumentChunk(
            id=result["id"],
            text=result["text"],
            score=result["score"],
            metadata=result.get("metadata", {})
        )
        retrieved_chunks.append(chunk)

        # log each retrieved chunk
        print(f"[INFO]\t[{i+1}] Score: {result['score']:.3f}")
        print(f"[INFO]\tID: {result['id']}")
        print(f"[INFO]\tTEXT: {result['text'][:100]}...")

    # Quality check (moved outside the loop)
    if len(retrieved_chunks) == 0:
        print(f"[WARNING]\tNo documents retrieved. Query may be too broad or unclear.")
    elif len(retrieved_chunks) < 3:
        print(f"[WARNING]\tLow document count ({len(retrieved_chunks)}). Consider refining query or lowering min_score threshold.")

    # log agent step
    step = AgentStep(
        agent_name="retrieval_agent",
        action=f"retrieved {len(retrieved_chunks)} documents",
        reasoning=f"top_k={top_k} min_score={min_score:.2f} strategy={retrieval_strategy}",
        timestamp=time.time()
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_strategy": retrieval_strategy,
        "agent_steps": [step]
    }
        