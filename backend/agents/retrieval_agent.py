"""
JOB: Search the vector database for relevant documents.
WHY: The quality of retrieval directly impacts the answer quality.
"""

import time
import asyncio
from typing import Dict
from backend.logger import get_logger
from backend.config import settings
from backend.services.vector_store import get_vector_store
from backend.agents.state import AgentStep, DocumentChunk, GraphState

logger = get_logger(__name__)

async def retrieve_documents(state: GraphState) -> Dict:
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

    logger.info("=" * 60)
    logger.info("Retrieval Agent")
    logger.info("=" * 60)

    top_k_map = {
        "simple_lookup": settings.TOP_K_SIMPLE,
        "complex_reasoning": settings.TOP_K_COMPLEX,
        "multi_hop": settings.TOP_K_MULTIHOP
    }

    top_k = top_k_map.get(query_type.lower(), settings.TOP_K_COMPLEX)


    # Adaptive Retrieval Strategy for Retries
    retrieval_strategy = "semantic" # Default: Vector Similarity Search
    
    # min_score_map = {
    #     "simple_lookup": 0.40,
    #     "complex_reasoning": 0.20,
    #     "multi_hop": 0.10
    # }
    
    # min_score = min_score_map.get(query_type,settings.RELEVANCE_THRESHOLD)

    min_score = settings.RELEVANCE_THRESHOLD;

    if retry_cnt > 0:
        logger.info(f"Retry# {retry_cnt}, adapting strategy...")

        # increase documents by 50%
        top_k = int(top_k * 1.5)

        # lower threshold to get more diverse results
        min_score = min_score * 0.85

        retrieval_strategy = "semantic_relaxed"

        logger.info(f"Retrieval Strategy: {retrieval_strategy}")
        logger.info(f"Min Score: {min_score}")
        logger.info(f"Top K: {top_k}")

    logger.info(f"Retrieving {top_k} documents for query: {search_query[:50]}...")
    if search_query != query:
        logger.info(f"(Optimized from original: {query[:50]}...)")

    vector_store = get_vector_store()
    try:
        loop = asyncio.get_running_loop()
        raw_results = await loop.run_in_executor(
            None,
            lambda: vector_store.search(
                query=search_query,
                top_k=top_k,
                min_score=min_score
            )
        )
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {str(e)}")
        raw_results = []

    logger.info(f"Retrieved {len(raw_results)} documents")

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
        logger.info(f"[{i+1}] Score: {result['score']:.3f}")
        logger.info(f"ID: {result['id']}")
        logger.info(f"TEXT: {result['text'][:100]}...")

    # Quality check (moved outside the loop)
    if len(retrieved_chunks) == 0:
        logger.warning("No documents retrieved. Query may be too broad or unclear.")
    elif len(retrieved_chunks) < 3:
        logger.warning(f"Low document count ({len(retrieved_chunks)}). Consider refining query or lowering min_score threshold.")

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
        