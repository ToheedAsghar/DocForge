import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.config import settings
from backend.agents.graph import run_graph, get_graph
from backend.agents.state import GraphState
from backend.services.llm_client import get_llm_client

app = FastAPI(title="RAG Agent API")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "rag-backend"}

@app.post(f"{settings.API_V1_PREFIX}/query")
async def query_agent(request: QueryRequest):
    try:
        # run_graph returns the final state
        result = await run_graph(request.query)

        # We might want to filter the result to return only relevant fields,
        # but returning the whole state is fine for now as it contains metadata.
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_PREFIX}/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    """
    async def event_generator():
        try:
            # Check cache first (replicating logic from run_graph)
            from backend.services.cache import get_cache_service
            cache = get_cache_service()
            cached_result = cache.get(request.query)

            if cached_result:
                yield f"data: {json.dumps({'type': 'cache_hit', 'data': cached_result})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # Setup graph execution
            graph = get_graph()
            initial_state = GraphState(
                query=request.query,
                retry_cnt=0,
                agent_steps=[],
                total_tokens_used=0,
                latency_ms=0.0
            )

            start_time = time.time()

            # Stream events
            async for event in graph.astream(initial_state):
                # Send the raw event data
                yield f"data: {json.dumps(event, default=str)}\n\n"

            # Calculate final stats (similar to run_graph)
            # Note: capturing the final state from stream is tricky, usually the last event contains it
            # For simplicity, we just signal completion here.
            # Ideally we would calculate tokens and latency here too and yield a 'final' event.

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
