from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.config import settings
from backend.agents.graph import run_graph

app = FastAPI(title="RAG Agent API")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "rag-backend"}

@app.post(f"{settings.API_V1_PREFIX}/query")
def query_agent(request: QueryRequest):
    try:
        # run_graph returns the final state
        result = run_graph(request.query)

        # We might want to filter the result to return only relevant fields,
        # but returning the whole state is fine for now as it contains metadata.
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
