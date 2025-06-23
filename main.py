# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_agent import run_query, generate_token_stream
from sse_starlette.sse import EventSourceResponse
from db import log_query

app = FastAPI(
    title="Autonomous Insight Engine",
    description="Mini LLM agent for grounded research queries",
    version="1.0.0"
)

# === Request/Response Schemas ===
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

config = {"configurable": {"thread_id": "abc123"}}

# === POST /query ===
@app.post("/query", response_model=QueryResponse)
def query_handler(request: QueryRequest):
    try:
        config = {"configurable": {"thread_id": "abc123"}}
        result = run_query(request.query, config)
        log_query(request.query, result["answer"], result["sources"])
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    return EventSourceResponse(generate_token_stream(request.query))