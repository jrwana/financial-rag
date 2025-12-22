"""
Routes
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.ingestion import ingest
from src.embeddings import create_index, save_index, load_index
from src.retrieval import create_chain, query
from src.api import require_api_key, require_admin_key

app = FastAPI(title="Financial RAG API")

# global state
chain = None


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list


class IngestResponse(BaseModel):
    status: str
    chunks_processed: int


@app.on_event("startup")
async def startup():
    """Load existing index on startup if available"""
    global chain
    try:
        index = load_index()
        chain = create_chain(index)
        print("Index loaded on startup")
    except Exception as e:
        print(f"No existing index found: {e}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(_: None = Depends(require_admin_key)):
    """Rebuild the vector database from documents"""
    global chain

    try:
        chunks = ingest()
        index = create_index(chunks)
        save_index(index)
        chain = create_chain(index)

        return IngestResponse(
            status="success",
            chunks_processed=len(chunks)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, _: None = Depends(require_api_key)):
    """Query the RAG system and return answer and sources"""
    global chain

    if chain is None:
        raise HTTPException(
            status_code=400,
            detail="No index loaded. Call /ingest first"
        )

    try:
        result = query(chain, request.question)

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "index_loaded": chain is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
