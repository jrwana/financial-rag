"""
Routes
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.ingestion import ingest
from src.embeddings import create_index, save_index, load_index
from src.retrieval import create_chain, query
from src.deps import require_api_key, require_admin_key, check_rate_limit
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from dataclasses import dataclass
from typing import Any

@dataclass
class RAGState:
    chain: Any
    index: Any

app = FastAPI(title="Financial RAG API")

app.state.rag = None

if settings.ENV == "prod":
      # Prod: strict allowlist from env
      origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
      if not origins:
          origins = []  # No CORS if not configured (safest default)
else:
    # Local: permissive for development
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Or ["*"] if needed
    allow_headers=["X-API-Key", "X-Admin-Key", "Content-Type"],
)


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
    try:
        index = load_index()
        chain = create_chain(index)
        app.state.rag = RAGState(chain=chain, index=index)
        print("Index loaded on startup")
    except Exception as e:
        print(f"No existing index found: {e}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(_: None = Depends(require_admin_key)):
    """Rebuild the vector database from documents"""
    try:
        # Build everything in local variables first
        chunks = ingest()
        new_index = create_index(chunks)
        save_index(new_index)
        new_chain = create_chain(new_index)

        # atomic swap - single assignment replaces entire state
        app.state.rag = RAGState(chain=new_chain, index=new_index)

        return IngestResponse(
            status="success",
            chunks_processed=len(chunks)
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    _auth: None = Depends(require_api_key),
    _rate: None = Depends(check_rate_limit),
    ):
    """Query the RAG system and return answer and sources"""
    rag = app.state.rag  # snapshot the reference

    if rag is None:
        raise HTTPException(
            status_code=400,
            detail="No index loaded. Call /ingest first"
        )

    try:
        result = query(rag.chain, request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "index_loaded": app.state.rag is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
