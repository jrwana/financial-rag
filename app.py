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
import asyncio
import uuid
from datetime import datetime

app = FastAPI(title="Financial RAG API")

app.state.rag = None
app.state.ingest_jobs = {}

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


@dataclass
class RAGState:
    chain: Any
    index: Any


class QueryRequest(BaseModel):
    question: str
    k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list


class IngestResponse(BaseModel):
    status: str
    chunks_processed: int


class IngestJobStatus(BaseModel):
    job_id: str
    status: str
    progress: str | None
    chunks_processed: int | None
    error: str | None


class IngestStarted(BaseModel):
    job_id: str
    status: str  # "started"


async def run_ingest_job(job_id: str):
    """Background task that runs ingestion and updates job status.

    Uses asyncio.to_thread() to run CPU-bound work in a thread pool,
    keeping the API responsive during ingestion.
    """
    jobs = app.state.ingest_jobs
    jobs[job_id]["status"] = "running"
    jobs[job_id]["progress"] = "Starting ingestion..."

    try:
        jobs[job_id]["progress"] = "Loading documents..."
        chunks = await asyncio.to_thread(ingest)

        jobs[job_id]["progress"] = "Creating embeddings..."
        new_index = await asyncio.to_thread(create_index, chunks)

        jobs[job_id]["progress"] = "Saving index..."
        await asyncio.to_thread(save_index, new_index)

        new_chain = await asyncio.to_thread(create_chain, new_index)

        # Atomic swap
        app.state.rag = RAGState(chain=new_chain, index=new_index)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["chunks_processed"] = len(chunks)
        jobs[job_id]["completed_at"] = datetime.utcnow()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.utcnow()


@app.get("/ingest/{job_id}", response_model=IngestJobStatus)
async def get_ingest_status(job_id: str, _: None = Depends(require_admin_key)):
    """Get status of an ingest job (prod only)."""
    if settings.ENV != "prod":
        raise HTTPException(status_code=404, detail="Job status only available in prod")

    job = app.state.ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return IngestJobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        chunks_processed=job["chunks_processed"],
        error=job["error"]
    )

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

    if settings.ENV == "prod":
        # Async: start background job
        job_id = str(uuid.uuid4())
        app.state.ingest_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": None,
            "chunks_processed": None,
            "error": None,
            "started_at": datetime.utcnow(),
            "completed_at": None
        }
        asyncio.create_task(run_ingest_job(job_id))
        return {"job_id": job_id, "status": "started"}

    else:
        # Local: synchronous
        try:
            chunks = ingest()
            new_index = create_index(chunks)
            save_index(new_index)
            new_chain = create_chain(new_index)
            app.state.rag = RAGState(chain=new_chain, index=new_index)
            return IngestResponse(status="success", chunks_processed=len(chunks))
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
