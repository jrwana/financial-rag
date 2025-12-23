# Financial RAG

A Retrieval-Augmented Generation (RAG) API for querying financial documents using LangChain and FAISS.

## Prerequisites

1. **Environment variables**: Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. **Documents**: Place your PDF files in the `data/` folder before running `/ingest`.

## Local Development (Conda)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate financial-rag

# Run tests
pytest -q

# Start API server
uvicorn app:app --reload
```

## Production (Docker)

```bash
# Build
docker build -t financial-rag .

# Run
docker run -p 8000:8000 --env-file .env financial-rag
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ingest` | POST | Ingest documents and build vector index |
| `/query` | POST | Query the RAG system |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue?", "k": 4}'
```

Response:
```json
{
  "answer": "The revenue was...",
  "sources": [...]
}
```

## Environment Notes

| Environment | Setup | Notes |
|-------------|-------|-------|
| **Local** | Conda | GPU support via pytorch/nvidia channels |
| **Production** | Docker | CPU deps only (`pip install .[cpu]`) |

## Project Structure

```
financial-rag/
├── app.py              # FastAPI application
├── src/
│   ├── ingestion.py    # Document loading and chunking
│   ├── embeddings.py   # Vector embeddings and FAISS index
│   └── retrieval.py    # RAG chain and query execution
├── tests/              # Unit tests
├── environment.yml     # Conda environment
├── pyproject.toml      # Package configuration
└── Dockerfile          # Production container
```

## CORS Configuration

  In production, set `CORS_ORIGINS` to a comma-separated list of allowed origins:

  ```bash
  CORS_ORIGINS=https://myapp.com,https://admin.myapp.com

  In local mode, CORS is permissive (all origins allowed).