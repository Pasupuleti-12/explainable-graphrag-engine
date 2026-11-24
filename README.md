# Deterministic Neuro-Symbolic Healthcare GraphRAG

Local research prototype for the thesis:

**Reducing Unsupported Biomedical Claims Using Deterministic Neuro-Symbolic Graph Traversal**

The prototype retrieves documents, extracts biomedical triples, builds a NetworkX graph, and supplies evidence paths to a local Ollama model. Traversal verifies that paths have source snippets and references; this does not prove that extracted relations or generated claims are scientifically correct. The current metrics are descriptive pipeline values and optional human-supplied labels, not benchmark results.

## Architecture

```text
User Query
  -> Streamlit
  -> FastAPI
  -> PubMed/PDF ingestion
  -> document chunking
  -> all-MiniLM-L6-v2 embeddings
  -> ChromaDB retrieval
  -> Extractor Agent
  -> Knowledge Graph Builder Agent
  -> NetworkX graph
  -> Symbolic Reasoner Agent
  -> validated BFS/DFS/shortest paths
  -> Generator Agent with graph-only context
  -> Evaluation dashboard
```

See [docs/architecture.md](docs/architecture.md) for the research workflow and validation strategy.

## Setup

1. Install Python 3.12.
2. Install Ollama and pull a local model:

```powershell
ollama pull llama3
ollama serve
```

3. Create and activate a virtual environment:

```powershell
C:\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

4. Optional spaCy model:

```powershell
python -m spacy download en_core_web_sm
```

## Run

In the first PowerShell window, start the API and leave it running:

```powershell
uvicorn backend.main:app --reload --port 8000
```

In a second PowerShell window, start the dashboard:

```powershell
streamlit run frontend/app.py
```

## Sample Healthcare Queries

- How does diabetes affect blood glucose regulation?
- What is the relationship between insulin resistance and type 2 diabetes?
- Which molecular pathways connect inflammation and cardiovascular disease?
- What evidence links hypertension to kidney disease?

## API

- `GET /health`
- `POST /ingest/pubmed`
- `POST /ingest/pdf`
- `POST /graph/build`
- `POST /reason`
- `GET /graph`
- `GET /logs`
- `POST /evaluate`
- `POST /experiment` (LLM-only, conventional RAG, unvalidated GraphRAG, and deterministic-path GraphRAG)

## Research Notes

The dashboard provides source-linked evidence cards, extracted relationship and polarity inspection, a same-question configuration comparison, and Markdown/CSV exports. Experiment latency and human reviews are exploratory outputs; the application does not calculate biomedical correctness or claim a reduction in unsupported claims.

The generator receives only:

- validated graph paths
- verified entities
- source evidence
- deterministic subgraphs

If no eligible positive evidence path exists, the system abstains. This is a limited evidence gate, not a guarantee against unsupported claims or medical advice.

## Tests

```powershell
pytest
```

The current unit tests cover selected graph traversal behavior. They are not an end-to-end or biomedical accuracy evaluation.
