from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from agents.orchestrator import NeuroSymbolicGraphRAG
from api.schemas import (
    EvaluationRequest,
    EvaluationResponse,
    ExperimentCase,
    ExperimentRequest,
    ExperimentResponse,
    ExperimentRun,
    PubMedIngestRequest,
    ReasonRequest,
)
from utils.logging import configure_logging, recent_logs

configure_logging()

app = FastAPI(title="Deterministic Neuro-Symbolic Healthcare GraphRAG", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
system = NeuroSymbolicGraphRAG()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "graph_nodes": system.graph_builder.graph.number_of_nodes()}


@app.post("/ingest/pubmed")
async def ingest_pubmed(request: PubMedIngestRequest):
    try:
        return await run_in_threadpool(system.ingest_pubmed, request.query, request.max_results)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"PubMed ingestion failed: {exc}") from exc


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    if Path(file.filename or "upload.pdf").suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted")
    contents = await file.read(15 * 1024 * 1024 + 1)
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds the 15 MB upload limit")
    tmp_path: str | None = None
    try:
        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        chunks = await run_in_threadpool(system.processor.chunks_from_pdf, tmp_path, f"pdf:{file.filename}")
        return await run_in_threadpool(system.ingest_chunks, chunks)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PDF ingestion failed: {exc}") from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/graph/build")
async def build_graph(query: str | None = None, use_llm: bool = False, model: str = "llama3"):
    return await run_in_threadpool(system.build_graph, query, use_llm, model)


@app.get("/graph")
def graph():
    return system.graph_builder.to_dict()


@app.post("/graph/clear")
def clear_graph():
    system.graph_builder.clear()
    system.pending_chunks.clear()
    system.vector_store.clear()
    return {"message": "Graph, pending chunks, and vector store cleared"}


@app.post("/reason")
async def reason(request: ReasonRequest):
    return await run_in_threadpool(system.reason, request)


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    response = await run_in_threadpool(
        system.reason,
        ReasonRequest(
            query=request.query,
            model=request.model,
            traversal=request.traversal,
            max_hops=request.max_hops,
            auto_pubmed=request.auto_pubmed,
        )
    )
    return EvaluationResponse(
        metrics=response.metrics,
        notes=[
            "Structural counts and token overlap are descriptive proxies, not correctness or faithfulness scores.",
            "This endpoint does not compare against a baseline or accept human correctness labels.",
        ],
        answer=response.answer,
        grounded=response.grounded,
        sources=response.sources,
    )


@app.post("/experiment", response_model=ExperimentResponse)
async def experiment(request: ExperimentRequest):
    def run_experiment() -> ExperimentResponse:
        cases = []
        for query in request.questions:
            retrieval_started = time.perf_counter()
            chunks = system.vector_store.query(query, n_results=10)
            retrieval_elapsed = time.perf_counter() - retrieval_started
            runs = []

            started = time.perf_counter()
            llm_answer = system.generator.generate_llm_only(query, request.model)
            llm_elapsed = time.perf_counter() - started
            runs.append(
                ExperimentRun(
                    configuration="llm_only",
                    answer=llm_answer,
                    latency_seconds=round(llm_elapsed, 3),
                    model_response_received=not llm_answer.startswith("LLM-only baseline unavailable"),
                )
            )

            started = time.perf_counter()
            triples = system.extractor.extract(chunks, use_llm=False, model=request.model)
            graph_rag_answer = system.generator.generate_from_relationships(query, triples, request.model)
            graph_rag_elapsed = time.perf_counter() - started
            runs.append(
                ExperimentRun(
                    configuration="graph_rag_unvalidated",
                    answer=graph_rag_answer,
                    sources=sorted({f"{triple.source}#{triple.chunk_id}" for triple in triples}),
                    latency_seconds=round(graph_rag_elapsed, 3),
                    model_response_received=bool(triples)
                    and not graph_rag_answer.startswith("Extracted graph evidence (not synthesized"),
                )
            )

            started = time.perf_counter()
            rag_answer = system.generator.generate_from_chunks(query, chunks, request.model)
            rag_elapsed = time.perf_counter() - started
            runs.append(
                ExperimentRun(
                    configuration="conventional_rag",
                    answer=rag_answer,
                    sources=sorted({chunk.source for chunk in chunks}),
                    latency_seconds=round(rag_elapsed, 3),
                    model_response_received=bool(chunks) and not rag_answer.startswith(
                        "Retrieved passages (not synthesized"
                    ),
                )
            )

            started = time.perf_counter()
            graph_response = system.reason_with_chunks(
                ReasonRequest(query=query, model=request.model, auto_pubmed=False),
                chunks,
                isolated_graph=True,
            )
            graph_elapsed = time.perf_counter() - started
            runs.append(
                ExperimentRun(
                    configuration="neurosymbolic_graphrag",
                    answer=graph_response.answer,
                    sources=graph_response.sources,
                    latency_seconds=round(graph_elapsed, 3),
                    model_response_received=bool(graph_response.grounded)
                    and not graph_response.answer.startswith("Grounded answer for:"),
                )
            )
            cases.append(
                ExperimentCase(
                    question=query,
                    runs=runs,
                    retrieval_latency_seconds=round(retrieval_elapsed, 3),
                    retrieved_sources=sorted({chunk.source for chunk in chunks}),
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
                )
            )
        return ExperimentResponse(model=request.model, cases=cases)

    return await run_in_threadpool(run_experiment)


@app.get("/logs")
def logs(limit: int = 100):
    return {"logs": recent_logs(limit)}
