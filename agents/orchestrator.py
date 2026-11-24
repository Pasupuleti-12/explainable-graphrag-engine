from __future__ import annotations

import logging

from agents.extractor import ExtractorAgent
from agents.generator import GeneratorAgent
from api.schemas import DocumentChunk, EvidenceRecord, IngestResponse, ReasonRequest, ReasonResponse
from evaluation.metrics import EvaluationAgent
from graph.knowledge_graph import KnowledgeGraphBuilderAgent
from mcp.pubmed_client import PubMedClient
from reasoning.symbolic_reasoner import SymbolicReasonerAgent
from utils.document_processing import DocumentProcessor
from utils.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class NeuroSymbolicGraphRAG:
    def __init__(self) -> None:
        self.processor = DocumentProcessor()
        self.vector_store = ChromaVectorStore()
        self.extractor = ExtractorAgent()
        self.graph_builder = KnowledgeGraphBuilderAgent()
        self.generator = GeneratorAgent()
        self.evaluator = EvaluationAgent()
        self.pubmed = PubMedClient(self.processor)
        self.pending_chunks: list[DocumentChunk] = []

    def ingest_chunks(self, chunks: list[DocumentChunk]) -> IngestResponse:
        self.pending_chunks.extend(chunks)
        self.vector_store.add_chunks(chunks)
        sources = sorted({chunk.source for chunk in chunks})
        return IngestResponse(chunks=len(chunks), sources=sources, message="Chunks ingested and embedded")

    def ingest_pubmed(self, query: str, max_results: int) -> IngestResponse:
        chunks = self.pubmed.ingest(query, max_results=max_results)
        return self.ingest_chunks(chunks)

    def build_graph(self, query: str | None = None, use_llm: bool = False, model: str = "llama3") -> dict:
        chunks = self.pending_chunks
        if query:
            chunks = self.vector_store.query(query, n_results=12)
        triples = self.extractor.extract(chunks, use_llm=use_llm, model=model)
        self.graph_builder.add_triples(triples)
        self.graph_builder.save_json()
        return {"triples": len(triples), **self.graph_builder.to_dict()}

    def reason(self, request: ReasonRequest) -> ReasonResponse:
        retrieved = self._retrieve_or_ingest_for_query(request)
        return self.reason_with_chunks(request, retrieved)

    def reason_with_chunks(
        self, request: ReasonRequest, retrieved: list[DocumentChunk], *, isolated_graph: bool = False
    ) -> ReasonResponse:
        triples = []
        if retrieved:
            triples = self.extractor.extract(retrieved, use_llm=False, model=request.model)
        graph_builder = KnowledgeGraphBuilderAgent() if isolated_graph else self.graph_builder
        if triples:
            graph_builder.add_triples(triples)
        candidate_entities: list[str] = []
        for chunk in retrieved:
            candidate_entities.extend(self.extractor.entities(chunk.text))
        reasoner = SymbolicReasonerAgent(graph_builder.graph)
        paths = reasoner.reason(request.query, candidate_entities, traversal=request.traversal, max_hops=request.max_hops)
        answer = self.generator.generate(request.query, paths, model=request.model)
        sources = sorted({step.reference for path in paths for step in path.steps})
        connected_entities = sorted({entity for path in paths for entity in path.entities})
        response = ReasonResponse(
            answer=answer,
            grounded=bool(paths),
            reasoning_paths=paths,
            connected_entities=connected_entities,
            sources=sources,
            metrics={},
            evidence=[
                EvidenceRecord(
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    title=str(chunk.metadata.get("title", "")),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
                for chunk in retrieved[:20]
            ],
            extracted_relationships=triples,
            limitations=[
                "Extracted relationships are heuristic and may be wrong.",
                "A source-backed graph path is not proof that a biomedical conclusion is valid.",
                "This research prototype is not for diagnosis or treatment decisions.",
            ],
        )
        response.metrics = self.evaluator.evaluate(request.query, response)
        return response

    def _retrieve_or_ingest_for_query(self, request: ReasonRequest) -> list[DocumentChunk]:
        retrieved = self.vector_store.query(request.query, n_results=10)
        if not request.auto_pubmed:
            return retrieved

        expansion_queries = self._expanded_pubmed_queries(request.query)
        for query in expansion_queries:
            try:
                chunks = self.pubmed.ingest(query, max_results=request.pubmed_max_results)
            except Exception as exc:
                logger.warning("Auto PubMed ingestion failed for query=%s: %s", query, exc)
                continue
            self.ingest_chunks(chunks)
        return self.vector_store.query(request.query, n_results=16)

    @staticmethod
    def _expanded_pubmed_queries(query: str) -> list[str]:
        query_lower = query.lower()
        queries = [query]
        if any(term in query_lower for term in ["treat", "drug", "cure", "therapy", "medicine"]):
            queries.append("type 2 diabetes treatment metformin insulin therapy glycemic control")
            queries.append("diabetes management drugs metformin insulin GLP-1 SGLT2")
        if any(term in query_lower for term in ["glucose", "insulin", "regulation"]):
            queries.append("diabetes insulin secretion blood glucose regulation hepatic glucose production")
        return list(dict.fromkeys(queries))
