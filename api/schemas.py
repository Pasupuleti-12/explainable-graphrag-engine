from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    source: str
    chunk_id: str
    evidence: str
    confidence: float = 1.0
    polarity: Literal["positive", "negative", "uncertain"] = "positive"
    qualifiers: list[str] = Field(default_factory=list)


class PubMedIngestRequest(BaseModel):
    query: str
    max_results: int = 5


class IngestResponse(BaseModel):
    chunks: int
    sources: list[str]
    message: str


class ReasonRequest(BaseModel):
    query: str
    traversal: Literal["bfs", "dfs", "shortest"] = "bfs"
    max_hops: int = 3
    model: str = "llama3"
    auto_pubmed: bool = True
    pubmed_max_results: int = 6


class PathStep(BaseModel):
    source: str
    relation: str
    target: str
    evidence: str
    reference: str
    polarity: Literal["positive", "negative", "uncertain"] = "positive"


class ReasoningPath(BaseModel):
    path_id: str
    entities: list[str]
    steps: list[PathStep]
    score: float


class EvidenceRecord(BaseModel):
    source: str
    chunk_id: str
    title: str = ""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasonResponse(BaseModel):
    answer: str
    grounded: bool
    reasoning_paths: list[ReasoningPath]
    connected_entities: list[str]
    sources: list[str]
    metrics: dict[str, float]
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    extracted_relationships: list[GraphTriple] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ExperimentRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=3)
    model: str = "llama3"


class ExperimentRun(BaseModel):
    configuration: Literal[
        "llm_only", "conventional_rag", "graph_rag_unvalidated", "neurosymbolic_graphrag"
    ]
    answer: str
    sources: list[str] = Field(default_factory=list)
    latency_seconds: float
    model_response_received: bool


class ExperimentCase(BaseModel):
    question: str
    runs: list[ExperimentRun]
    retrieval_latency_seconds: float = 0.0
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class ExperimentResponse(BaseModel):
    model: str
    cases: list[ExperimentCase]


class EvaluationRequest(BaseModel):
    query: str
    model: str = "llama3"
    traversal: Literal["bfs", "dfs", "shortest"] = "bfs"
    max_hops: int = 3
    auto_pubmed: bool = False


class EvaluationResponse(BaseModel):
    metrics: dict[str, float]
    notes: list[str]
    answer: str
    grounded: bool
    sources: list[str]
