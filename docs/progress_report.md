# M.Tech Thesis Project Progress Report

## Thesis Title

**Reducing Unsupported Biomedical Claims Using Deterministic Neuro-Symbolic Graph Traversal**

## Student Project Status

**Submission Target:** To be confirmed  
**Current Prototype Stage:** Local prototype; experimental and safety validation incomplete  
**Domain:** Healthcare AI / Biomedical Question Answering  
**Implementation Stack:** Python 3.12, FastAPI, Streamlit, LangGraph/LangChain, ChromaDB, NetworkX, Ollama Llama3, PubMed API, PyMuPDF

---

## 1. Introduction

> **Status clarification:** This report describes intended design alongside implemented prototype features. Documentation of a feature is not experimental verification. The current implementation does not establish that hallucinations are eliminated or reduced; controlled baseline experiments and biomedical safety evaluation remain necessary.

Large Language Models are increasingly used for healthcare question answering, biomedical literature analysis, and clinical decision-support research. However, conventional LLM-based Retrieval-Augmented Generation systems can still produce unsupported associative claims, especially when semantic retrieval provides related text but does not enforce explicit evidence-backed reasoning.

This thesis project investigates a deterministic Neuro-Symbolic GraphRAG framework that reduces hallucinations by constraining answer generation through symbolic knowledge graph traversal. Instead of allowing the LLM to answer directly from retrieved chunks, the system converts biomedical literature into explicit graph triples, validates reasoning paths using deterministic graph algorithms, and only then passes verified graph evidence to the local LLM.

The current prototype focuses on healthcare AI reasoning using biomedical papers, PubMed abstracts, and uploaded PDFs.

---

## 2. Research Objective

The research question is whether a local graph-grounded evidence gate can reduce unsupported biomedical claims compared with controlled baselines. The current implementation does not establish that outcome.

The system aims to:

- Dynamically ingest biomedical literature from PubMed and healthcare PDFs.
- Build deterministic biomedical knowledge graphs.
- Perform symbolic graph traversal using BFS, DFS, and shortest-path reasoning.
- Validate source-backed reasoning paths.
- Generate grounded and explainable healthcare responses.
- Abstain when no validated evidence path is available.
- Planned: compare standard RAG against Neuro-Symbolic GraphRAG using annotated claim support and other reproducible metrics.

---

## 3. Problem Statement

Traditional RAG systems retrieve semantically similar passages and provide them directly to an LLM. In healthcare, this is risky because semantic similarity does not guarantee logical or biomedical validity. A model may combine unrelated facts, infer unsupported treatments, or produce medically unsafe explanations.

The proposed system addresses this by introducing a deterministic reasoning layer:

1. Retrieve biomedical evidence.
2. Extract explicit entities and relationships.
3. Build a symbolic knowledge graph.
4. Traverse the graph deterministically.
5. Validate source-backed graph paths.
6. Generate responses only from verified paths.

If no validated path exists, the system abstains instead of hallucinating.

---

## 4. Current System Architecture

The implemented prototype follows the required thesis architecture:

```text
User Healthcare Query
        |
        v
Streamlit Frontend
        |
        v
FastAPI Backend
        |
        v
PubMed Retrieval / PDF Upload
        |
        v
Document Processing and Chunking
        |
        v
Embedding Generation
        |
        v
ChromaDB Semantic Retrieval
        |
        v
Extractor Agent
        |
        v
Entity and Relationship Extraction
        |
        v
Knowledge Graph Builder Agent
        |
        v
NetworkX Knowledge Graph
        |
        v
Symbolic Reasoner Agent
        |
        v
BFS / DFS / Shortest Path Validation
        |
        v
Validated Subgraph
        |
        v
Generator Agent
        |
        v
Grounded Explainable Healthcare Response
        |
        v
Evaluation Dashboard
```

---

## 5. Implemented Modules

### 5.1 FastAPI Backend

A backend service has been implemented using FastAPI. It exposes endpoints for:

- Health checking
- PubMed ingestion
- PDF ingestion
- Knowledge graph construction
- Graph retrieval
- Graph clearing
- Deterministic reasoning
- Evaluation
- Runtime logs

The backend orchestrates the entire Neuro-Symbolic GraphRAG pipeline.

### 5.2 Streamlit Frontend

A professional research dashboard has been implemented using Streamlit. It contains the following tabs:

- **Upload:** PubMed ingestion and PDF upload
- **Knowledge Graph:** graph visualization and edge table
- **Reasoning:** healthcare query, grounded answer, traversal path, metrics
- **Evaluation:** comparison of standard RAG and GraphRAG metrics
- **Logs:** system execution logs

The UI supports selecting a local model, traversal algorithm, and maximum reasoning hops.

### 5.3 PubMed Retrieval

The system integrates PubMed retrieval through the free NCBI E-utilities API. It supports:

- Biomedical article search
- PMID retrieval
- Metadata extraction
- Abstract ingestion
- Chunking and embedding

The current system also supports automatic PubMed retrieval during reasoning, making the prototype behave more like a healthcare research search engine.

### 5.4 PDF Processing

Healthcare PDF ingestion has been implemented using PyMuPDF. Uploaded PDFs are:

- Parsed page by page
- Converted into text
- Cleaned
- Split into overlapping chunks
- Stored in ChromaDB

### 5.5 ChromaDB Vector Store

ChromaDB is used as the semantic retrieval layer. It stores:

- Biomedical chunks
- Metadata
- Source references
- Chunk IDs

Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.

### 5.6 Extractor Agent

The Extractor Agent extracts biomedical entities and relationships from retrieved chunks. The current version uses:

- Deterministic biomedical term recognition
- Rule-based relationship extraction
- Optional local LLM-assisted extraction through Ollama
- spaCy fallback support

Extracted triples include:

```text
Subject -> Predicate -> Object
```

Example:

```text
Impaired Insulin Secretion -> causes -> Hepatic Glucose Production
```

### 5.7 Knowledge Graph Builder Agent

The Knowledge Graph Builder Agent uses NetworkX to build a directed biomedical graph. Nodes represent biomedical concepts such as:

- Diseases
- Symptoms
- Hormones
- Drugs
- Proteins
- Medical concepts

Edges represent evidence-backed biomedical relationships. Each edge stores:

- Predicate
- Source
- Chunk ID
- Evidence sentence
- Confidence value

### 5.8 Symbolic Reasoner Agent

This is the core research contribution of the prototype.

The Symbolic Reasoner Agent performs deterministic traversal over the knowledge graph. It supports:

- BFS traversal
- DFS traversal
- Shortest-path reasoning
- Multi-hop reasoning
- Source-backed path validation
- Incoming treatment-edge reasoning

For example, if the graph contains:

```text
Metformin -> treats -> Type 2 Diabetes
```

the reasoner can expose it from the disease perspective as:

```text
Type 2 Diabetes -> treated_by -> Metformin
```

The reasoner rejects unsupported paths and only returns verified graph evidence.

### 5.9 Generator Agent

The Generator Agent uses local Llama3 through Ollama. It receives only:

- Validated graph paths
- Verified entities
- Evidence sentences
- Source references

It does not receive unrestricted retrieved chunks. This constrains the generation context, but does not ensure every generated claim is supported.

If no validated graph evidence is available, the generator abstains with a message explaining that the graph does not support the answer.

### 5.10 Evaluation Agent

The evaluation module currently reports descriptive structural proxies only:

- Whether a path was found
- Number of paths and evidence steps
- Number of distinct sources
- Abstention flag
- Query-token overlap proxy

These values are not correctness, retrieval-accuracy, faithfulness, or hallucination-reduction measurements. The project dependencies include DeepEval and RAGAS, but benchmark experiments are not implemented.

---

## 6. Current Prototype Features

The following features are currently working:

- Local execution using Python 3.12.
- Local LLM generation using Ollama Llama3.
- PubMed biomedical abstract ingestion.
- Healthcare PDF upload and processing.
- Biomedical chunking and embedding.
- ChromaDB semantic retrieval.
- Rule-based biomedical triple extraction.
- Dynamic NetworkX knowledge graph construction.
- Deterministic BFS/DFS/shortest-path reasoning.
- Treatment-edge traversal support.
- Grounded answer generation from validated graph paths.
- Abstention when graph evidence is insufficient.
- Streamlit dashboard with graph visualization.
- Evaluation metrics dashboard.
- Logging system.
- Unit tests for deterministic graph reasoning.

---

## 7. Initial Testing and Observations

The prototype was tested using diabetes-related biomedical queries.

Example query:

```text
How does diabetes affect blood glucose regulation and what are the treatment ways and drugs?
```

The system retrieved PubMed literature, extracted graph relationships, validated graph paths, and generated a grounded response only from available graph evidence.

An important observation is that the system correctly abstains when no validated treatment or drug path exists. This behavior is aligned with the thesis goal because the system prioritizes evidence-grounding over generating fluent but unsupported healthcare claims.

During testing, improvements were made to:

- Normalize biomedical entities such as `Plasma Glucose` to `Blood Glucose`.
- Improve treatment and drug entity recognition.
- Support automatic PubMed retrieval during reasoning.
- Support incoming treatment edges.
- Clear both graph and vector store for clean experiments.

Current unit test status:

```text
2 tests passed under Python 3.12
```

---

## 8. Research Contribution So Far

The main contribution implemented so far is a deterministic graph-grounding pipeline that restricts LLM generation using symbolic graph traversal.

Key research contributions include:

- A graph-first healthcare RAG pipeline.
- Source-backed biomedical triple extraction.
- Deterministic multi-hop graph traversal.
- Evidence path validation before generation.
- Abstention mechanism for unsupported queries.
- Local-only architecture without paid APIs.
- Explainable reasoning chain with graph visualization.

This differentiates the prototype from a generic chatbot or simple RAG system because the LLM is not allowed to directly answer from retrieved text.

---

## 9. Limitations Identified

The current prototype is functional but still has research limitations:

- Rule-based triple extraction can miss complex biomedical relationships.
- Biomedical entity recognition needs improvement using domain-specific models such as SciSpacy or biomedical NER.
- PubMed abstracts may not always contain enough treatment-level detail.
- Full-text paper ingestion is not yet deeply optimized.
- DeepEval and RAGAS are included but not yet fully integrated into benchmark experiments.
- The graph currently uses surface-level entity normalization and needs stronger ontology alignment.
- Full clinical safety validation is not implemented; the system is for research only.

---

## 10. Planned Work for Next Phase

The next phase of the thesis will focus on improving research quality, evaluation rigor, and biomedical accuracy.

Planned tasks:

1. Integrate biomedical NER using SciSpacy or biomedical transformer models.
2. Improve relation extraction using local LLM-assisted structured extraction.
3. Add ontology alignment with UMLS, MeSH, or SNOMED-style terminology where feasible.
4. Improve PubMed retrieval ranking and query expansion.
5. Add graph confidence scoring based on repeated evidence across papers.
6. Integrate DeepEval and RAGAS for benchmark-based evaluation.
7. Create a standard RAG baseline for comparison.
8. Measure hallucination reduction over a curated healthcare QA dataset.
9. Improve graph visualization with highlighted validated subgraphs.
10. Add exportable experiment reports for thesis evaluation.

---

## 11. Three-Month Development Plan

### Month 1: Foundation and Prototype Architecture

- Literature review on hallucination reduction, GraphRAG, Neuro-Symbolic AI, and healthcare QA.
- Finalize system architecture.
- Implement FastAPI backend and Streamlit frontend.
- Implement PubMed/PDF ingestion.
- Implement ChromaDB retrieval.
- Build initial NetworkX graph pipeline.

### Month 2: Neuro-Symbolic Reasoning and Multi-Agent Framework

- Improve Extractor Agent.
- Implement Knowledge Graph Builder Agent.
- Implement Symbolic Reasoner Agent with BFS/DFS/shortest path.
- Implement Generator Agent using graph-only evidence.
- Add abstention mechanism.
- Add graph visualization and reasoning trace.

### Month 3: Evaluation, Experiments, and Thesis Report

- Implement standard RAG baseline.
- Integrate DeepEval and RAGAS metrics.
- Evaluate hallucination reduction.
- Perform experiments on healthcare queries.
- Improve biomedical extraction accuracy.
- Prepare thesis documentation, diagrams, and results.

---

## 12. Current File Structure

```text
backend/      FastAPI backend
frontend/     Streamlit dashboard
agents/       Extractor, generator, orchestrator, LangGraph workflow
graph/        NetworkX knowledge graph builder
reasoning/    Deterministic symbolic reasoner
evaluation/   Evaluation metrics
api/          Pydantic schemas
utils/        Logging, vector store, document processing
mcp/          PubMed retrieval adapter
tests/        Unit tests
docs/         Architecture and progress documentation
data/         Local ChromaDB and generated graph storage
```

---

## 13. Conclusion

At the current stage, the project has a local prototype with ingestion, graph construction, traversal, and local LLM generation components. The traversal checks source/evidence fields and excludes extracted edges labeled negative or uncertain; it does not verify scientific correctness or guarantee generated answers are grounded.

The prototype implements a testable version of the thesis idea. Whether it reduces unsupported claims must be established through controlled experiments against baselines using annotated biomedical questions and evidence.

The next stage will focus on improving biomedical extraction quality, integrating stronger evaluation metrics, comparing against a standard RAG baseline, and preparing experimental results for the final thesis submission.
