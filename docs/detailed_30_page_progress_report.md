# Detailed M.Tech Thesis Progress Report

## Project Title

**Reducing Unsupported Biomedical Claims Using Deterministic Neuro-Symbolic Graph Traversal**

## Project Domain

Healthcare AI, Biomedical Literature Reasoning, Neuro-Symbolic AI, GraphRAG, Explainable AI

## Current Status

Local prototype implemented using Python 3.12, FastAPI, Streamlit, ChromaDB, NetworkX, PubMed retrieval, and Ollama. Controlled experimental and safety validation are incomplete.

> **Evidence limitation:** This draft describes the intended system and proposed experiments. The code currently checks graph evidence fields and reports descriptive pipeline metrics; it does not prove that claims are scientifically valid or that hallucinations are reduced. Treat performance, safety, and evaluation outcomes described as goals unless backed by recorded experiments.

## Note for Submission

This report is structured as a 30-page draft using explicit page markers. When converted to Word or PDF, formatting, font size, line spacing, diagrams, and university front matter can be adjusted to match the required institutional template.

---

<!-- Page 1 -->

# Page 1: Cover Information

## Thesis Progress Report

**Title:** Reducing Unsupported Biomedical Claims Using Deterministic Neuro-Symbolic Graph Traversal

**Degree:** M.Tech Thesis Project

**Domain:** Healthcare Artificial Intelligence

**Technology Area:** Neuro-Symbolic AI, GraphRAG, Local LLMs, Knowledge Graphs, Biomedical Information Retrieval

**Implementation Language:** Python 3.12

**Prototype Status:** Working local research prototype

**Prepared Date:** May 23, 2026

### Student Details

Name: _______________________________

Roll Number: ________________________

Department: _________________________

Institute: __________________________

Supervisor / Guide: _________________

### Project Summary

This project proposes and implements a deterministic Neuro-Symbolic GraphRAG framework for reducing hallucinations in healthcare LLM systems. The main idea is to prevent a language model from generating unsupported medical claims by forcing all answers to pass through symbolic knowledge graph validation. The prototype dynamically retrieves biomedical literature, processes documents, extracts biomedical triples, builds a deterministic knowledge graph, performs graph traversal, validates source-backed reasoning paths, and generates grounded responses using a local LLM through Ollama.

The current system is not designed as a generic chatbot. It is designed as a research prototype for explainable healthcare reasoning where every generated answer must be supported by graph evidence. If no validated graph path exists, the system abstains instead of producing a speculative answer.

<div style="page-break-after: always;"></div>

<!-- Page 2 -->

# Page 2: Abstract

Large Language Models have shown strong performance in natural language generation and biomedical question answering. However, they remain prone to hallucinations, especially when they generate answers from statistical associations rather than validated evidence. In healthcare applications, unsupported generation can be dangerous because an incorrect medical claim may mislead users, researchers, or clinicians.

This thesis project addresses the hallucination problem by designing a deterministic Neuro-Symbolic GraphRAG framework. The proposed system combines neural retrieval and local LLM generation with symbolic knowledge graph construction and graph traversal. Instead of passing retrieved text directly to an LLM, the framework first extracts biomedical entities and relationships from PubMed literature and healthcare PDFs. These extracted facts are represented as a NetworkX knowledge graph. The Symbolic Reasoner Agent then performs deterministic traversal using BFS, DFS, and shortest-path reasoning. Only source-backed graph paths are passed to the Generator Agent.

The current prototype has been implemented locally using free and open-source tools. The frontend is built with Streamlit, the backend with FastAPI, the semantic retrieval layer with ChromaDB, the knowledge graph with NetworkX, and the local LLM layer with Ollama Llama3. The system supports PubMed abstract retrieval, PDF ingestion, document chunking, embedding generation, biomedical triple extraction, graph visualization, reasoning trace display, and evaluation metrics.

The key research contribution is the enforcement of a graph evidence contract. The LLM is not allowed to answer directly from retrieved chunks. It receives only validated graph paths, verified entities, evidence sentences, and source references. This design aims to reduce unsupported associative hallucination and improve explainability in healthcare AI reasoning.

<div style="page-break-after: always;"></div>

<!-- Page 3 -->

# Page 3: Background and Motivation

Healthcare AI systems require high reliability, traceability, and explainability. Unlike general-purpose chatbots, medical question answering systems must avoid unsupported claims because incorrect answers may affect patient understanding, clinical decisions, or biomedical research interpretation. Conventional LLMs are probabilistic models trained to predict likely tokens. Even when they sound fluent and confident, they can produce claims that are not grounded in evidence.

Retrieval-Augmented Generation was introduced to reduce hallucination by providing external documents to the LLM. However, standard RAG still has limitations. It retrieves semantically similar text, but semantic similarity does not guarantee logical validity. A retrieved passage may mention related biomedical concepts without supporting the specific claim made by the model. For example, a query about diabetes treatment may retrieve papers about glucose metabolism, but the LLM may still infer treatment details that were not actually present in the retrieved context.

This motivates a stricter architecture. Instead of treating retrieval as sufficient evidence, the proposed system transforms evidence into explicit graph structures. A graph can represent biomedical relationships such as:

```text
Diabetes -> affects -> Blood Glucose
Insulin Secretion -> regulates -> Blood Glucose
Metformin -> treats -> Type 2 Diabetes
```

Once knowledge is represented as a graph, reasoning becomes more deterministic. A path must exist before the system can support an answer. This thesis therefore investigates whether graph traversal and symbolic validation can reduce hallucination in healthcare LLM systems.

The motivation is not to replace LLMs, but to constrain them. The LLM remains useful for natural language explanation, but the content it receives is restricted to verified graph evidence.

<div style="page-break-after: always;"></div>

<!-- Page 4 -->

# Page 4: Problem Statement

The central problem addressed in this thesis is unsupported associative hallucination in healthcare LLM systems. Unsupported associative hallucination occurs when an LLM connects concepts because they are statistically or semantically related, even though the specific relationship is not validated by source evidence.

In standard healthcare RAG systems, the workflow is usually:

```text
User Query -> Semantic Retrieval -> Retrieved Chunks -> LLM Answer
```

This workflow has the following weaknesses:

1. Retrieved chunks may be relevant but not sufficient.
2. The LLM may combine facts from different passages without a valid reasoning path.
3. The answer may not expose the exact evidence chain.
4. The system may not know when to abstain.
5. The evaluation of hallucination is often performed after generation rather than before generation.

The thesis problem can be stated as:

**How can a healthcare LLM system be designed so that generated answers are constrained by deterministic symbolic reasoning paths rather than unsupported neural associations?**

The proposed solution is to introduce a graph validation layer between retrieval and generation. The LLM should not receive raw retrieved context as an unrestricted prompt. Instead, biomedical text should be converted into graph triples, graph paths should be validated, and only validated evidence should be passed to the LLM.

This changes the responsibility of the LLM. It is no longer the primary reasoner. It becomes a natural language generator that explains already validated symbolic evidence.

<div style="page-break-after: always;"></div>

<!-- Page 5 -->

# Page 5: Research Objectives

The primary objective is to develop a deterministic Neuro-Symbolic GraphRAG framework that reduces hallucinations in healthcare question answering by grounding generation in validated knowledge graph paths.

The specific objectives are:

1. Build a local healthcare AI prototype using only free and open-source tools.
2. Dynamically ingest biomedical knowledge from PubMed and healthcare PDFs.
3. Process documents into chunks suitable for retrieval and extraction.
4. Store chunks and metadata in ChromaDB using sentence-transformer embeddings.
5. Extract biomedical entities and relationships from retrieved chunks.
6. Build a deterministic knowledge graph using NetworkX.
7. Implement symbolic reasoning using BFS, DFS, and shortest-path traversal.
8. Validate source-backed multi-hop graph paths.
9. Restrict the LLM generator to validated graph evidence only.
10. Display explainable reasoning chains and graph visualizations.
11. Compare standard RAG and Neuro-Symbolic GraphRAG using evaluation metrics.
12. Create a foundation for future thesis experiments on hallucination reduction.

The current prototype has implemented the main pipeline and validates the core research concept. The next stage will focus on stronger biomedical extraction, formal evaluation, baseline comparison, and experiment reporting.

The project is intentionally local-first. No paid APIs are used. The system uses Ollama for local LLM inference and PubMed APIs for free biomedical literature retrieval.

<div style="page-break-after: always;"></div>

<!-- Page 6 -->

# Page 6: Scope of the Project

The project scope is focused on healthcare AI research rather than clinical deployment. The current system is a research prototype that demonstrates how symbolic graph validation can be used to constrain LLM generation.

### Included Scope

- Biomedical literature search through PubMed.
- Healthcare PDF ingestion.
- Local document processing.
- Chunking and embedding generation.
- ChromaDB semantic retrieval.
- Biomedical entity and relationship extraction.
- Knowledge graph construction using NetworkX.
- Deterministic graph traversal.
- Grounded generation using Ollama Llama3.
- Reasoning path visualization.
- Evaluation metric dashboard.
- Local execution using Python 3.12.

### Excluded Scope

- Direct clinical diagnosis.
- Real patient-specific treatment recommendation.
- Replacement of medical professionals.
- Use of paid LLM APIs.
- Hospital-grade deployment.
- Regulated clinical decision support.

The system can answer only from available literature evidence. If the graph does not contain a validated path, the system abstains. This is not a limitation of usability alone; it is part of the safety-oriented research design.

<div style="page-break-after: always;"></div>

<!-- Page 7 -->

# Page 7: Proposed Solution Overview

The proposed solution is a deterministic multi-agent Neuro-Symbolic GraphRAG system. The architecture combines five major capabilities:

1. Neural retrieval using embeddings.
2. Symbolic representation using graph triples.
3. Deterministic reasoning using graph traversal.
4. Grounded generation using a local LLM.
5. Evaluation using faithfulness and graph coverage metrics.

The system does not follow the traditional RAG pattern where the LLM receives retrieved text directly. Instead, it follows a graph-first reasoning workflow:

```text
Retrieve Text -> Extract Triples -> Build Graph -> Traverse Graph -> Validate Path -> Generate Answer
```

The LLM receives only structured graph evidence. This prevents the model from freely inventing biomedical relationships. The generator prompt explicitly instructs the LLM to answer only using validated graph evidence and to state limitations when evidence is incomplete.

The system is implemented as a multi-agent pipeline:

- Extractor Agent
- Knowledge Graph Builder Agent
- Symbolic Reasoner Agent
- Generator Agent
- Evaluation Agent

The most important agent is the Symbolic Reasoner Agent. It performs deterministic graph traversal and acts as a gatekeeper before generation. This agent is the core of the thesis contribution.

<div style="page-break-after: always;"></div>

<!-- Page 8 -->

# Page 8: High-Level Architecture Design

The implemented architecture is shown below:

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
Grounded Explainable Response
        |
        v
Evaluation Dashboard
```

Each stage has a specific purpose. Retrieval finds candidate evidence. Extraction transforms evidence into structured triples. Graph construction stores biomedical facts. Traversal validates reasoning. Generation converts validated evidence into human-readable language. Evaluation measures whether the system is grounded and explainable.

<div style="page-break-after: always;"></div>

<!-- Page 9 -->

# Page 9: Detailed Component Architecture

The current prototype is organized into modular folders:

```text
backend/      FastAPI application and endpoints
frontend/     Streamlit research dashboard
agents/       Multi-agent orchestration and generation logic
graph/        NetworkX graph builder
reasoning/    Symbolic graph traversal engine
evaluation/   Evaluation metric implementation
api/          Pydantic request and response schemas
utils/        Document processing, vector store, logging
mcp/          PubMed retrieval adapter
tests/        Unit tests
docs/         Architecture and progress reports
data/         Local ChromaDB and graph artifacts
```

### Backend Layer

The backend exposes APIs for ingestion, graph construction, reasoning, evaluation, and logs. It is implemented using FastAPI and supports asynchronous endpoints.

### Frontend Layer

The frontend provides a Streamlit dashboard with tabs for upload, graph visualization, reasoning, evaluation, and logs.

### Agent Layer

The agent layer handles extraction, orchestration, graph building, reasoning, generation, and evaluation.

### Storage Layer

ChromaDB stores vectorized chunks. NetworkX stores symbolic graph relationships. Local files store graph exports and logs.

### Model Layer

Ollama runs Llama3 locally for grounded answer generation and optional extraction support.

<div style="page-break-after: always;"></div>

<!-- Page 10 -->

# Page 10: Data Ingestion Design

The system supports two major data ingestion sources:

1. PubMed biomedical literature.
2. Uploaded healthcare PDFs.

### PubMed Ingestion

The PubMed client uses NCBI E-utilities to search biomedical literature. The workflow is:

```text
Search Query -> PMID List -> Abstract Metadata -> Text Extraction -> Chunking -> ChromaDB
```

For each article, metadata such as PMID, title, journal, and source type is stored. This makes later reasoning explainable because graph edges can be traced back to source articles.

### PDF Ingestion

PDF ingestion uses PyMuPDF. The document processor reads every page, extracts text, cleans whitespace, and creates overlapping text chunks.

### Why Dynamic Ingestion Matters

The thesis requires the system to avoid hardcoded medical facts. Dynamic ingestion helps ensure that the knowledge graph is built from actual biomedical sources. The graph evolves based on retrieved literature rather than fixed manually written knowledge.

### Current Status

PubMed and PDF ingestion are implemented. PubMed ingestion also runs automatically during reasoning when the system needs additional evidence for a query.

<div style="page-break-after: always;"></div>

<!-- Page 11 -->

# Page 11: Document Processing and Chunking

Document processing converts raw biomedical text into manageable chunks. This stage is important because both embedding retrieval and relationship extraction operate on chunks rather than full documents.

The chunking design uses:

- Fixed maximum chunk size.
- Overlap between chunks.
- Stable chunk IDs.
- Metadata preservation.

Each chunk stores:

```json
{
  "chunk_id": "pubmed_2863979_691dd1e9b5c9",
  "text": "biomedical text...",
  "source": "pubmed:2863979",
  "metadata": {
    "pmid": "2863979",
    "title": "...",
    "source_type": "pubmed_article"
  }
}
```

Chunk overlap is used so that relationships crossing sentence boundaries are less likely to be lost. Metadata is essential because graph edges must later point back to evidence sources.

The current system uses a lightweight chunking strategy. Future work will improve chunking using section-aware splitting for biomedical PDFs. For example, abstracts, methods, results, and conclusions should be treated differently because they have different evidence value.

<div style="page-break-after: always;"></div>

<!-- Page 12 -->

# Page 12: ChromaDB Semantic Retrieval Layer

ChromaDB is used as the vector database. It stores biomedical chunks and metadata. Embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The retrieval layer is not treated as final evidence. It is used only to find candidate chunks for extraction and graph construction.

### Why Retrieval Alone Is Not Enough

Semantic retrieval can find text that is related to a query, but it cannot validate whether a specific biomedical claim is supported. Therefore, retrieved chunks must be converted into symbolic graph evidence before generation.

### Stored Data

ChromaDB stores:

- Chunk text
- Chunk ID
- Source reference
- PubMed metadata
- PDF metadata

### Current Implementation

The vector store supports:

- Adding chunks
- Querying relevant chunks
- Clearing the vector store for clean experiments

The clear function was added during prototype testing because previous noisy retrieval results could affect later graph construction. Clean experimental resets are important for reproducible thesis evaluation.

<div style="page-break-after: always;"></div>

<!-- Page 13 -->

# Page 13: Extractor Agent Design

The Extractor Agent converts biomedical chunks into graph triples. This is one of the most important stages because graph quality depends on extraction quality.

The current Extractor Agent uses:

- Deterministic biomedical term recognition.
- Rule-based relationship extraction.
- Optional spaCy support.
- Optional local Ollama-assisted extraction.

The output format is:

```text
Subject -> Predicate -> Object
```

Example:

```text
Impaired Insulin Secretion -> causes -> Hepatic Glucose Production
```

Each triple stores:

- Subject
- Predicate
- Object
- Source
- Chunk ID
- Evidence sentence
- Confidence score

### Current Supported Relationships

The prototype supports relationships such as:

- affects
- causes
- associated_with
- treats
- regulates
- expressed_in

### Recent Improvements

During testing, the extractor was improved to normalize biomedical terms such as:

```text
Plasma Glucose -> Blood Glucose
Insulin Treatment -> Insulin Therapy
Non-Insulin-Dependent Diabetes Mellitus -> Type 2 Diabetes
```

Treatment-related terms such as Metformin, insulin therapy, GLP-1 receptor agonists, SGLT2 inhibitors, lifestyle intervention, diet, and exercise were also added for better diabetes treatment reasoning.

<div style="page-break-after: always;"></div>

<!-- Page 14 -->

# Page 14: Knowledge Graph Builder Design

The Knowledge Graph Builder Agent uses NetworkX to create a directed biomedical knowledge graph.

### Node Design

Nodes represent biomedical concepts such as:

- Diseases
- Symptoms
- Hormones
- Drugs
- Genes
- Proteins
- Physiological processes
- Treatment methods

Example nodes:

```text
Diabetes
Blood Glucose
Insulin Secretion
Metformin
Type 2 Diabetes
Hepatic Glucose Production
```

### Edge Design

Edges represent source-backed biomedical relationships.

Example:

```text
Metformin -> treats -> Type 2 Diabetes
```

Each edge stores:

- Predicate
- Evidence sentence
- Source reference
- Chunk ID
- Confidence value

### Importance of Edge Metadata

The edge metadata is critical for explainability. A graph edge is not just a symbolic relationship; it is a relationship backed by a source sentence. This allows the system to display both the reasoning path and the source evidence.

### Current Status

The graph builder can add triples, clear the graph, serialize the graph, and provide graph data to the Streamlit visualization.

<div style="page-break-after: always;"></div>

<!-- Page 15 -->

# Page 15: Symbolic Reasoner Agent Design

The Symbolic Reasoner Agent is the core research contribution of the prototype. It performs deterministic reasoning over the knowledge graph.

### Supported Algorithms

The current system supports:

- Breadth-First Search
- Depth-First Search
- Shortest-path traversal

### Reasoning Process

The reasoner follows these steps:

1. Identify query-related graph entities.
2. Select anchor nodes.
3. Traverse graph paths up to a maximum hop limit.
4. Validate that every edge has source-backed evidence.
5. Return only validated reasoning paths.

### Evidence Validation

A path is valid only if every step contains:

- Evidence sentence
- Source reference
- Chunk ID

If the graph does not contain a supported path, no answer is generated. This is the main hallucination-control mechanism.

### Incoming Treatment Edge Support

Treatment facts often appear in the graph as:

```text
Metformin -> treats -> Type 2 Diabetes
```

When the user asks about Type 2 Diabetes, the reasoner now exposes this as:

```text
Type 2 Diabetes -> treated_by -> Metformin
```

This improves treatment-oriented reasoning without changing the underlying graph fact.

<div style="page-break-after: always;"></div>

<!-- Page 16 -->

# Page 16: Generator Agent Design

The Generator Agent uses Ollama Llama3 locally. Its task is not to perform free-form medical reasoning. Its task is to convert validated graph evidence into a readable explanation.

The generator receives only:

- Validated graph paths
- Connected entities
- Evidence sentences
- Source references

It does not receive raw retrieved chunks as unrestricted context. This is intentional because raw text can allow the LLM to infer unsupported claims.

### Generator Prompt Design

The prompt instructs the model to:

- Answer only using validated graph evidence.
- Avoid adding unsupported medical facts.
- State limitations when evidence is incomplete.
- Avoid presenting claims as clinical advice.

### Abstention Behavior

If no graph path exists, the generator returns:

```text
I cannot provide a grounded answer because the knowledge graph did not contain a validated, source-backed reasoning path for this query.
```

This behavior is a feature, not a failure. It prevents unsupported generation.

### Current Observation

During testing, the system correctly abstained when the graph did not contain a treatment path. After improvements, it can automatically retrieve PubMed evidence and attempt to build additional treatment-related graph paths.

<div style="page-break-after: always;"></div>

<!-- Page 17 -->

# Page 17: Streamlit Frontend Design

The frontend is implemented using Streamlit as a research dashboard.

### Sidebar Controls

The sidebar supports:

- FastAPI backend URL
- Local model selection
- Traversal algorithm selection
- Maximum hop selection

### Main Tabs

The dashboard includes five main tabs:

1. Upload
2. Knowledge Graph
3. Reasoning
4. Evaluation
5. Logs

### Upload Tab

The Upload tab supports:

- PubMed query input
- Maximum article selection
- PDF upload
- Graph build button
- Graph clear button

### Knowledge Graph Tab

This tab displays:

- Number of biomedical entities
- Number of verified relationships
- Graph visualization
- Edge table

### Reasoning Tab

This tab displays:

- Healthcare query input
- Grounded answer
- Traversal paths
- Evidence steps
- Evaluation metrics
- Highlighted graph visualization

### Evaluation Tab

This tab supports comparison between standard RAG and Neuro-Symbolic GraphRAG using available metrics.

### Logs Tab

The logs tab helps debug ingestion, extraction, graph construction, and reasoning.

<div style="page-break-after: always;"></div>

<!-- Page 18 -->

# Page 18: FastAPI Backend Design

The backend is implemented using FastAPI. It exposes the system pipeline as API endpoints.

### Implemented Endpoints

```text
GET  /health
POST /ingest/pubmed
POST /ingest/pdf
POST /graph/build
GET  /graph
POST /graph/clear
POST /reason
POST /evaluate
GET  /logs
```

### Endpoint Responsibilities

`/ingest/pubmed` retrieves PubMed abstracts and stores chunks in ChromaDB.

`/ingest/pdf` processes uploaded healthcare PDFs.

`/graph/build` extracts triples and builds the NetworkX graph.

`/reason` performs retrieval, graph expansion, symbolic traversal, grounded generation, and evaluation.

`/evaluate` computes metrics for comparing standard RAG and GraphRAG behavior.

`/graph/clear` resets the graph and vector store for clean experiments.

### Backend Orchestration

The backend uses a central orchestrator class that connects:

- Document processor
- Vector store
- PubMed client
- Extractor Agent
- Knowledge Graph Builder Agent
- Symbolic Reasoner Agent
- Generator Agent
- Evaluation Agent

This modular backend design makes it easier to improve individual research components without rewriting the full application.

<div style="page-break-after: always;"></div>

<!-- Page 19 -->

# Page 19: Multi-Agent Workflow

The project follows a multi-agent design. Each agent has a specialized responsibility.

### Extractor Agent

Processes biomedical text chunks and extracts:

- Entities
- Relationships
- Graph triples
- Evidence sentences

### Knowledge Graph Builder Agent

Builds and updates the NetworkX knowledge graph. It adds nodes, edges, and metadata.

### Symbolic Reasoner Agent

Performs deterministic graph traversal and path validation. It is responsible for hallucination prevention.

### Generator Agent

Generates natural language answers only from validated graph evidence.

### Evaluation Agent

Computes metrics such as:

- Faithfulness
- Answer relevancy
- Context precision
- Retrieval accuracy
- Graph coverage
- Hallucination reduction

### LangGraph Role

The project includes a LangGraph workflow wrapper. This provides a research-friendly way to represent the pipeline as a graph of agent states:

```text
Retrieve -> Extract -> Build Graph -> Reason -> Generate
```

Future work will make LangGraph the main orchestration layer for advanced agent routing and experiment tracing.

<div style="page-break-after: always;"></div>

<!-- Page 20 -->

# Page 20: Evaluation Design

The evaluation goal is to compare standard RAG with Neuro-Symbolic GraphRAG.

### Standard RAG Baseline

The standard RAG baseline follows:

```text
Query -> Semantic Retrieval -> LLM Answer
```

This baseline will be used to measure whether the LLM produces unsupported claims from retrieved text.

### Neuro-Symbolic GraphRAG

The proposed method follows:

```text
Query -> Retrieval -> Triple Extraction -> Graph Traversal -> Validated Paths -> LLM Answer
```

### Metrics

The current prototype computes local proxy metrics:

- Faithfulness
- Answer relevancy
- Context precision
- Retrieval accuracy
- Graph coverage
- Hallucination reduction

### Future Evaluation Tools

The project dependencies include:

- DeepEval
- RAGAS

These tools will be used for more formal evaluation during the next phase.

### Planned Evaluation Dataset

The future evaluation dataset will include healthcare questions from diabetes, cardiovascular disease, kidney disease, hypertension, inflammation, and drug-treatment reasoning. Each answer will be checked for graph support, evidence traceability, and hallucination rate.

<div style="page-break-after: always;"></div>

<!-- Page 21 -->

# Page 21: Current Testing and Results

The current prototype has been tested using diabetes-related biomedical queries.

### Example Query

```text
How does diabetes affect blood glucose regulation and what are the treatment ways and drugs?
```

### Observed Behavior

The system retrieved PubMed literature, built graph relationships, and generated a grounded response only when a source-backed graph path existed.

When the graph contained insufficient evidence, the system abstained. This confirmed that the generator was constrained by symbolic validation.

### Improvements Made During Testing

Several practical improvements were implemented:

1. Fixed Streamlit page configuration issue.
2. Fixed graph edge serialization conflict.
3. Added unique graph visualization keys.
4. Improved biomedical entity normalization.
5. Added treatment and drug recognition.
6. Added incoming treatment-edge traversal.
7. Added automatic PubMed retrieval during reasoning.
8. Added graph and vector-store clearing.

### Unit Tests

Current Python 3.12 test status:

```text
2 tests passed
```

The tests validate:

- Source-backed multi-hop graph reasoning.
- Incoming treatment-edge traversal.

<div style="page-break-after: always;"></div>

<!-- Page 22 -->

# Page 22: Research Contribution Achieved So Far

The current prototype demonstrates the main thesis hypothesis: hallucinations can be reduced by enforcing symbolic graph grounding before generation.

### Key Contributions

1. A local healthcare GraphRAG prototype using open-source tools.
2. A deterministic graph validation layer between retrieval and generation.
3. A graph evidence contract that restricts LLM input.
4. A multi-agent architecture for extraction, graph building, reasoning, generation, and evaluation.
5. An abstention mechanism for unsupported medical claims.
6. Explainable reasoning paths with evidence references.
7. Dynamic PubMed ingestion and graph expansion.

### Difference from Generic RAG

Generic RAG retrieves documents and lets the LLM answer from them. This prototype retrieves documents, but then forces the evidence through graph extraction and traversal before generation.

### Difference from a Chatbot

The system is not a chatbot because it does not answer freely. It behaves as a graph-grounded biomedical reasoning system. Its main output is not just an answer, but also:

- Reasoning path
- Connected entities
- Evidence sentence
- Source references
- Evaluation metrics

<div style="page-break-after: always;"></div>

<!-- Page 23 -->

# Page 23: Limitations of Current Prototype

The current prototype is functional, but several limitations remain.

### Extraction Limitations

The current extractor is rule-based with optional LLM assistance. It may miss complex biomedical relationships or extract incomplete triples. Biomedical relation extraction is a challenging NLP problem and requires stronger domain-specific models.

### Entity Normalization Limitations

The system currently uses surface-level normalization. It needs stronger mapping to biomedical ontologies such as:

- MeSH
- UMLS
- SNOMED CT
- DrugBank

### PubMed Abstract Limitation

Many PubMed abstracts do not contain full treatment details. Full-text articles may be required for deeper treatment reasoning.

### Evaluation Limitation

The current metrics are proxy metrics. Formal evaluation with DeepEval, RAGAS, and curated benchmark datasets is still pending.

### Clinical Limitation

The system is not a clinical decision support system. It is a research prototype and should not be used for real medical decision-making.

### LLM Limitation

Even though generation is constrained, the LLM may still phrase evidence imperfectly. Future work will include stricter structured answer templates and citation checking.

<div style="page-break-after: always;"></div>

<!-- Page 24 -->

# Page 24: Future Improvements - Technical Roadmap

The next development phase will improve system accuracy, robustness, and research evaluation.

### 1. Biomedical NER

Integrate SciSpacy or biomedical transformer-based NER to identify:

- Diseases
- Drugs
- Genes
- Proteins
- Symptoms
- Procedures

### 2. Relation Extraction

Improve relation extraction using:

- Local LLM structured extraction
- Biomedical relation classifiers
- Dependency parsing
- Prompted JSON extraction

### 3. Ontology Alignment

Map extracted entities to biomedical ontologies:

- MeSH descriptors
- UMLS CUIs
- DrugBank drug IDs

### 4. Graph Confidence Scoring

Assign stronger confidence to relationships supported by multiple papers.

### 5. Citation-Aware Generation

Force every answer sentence to cite the graph edge or source PMID supporting it.

### 6. Better Query Decomposition

Complex queries should be decomposed into subquestions such as:

```text
Mechanism question
Treatment question
Drug question
Risk question
```

Each subquestion should produce separate graph paths and evidence.

<div style="page-break-after: always;"></div>

<!-- Page 25 -->

# Page 25: Future Improvements - Evaluation Roadmap

Formal evaluation is essential for the final thesis.

### Standard RAG Baseline

A baseline system will be implemented where retrieved chunks are directly passed to the LLM. This will be compared against the GraphRAG system.

### Evaluation Questions

The benchmark will include questions such as:

- How does diabetes affect blood glucose?
- What drugs are used to treat type 2 diabetes?
- How is hypertension linked to kidney disease?
- What is the relationship between inflammation and cardiovascular disease?
- What molecular pathways connect insulin resistance and obesity?

### Evaluation Dimensions

The evaluation will measure:

- Whether the answer is supported by evidence.
- Whether the reasoning path exists in the graph.
- Whether unsupported claims are generated.
- Whether citations are correct.
- Whether graph coverage improves over time.

### DeepEval and RAGAS

DeepEval and RAGAS will be integrated to calculate:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall
- Hallucination score

### Human Evaluation

If possible, a small manual evaluation will be conducted where generated answers are checked against PubMed evidence.

<div style="page-break-after: always;"></div>

<!-- Page 26 -->

# Page 26: Future Improvements - System Design Roadmap

The current prototype can be improved into a more complete research platform.

### Experiment Manager

Add a module that stores:

- Query
- Retrieved papers
- Extracted triples
- Graph paths
- Generated answer
- Evaluation metrics
- Timestamp

### Report Export

Add export buttons for:

- Reasoning trace PDF
- Graph JSON
- Evaluation CSV
- Experiment summary

### Graph Visualization Improvements

Improve graph visualization by:

- Highlighting validated paths.
- Coloring node types.
- Showing edge labels.
- Showing PubMed evidence on hover.
- Filtering by source or relationship type.

### Reproducibility

Add experiment configuration files so that results can be reproduced.

### Performance

Add caching for:

- PubMed searches
- Embeddings
- Extracted triples
- Graph snapshots

### Better Local Model Support

Support additional Ollama models such as:

- Mistral
- Phi
- Llama3.1
- Meditron-style local medical models if available

<div style="page-break-after: always;"></div>

<!-- Page 27 -->

# Page 27: Month-Wise Plan Until Final Thesis Submission

The project will continue for the remaining thesis months with the following roadmap.

## Month 1: Prototype Stabilization

- Fix runtime issues.
- Improve biomedical extraction.
- Improve graph visualization.
- Add automatic PubMed expansion.
- Add clean experiment reset.
- Improve README and setup instructions.

## Month 2: Biomedical Intelligence Improvement

- Integrate SciSpacy or biomedical NER.
- Improve drug and disease normalization.
- Add relation extraction templates.
- Add query decomposition.
- Improve treatment reasoning.

## Month 3: Evaluation Baseline

- Implement standard RAG baseline.
- Create benchmark query set.
- Store answers from both systems.
- Compare hallucination rate.
- Start DeepEval and RAGAS integration.

## Month 4: Graph Reasoning Enhancement

- Add weighted graph traversal.
- Add evidence aggregation.
- Add repeated-source confidence scoring.
- Add path ranking based on source quality.
- Improve multi-hop reasoning.

## Month 5: Thesis Experimentation

- Run experiments on multiple healthcare topics.
- Collect metric tables.
- Generate graphs and charts.
- Analyze failure cases.
- Compare GraphRAG and standard RAG.

## Month 6: Final Thesis Writing

- Write methodology chapter.
- Write implementation chapter.
- Write results and discussion.
- Add architecture diagrams.
- Prepare final presentation and demo.

<div style="page-break-after: always;"></div>

<!-- Page 28 -->

# Page 28: Risk Analysis and Mitigation

### Risk 1: Poor Biomedical Extraction

Rule-based extraction may miss relationships.

Mitigation:

- Add SciSpacy.
- Add local LLM JSON extraction.
- Add relation extraction models.

### Risk 2: Insufficient PubMed Abstract Evidence

Abstracts may not contain complete treatment information.

Mitigation:

- Support full-text PDFs.
- Use open-access PMC where possible.
- Improve PDF section extraction.

### Risk 3: LLM Still Adds Unsupported Language

Even with graph evidence, the LLM may phrase an unsupported generalization.

Mitigation:

- Use stricter prompt templates.
- Add sentence-level citation validation.
- Add post-generation claim checking.

### Risk 4: Evaluation Complexity

Hallucination evaluation is difficult.

Mitigation:

- Use multiple metrics.
- Include human review.
- Compare against baseline RAG.
- Store all experiment artifacts.

### Risk 5: System Performance

Local embeddings and PubMed retrieval can be slow.

Mitigation:

- Add caching.
- Reduce repeated ingestion.
- Store graph snapshots.
- Use batch embedding.

<div style="page-break-after: always;"></div>

<!-- Page 29 -->

# Page 29: Expected Final Thesis Outcomes

By the final submission, the project is expected to deliver:

1. A complete local Neuro-Symbolic GraphRAG prototype.
2. A standard RAG baseline system.
3. Biomedical knowledge graph construction pipeline.
4. Deterministic graph traversal and validation module.
5. Explainable healthcare answer generation.
6. Graph visualization and reasoning trace dashboard.
7. Evaluation results comparing hallucination behavior.
8. Thesis documentation with architecture, methodology, experiments, and results.

### Expected Research Findings

The expected finding is that graph-grounded generation reduces unsupported claims compared to standard RAG. The system may sometimes abstain more often than standard RAG, but this is expected because abstention is safer than hallucination in healthcare reasoning.

### Expected Demonstration

The final demo should show:

- A healthcare query.
- PubMed retrieval.
- Graph construction.
- Validated path traversal.
- Grounded response.
- Evidence references.
- Comparison with standard RAG.
- Evaluation metrics.

### Success Criteria

The project will be considered successful if it demonstrates that deterministic graph validation improves answer faithfulness and explainability compared to direct retrieval-based generation.

<div style="page-break-after: always;"></div>

<!-- Page 30 -->

# Page 30: Conclusion

This thesis project has progressed from concept to a working local research prototype. The current system implements the core idea of deterministic Neuro-Symbolic GraphRAG for healthcare AI. It retrieves biomedical literature, processes documents, builds a knowledge graph, performs symbolic traversal, validates reasoning paths, and generates grounded responses through a local LLM.

The most important achievement so far is the implementation of a graph evidence contract. The LLM is not allowed to answer directly from retrieved chunks. It receives only validated graph evidence. This design directly addresses the thesis problem of unsupported associative hallucination.

The prototype has also demonstrated practical research behavior. When evidence is available, it can produce grounded answers with reasoning paths. When evidence is insufficient, it abstains. This abstention behavior is essential in healthcare AI because unsupported answers can be risky.

The next phase will focus on improving biomedical extraction, integrating stronger evaluation methods, implementing a standard RAG baseline, and running controlled experiments. Future improvements will also include ontology alignment, citation-aware generation, graph confidence scoring, query decomposition, and better visualization.

Overall, the project is on track toward a complete M.Tech thesis contribution. It already provides a working foundation for demonstrating how deterministic symbolic graph traversal can reduce hallucination in healthcare LLM systems while improving transparency and explainability.

## Final Statement

The current prototype is not merely a chatbot or a simple RAG application. It is a research-oriented Neuro-Symbolic GraphRAG framework that uses deterministic graph grounding as the central mechanism for hallucination control in healthcare AI.

