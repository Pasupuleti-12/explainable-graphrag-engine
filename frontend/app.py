from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Biomedical Literature Evidence Explorer", layout="wide")

API_URL = st.sidebar.text_input("FastAPI URL", value="http://localhost:8000")
MODEL = st.sidebar.selectbox("Local model", ["llama3", "llama3.1", "mistral", "phi3"])
TRAVERSAL = st.sidebar.selectbox("Traversal", ["bfs", "dfs", "shortest"])
MAX_HOPS = st.sidebar.slider("Max hops", 1, 5, 3)

st.title("Biomedical Literature Evidence Explorer")


def post(path: str, payload: dict | None = None, **kwargs):
    try:
        timeout = kwargs.pop("timeout", 120)
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Cannot reach the API at {API_URL}. Start FastAPI in another terminal. Details: {exc}")
        return None


def get(path: str):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.sidebar.warning(f"API unavailable at {API_URL}. Start FastAPI in another terminal.")
        return {}


def draw_graph(graph_payload: dict, highlighted: set[str] | None = None, chart_key: str = "knowledge_graph"):
    highlighted = highlighted or set()
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    if not nodes:
        st.info("Build the graph to visualize biomedical entities and relationships.")
        return
    ids = [node["id"] for node in nodes]
    positions = {node: (idx % 8, idx // 8) for idx, node in enumerate(ids)}
    edge_x, edge_y = [], []
    for edge in edges:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [positions[node][0] for node in ids]
    node_y = [positions[node][1] for node in ids]
    colors = ["#d62728" if node in highlighted else "#1f77b4" for node in ids]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=18, color=colors),
            text=ids,
            textposition="top center",
        )
    )
    fig.update_layout(height=560, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


tabs = st.tabs(["Upload", "Knowledge Graph", "Reasoning", "Experiments", "Logs"])

with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.subheader("PubMed Ingestion")
        pubmed_query = st.text_input("Biomedical literature search", "diabetes insulin blood glucose")
        max_results = st.number_input("Max articles", 1, 20, 5)
        if st.button("Search and ingest PubMed"):
            with st.spinner("Retrieving PubMed abstracts and embedding chunks"):
                st.json(post("/ingest/pubmed", {"query": pubmed_query, "max_results": int(max_results)}))
    with right:
        st.subheader("PDF Ingestion")
        upload = st.file_uploader("Healthcare PDF", type=["pdf"])
        if upload and st.button("Upload PDF"):
            files = {"file": (upload.name, upload.getvalue(), "application/pdf")}
            st.json(post("/ingest/pdf", files=files, timeout=120))
    if st.button("Build deterministic knowledge graph"):
        st.json(post(f"/graph/build?model={MODEL}"))
    if st.button("Clear graph"):
        st.json(post("/graph/clear"))

with tabs[1]:
    graph_payload = get("/graph")
    st.metric("Biomedical entities", len(graph_payload.get("nodes", [])))
    st.metric("Extracted relationships", len(graph_payload.get("edges", [])))
    draw_graph(graph_payload, chart_key="knowledge_graph_tab")
    st.dataframe(pd.DataFrame(graph_payload.get("edges", [])), use_container_width=True)

with tabs[2]:
    query = st.text_area("Healthcare query", "How does diabetes affect blood glucose regulation?")
    if st.button("Run symbolic GraphRAG reasoning"):
        payload = {"query": query, "traversal": TRAVERSAL, "max_hops": MAX_HOPS, "model": MODEL}
        with st.spinner("Retrieving passages and checking evidence paths"):
            result = post("/reason", payload)
        if result:
            st.session_state["last_reason_result"] = result
            st.session_state["last_reason_query"] = query

    result = st.session_state.get("last_reason_result")
    if result:
        result_query = st.session_state.get("last_reason_query", query)
        st.subheader("Answer")
        st.write(result["answer"])
        if not result["grounded"]:
            st.warning("The system abstained because it found no eligible positive evidence path.")
        st.caption("A path is a trace through extracted statements; it does not establish scientific correctness.")

        st.subheader("Evidence used")
        used_refs = {step["reference"] for path in result["reasoning_paths"] for step in path["steps"]}
        for evidence in result.get("evidence", []):
            reference = f'{evidence["source"]}#{evidence["chunk_id"]}'
            with st.expander(f'{"Used · " if reference in used_refs else "Retrieved · "}{evidence.get("title") or evidence["source"]} · {reference}'):
                st.write(evidence["text"])
                if evidence.get("metadata"):
                    st.json(evidence["metadata"])

        st.subheader("Graph paths")
        if result["reasoning_paths"]:
            for path in result["reasoning_paths"]:
                st.code(" -> ".join(path["entities"]))
                st.dataframe(pd.DataFrame(path["steps"]), use_container_width=True)
        else:
            st.info("No source-backed path was found, so the system abstained.")

        relationships = result.get("extracted_relationships", [])
        if relationships:
            st.subheader("Extracted relationships and conflicting evidence")
            rel_df = pd.DataFrame(relationships)
            keys = ["subject", "predicate", "object"]
            polarity_counts = rel_df.groupby(keys)["polarity"].nunique()
            conflicts = set(polarity_counts[polarity_counts > 1].index.tolist())
            rel_df["conflicting_polarity"] = [
                tuple(row[key] for key in keys) in conflicts for _, row in rel_df.iterrows()
            ]
            if rel_df["conflicting_polarity"].any():
                st.warning("Some extracted source statements disagree in polarity. Review the passages before drawing a conclusion.")
            st.dataframe(rel_df, use_container_width=True)

        if result.get("limitations"):
            st.subheader("Limitations")
            for limitation in result["limitations"]:
                st.write(f"- {limitation}")
        st.json(result["metrics"])
        draw_graph(get("/graph"), highlighted=set(result["connected_entities"]), chart_key="reasoning_graph")

        report_lines = [
            "# Biomedical evidence report",
            "",
            f'**Question:** {result_query}',
            "",
            "## Answer",
            result["answer"],
            "",
            f'**Eligible graph path found:** {result["grounded"]}',
            "",
            "## Evidence passages",
        ]
        for evidence in result.get("evidence", []):
            ref = f'{evidence["source"]}#{evidence["chunk_id"]}'
            report_lines.extend([f'### {evidence.get("title") or evidence["source"]}', f"Reference: {ref}", evidence["text"], ""])
        report_lines.extend(["## Limitations", *[f"- {item}" for item in result.get("limitations", [])]])
        report_lines.extend(["", "## Evidence paths"])
        for path in result.get("reasoning_paths", []):
            report_lines.append(" -> ".join(path["entities"]))
            for step in path["steps"]:
                report_lines.append(
                    f'- {step["source"]} --{step["relation"]}--> {step["target"]}: '
                    f'{step["evidence"]} [{step["reference"]}]'
                )
        st.download_button(
            "Download evidence report (Markdown)",
            data="\n".join(report_lines),
            file_name="biomedical_evidence_report.md",
            mime="text/markdown",
        )

with tabs[3]:
    st.subheader("Compare answer configurations")
    st.caption("Each question uses the same retrieved chunks and selected local model. This is a demo comparison, not a measured research result until answers are reviewed against a labeled dataset.")
    experiment_questions = st.text_area(
        "Questions (one per line, up to 3; begin with one question on CPU)",
        "What relationship do the papers report between insulin resistance and type 2 diabetes?",
    )
    if st.button("Run side-by-side experiment"):
        questions = [line.strip() for line in experiment_questions.splitlines() if line.strip()]
        if not questions or len(questions) > 3:
            st.error("Enter between 1 and 3 questions.")
        else:
            with st.spinner("Running four configurations per question; this may take a few minutes on CPU"):
                experiment_result = post("/experiment", {"questions": questions, "model": MODEL}, timeout=900)
            if experiment_result:
                st.session_state["experiment_result"] = experiment_result
                st.session_state["experiment_labels"] = {}

    experiment_result = st.session_state.get("experiment_result")
    if experiment_result:
        export_rows = []
        for case_index, case in enumerate(experiment_result["cases"]):
            st.markdown(f'### Question {case_index + 1}: {case["question"]}')
            st.caption(
                "Shared retrieved corpus for RAG and GraphRAG: "
                + (", ".join(case.get("retrieved_sources", [])) or "no documents retrieved")
            )
            st.caption(
                f'Retrieval latency: {case.get("retrieval_latency_seconds", 0):.3f}s '
                "(measured once and shared by retrieval-based configurations)"
            )
            columns = st.columns(len(case["runs"]))
            for column, run in zip(columns, case["runs"]):
                label = run["configuration"].replace("_", " ").title()
                with column:
                    st.markdown(f"**{label}**")
                    st.caption(f'Latency: {run["latency_seconds"]:.3f}s · Model reply: {run["model_response_received"]}')
                    st.write(run["answer"])
                    if run.get("sources"):
                        st.caption("Sources: " + ", ".join(run["sources"]))
                    key = f"review_{case_index}_{run['configuration']}"
                    review = st.selectbox(
                        "Your evidence-support review",
                        ["Not reviewed", "Supported", "Partly supported", "Unsupported", "Uncertain"],
                        key=key,
                    )
                    st.session_state["experiment_labels"][key] = review
                    export_rows.append(
                        {
                            "question": case["question"],
                            "configuration": run["configuration"],
                            "answer": run["answer"],
                            "sources": "; ".join(run.get("sources", [])),
                            "shared_retrieved_sources": "; ".join(case.get("retrieved_sources", [])),
                            "shared_retrieval_latency_seconds": case.get("retrieval_latency_seconds", 0),
                            "latency_seconds": run["latency_seconds"],
                            "model_response_received": run["model_response_received"],
                            "review": review,
                        }
                    )
        if export_rows:
            st.subheader("Manual review counts")
            review_summary = pd.crosstab(
                pd.Series([row["configuration"] for row in export_rows], name="configuration"),
                pd.Series([row["review"] for row in export_rows], name="review"),
            )
            st.dataframe(review_summary, use_container_width=True)
            st.download_button(
                "Download experiment answers and reviews (CSV)",
                data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
                file_name="graphrag_experiment.csv",
                mime="text/csv",
            )
            st.caption("Reviews are your manual judgments; they are not ground truth unless assessed with a defined annotation protocol.")

with tabs[4]:
    st.code("\n".join(get("/logs").get("logs", [])))
