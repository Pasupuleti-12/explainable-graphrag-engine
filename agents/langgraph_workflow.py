from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    query: str
    chunks: list
    triples: list
    paths: list
    answer: str


def build_langgraph_workflow(orchestrator):
    """Optional LangGraph orchestration wrapper for research demos."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(WorkflowState)

    def retrieve(state: WorkflowState) -> WorkflowState:
        state["chunks"] = orchestrator.vector_store.query(state["query"], n_results=10)
        return state

    def extract(state: WorkflowState) -> WorkflowState:
        state["triples"] = orchestrator.extractor.extract(state["chunks"])
        return state

    def build_graph(state: WorkflowState) -> WorkflowState:
        orchestrator.graph_builder.add_triples(state["triples"])
        return state

    def reason(state: WorkflowState) -> WorkflowState:
        from reasoning.symbolic_reasoner import SymbolicReasonerAgent

        reasoner = SymbolicReasonerAgent(orchestrator.graph_builder.graph)
        state["paths"] = reasoner.reason(state["query"], [])
        return state

    def generate(state: WorkflowState) -> WorkflowState:
        state["answer"] = orchestrator.generator.generate(state["query"], state["paths"])
        return state

    graph.add_node("retrieve", retrieve)
    graph.add_node("extract", extract)
    graph.add_node("build_graph", build_graph)
    graph.add_node("reason", reason)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "extract")
    graph.add_edge("extract", "build_graph")
    graph.add_edge("build_graph", "reason")
    graph.add_edge("reason", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
