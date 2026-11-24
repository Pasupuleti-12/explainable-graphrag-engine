from api.schemas import GraphTriple
from graph.knowledge_graph import KnowledgeGraphBuilderAgent
from reasoning.symbolic_reasoner import SymbolicReasonerAgent


def test_reasoner_returns_only_source_backed_paths():
    builder = KnowledgeGraphBuilderAgent()
    builder.add_triples(
        [
            GraphTriple(
                subject="Diabetes",
                predicate="affects",
                object="Insulin Production",
                source="test",
                chunk_id="chunk1",
                evidence="Diabetes affects insulin production.",
            ),
            GraphTriple(
                subject="Insulin Production",
                predicate="regulates",
                object="Blood Glucose",
                source="test",
                chunk_id="chunk2",
                evidence="Insulin production regulates blood glucose.",
            ),
        ]
    )
    reasoner = SymbolicReasonerAgent(builder.graph)
    paths = reasoner.reason("How does diabetes affect blood glucose?", ["Diabetes"], max_hops=2)
    assert paths
    assert any(path.entities == ["Diabetes", "Insulin Production", "Blood Glucose"] for path in paths)
    assert all(step.reference for path in paths for step in path.steps)


def test_reasoner_exposes_incoming_treatment_edges():
    builder = KnowledgeGraphBuilderAgent()
    builder.add_triples(
        [
            GraphTriple(
                subject="Metformin",
                predicate="treats",
                object="Type 2 Diabetes",
                source="test",
                chunk_id="chunk3",
                evidence="Metformin is used to treat type 2 diabetes.",
            )
        ]
    )
    reasoner = SymbolicReasonerAgent(builder.graph)
    paths = reasoner.reason("What drugs treat type 2 diabetes?", ["Type 2 Diabetes"], max_hops=1)
    assert paths
    assert paths[0].steps[0].source == "Type 2 Diabetes"
    assert paths[0].steps[0].relation == "treated_by"
    assert paths[0].steps[0].target == "Metformin"
