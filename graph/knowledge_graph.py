from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

from api.schemas import GraphTriple

logger = logging.getLogger(__name__)


class KnowledgeGraphBuilderAgent:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_triples(self, triples: list[GraphTriple]) -> int:
        count = 0
        for triple in triples:
            self.graph.add_node(triple.subject, label=triple.subject, type="biomedical_concept")
            self.graph.add_node(triple.object, label=triple.object, type="biomedical_concept")
            self.graph.add_edge(
                triple.subject,
                triple.object,
                key=f"{triple.predicate}:{triple.chunk_id}",
                predicate=triple.predicate,
                source=triple.source,
                chunk_id=triple.chunk_id,
                evidence=triple.evidence,
                confidence=triple.confidence,
                polarity=triple.polarity,
                qualifiers=triple.qualifiers,
            )
            count += 1
        logger.info("Added %s triples; graph now has %s nodes and %s edges", count, self.graph.number_of_nodes(), self.graph.number_of_edges())
        return count

    def clear(self) -> None:
        self.graph.clear()

    def to_dict(self) -> dict[str, Any]:
        nodes = [{"id": node, **data} for node, data in self.graph.nodes(data=True)]
        edges = []
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            edge_data = dict(data)
            reference_source = edge_data.pop("source", None)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "key": key,
                    "reference_source": reference_source,
                    **edge_data,
                }
            )
        return {"nodes": nodes, "edges": edges}

    def save_json(self, path: str = "data/knowledge_graph.json") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load_json(self, path: str = "data/knowledge_graph.json") -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.graph.clear()
        for node in payload.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        for edge in payload.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            key = edge.pop("key")
            self.graph.add_edge(source, target, key=key, **edge)
