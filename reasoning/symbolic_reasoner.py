from __future__ import annotations

import logging
from collections import deque
from uuid import uuid4

import networkx as nx

from api.schemas import PathStep, ReasoningPath

logger = logging.getLogger(__name__)


class SymbolicReasonerAgent:
    """Deterministic traversal over explicit, source-backed biomedical edges."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph

    def reason(self, query: str, candidate_entities: list[str], traversal: str = "bfs", max_hops: int = 3) -> list[ReasoningPath]:
        anchors = self._match_entities(query, candidate_entities)
        if not anchors:
            anchors = self._match_entities(query, list(self.graph.nodes))
        paths: list[ReasoningPath] = []
        for anchor in anchors[:5]:
            if traversal == "dfs":
                paths.extend(self._dfs(anchor, max_hops=max_hops))
            elif traversal == "shortest":
                paths.extend(self._shortest_paths(anchor, anchors, max_hops=max_hops))
            else:
                paths.extend(self._bfs(anchor, max_hops=max_hops))
        validated = [path for path in paths if self._validate_path(path)]
        validated.sort(key=lambda item: item.score, reverse=True)
        logger.info("Validated %s reasoning paths from anchors=%s", len(validated), anchors)
        return validated[:8]

    def _bfs(self, start: str, max_hops: int) -> list[ReasoningPath]:
        paths: list[ReasoningPath] = []
        queue = deque([(start, [], {start})])
        while queue:
            node, steps, visited = queue.popleft()
            if 0 < len(steps) <= max_hops:
                paths.append(self._path_from_steps(steps))
            if len(steps) == max_hops:
                continue
            for target, step in self._supported_neighbors(node):
                if target not in visited:
                    queue.append((target, steps + [step], visited | {target}))
        return paths

    def _dfs(self, start: str, max_hops: int) -> list[ReasoningPath]:
        paths: list[ReasoningPath] = []

        def visit(node: str, steps: list[PathStep], visited: set[str]) -> None:
            if 0 < len(steps) <= max_hops:
                paths.append(self._path_from_steps(steps))
            if len(steps) == max_hops:
                return
            for target, step in self._supported_neighbors(node):
                if target not in visited:
                    visit(target, steps + [step], visited | {target})

        visit(start, [], {start})
        return paths

    def _shortest_paths(self, start: str, anchors: list[str], max_hops: int) -> list[ReasoningPath]:
        paths: list[ReasoningPath] = []
        simple_graph = nx.DiGraph()
        for source, target, data in self.graph.edges(data=True):
            if self._edge_supported(data):
                simple_graph.add_edge(source, target)
        for target in anchors:
            if target == start:
                continue
            try:
                nodes = nx.shortest_path(simple_graph, start, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(nodes) - 1 > max_hops:
                continue
            steps: list[PathStep] = []
            for source, dest in zip(nodes, nodes[1:]):
                edge_data = next(
                    data
                    for data in self.graph.get_edge_data(source, dest).values()
                    if self._edge_supported(data)
                )
                steps.append(self._step(source, dest, edge_data))
            paths.append(self._path_from_steps(steps))
        return paths

    def _validate_path(self, path: ReasoningPath) -> bool:
        if not path.steps:
            return False
        if not all(step.evidence.strip() and step.reference.strip() and step.polarity == "positive" for step in path.steps):
            return False
        if not all(left.target == right.source for left, right in zip(path.steps, path.steps[1:])):
            return False
        return not any(self._has_conflicting_polarity(step) for step in path.steps)

    def _has_conflicting_polarity(self, step: PathStep) -> bool:
        inverse = {
            "treated_by": "treats",
            "has_cause": "causes",
            "regulated_by": "regulates",
            "affected_by": "affects",
            "associated_with": "associated_with",
            "related_to": "related_to",
            "correlated_with": "correlated_with",
            "linked_to": "linked_to",
        }
        source, target = step.source, step.target
        predicate = inverse.get(step.relation, step.relation)
        if step.relation in inverse:
            source, target = target, source
        if not self.graph.has_edge(source, target):
            return False
        return any(
            data.get("predicate") == predicate and data.get("polarity", "positive") != "positive"
            for data in self.graph.get_edge_data(source, target).values()
        )

    def _path_from_steps(self, steps: list[PathStep]) -> ReasoningPath:
        entities = [steps[0].source] + [step.target for step in steps]
        score = sum(1.0 for step in steps if step.evidence) / max(len(steps), 1)
        score += min(len(steps), 3) * 0.05
        return ReasoningPath(path_id=str(uuid4()), entities=entities, steps=steps, score=round(score, 3))

    @staticmethod
    def _edge_supported(data: dict) -> bool:
        return bool(
            data.get("source")
            and data.get("chunk_id")
            and data.get("evidence")
            and data.get("polarity", "positive") == "positive"
        )

    def _supported_neighbors(self, node: str) -> list[tuple[str, PathStep]]:
        neighbors: list[tuple[str, PathStep]] = []
        for _, target, data in self.graph.out_edges(node, data=True):
            if self._edge_supported(data):
                neighbors.append((target, self._step(node, target, data)))
        for source, _, data in self.graph.in_edges(node, data=True):
            if not self._edge_supported(data):
                continue
            predicate = str(data.get("predicate", "related_to"))
            if predicate == "treats":
                relation = "treated_by"
            elif predicate == "causes":
                relation = "has_cause"
            elif predicate == "regulates":
                relation = "regulated_by"
            elif predicate == "affects":
                relation = "affected_by"
            elif predicate in {"associated_with", "related_to", "correlated_with", "linked_to"}:
                relation = predicate
            else:
                continue
            neighbors.append((source, self._step(node, source, {**data, "predicate": relation})))
        return neighbors

    @staticmethod
    def _step(source: str, target: str, data: dict) -> PathStep:
        return PathStep(
            source=source,
            relation=str(data.get("predicate", "related_to")),
            target=target,
            evidence=str(data.get("evidence", "")),
            reference=f"{data.get('source')}#{data.get('chunk_id')}",
            polarity=str(data.get("polarity", "positive")),
        )

    def _match_entities(self, query: str, candidates: list[str]) -> list[str]:
        query_lower = query.lower()
        matches = [node for node in self.graph.nodes if node.lower() in query_lower]
        for candidate in candidates:
            if candidate in self.graph and candidate not in matches:
                cand_tokens = set(candidate.lower().split())
                if cand_tokens and cand_tokens.intersection(query_lower.split()):
                    matches.append(candidate)
        return matches
