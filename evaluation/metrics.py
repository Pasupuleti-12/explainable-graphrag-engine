from __future__ import annotations

from api.schemas import ReasonResponse


class EvaluationAgent:
    """Report observable pipeline facts separately from metrics requiring human labels."""

    def evaluate(self, query: str, graph_response: ReasonResponse) -> dict[str, float]:
        paths = graph_response.reasoning_paths
        step_count = sum(len(path.steps) for path in paths)
        source_count = len(graph_response.sources)
        return {
            "has_reasoning_path": float(bool(paths)),
            "reasoning_path_count": float(len(paths)),
            "evidence_step_count": float(step_count),
            "distinct_source_count": float(source_count),
            "abstained": float(not graph_response.grounded),
            "answer_query_token_overlap_proxy": round(self._token_overlap(query, graph_response.answer), 3),
        }

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        left = {token.lower().strip(".,?!:;()[]") for token in a.split() if len(token.strip(".,?!:;()[]")) > 3}
        right = {token.lower().strip(".,?!:;()[]") for token in b.split() if len(token.strip(".,?!:;()[]")) > 3}
        if not left:
            return 0.0
        return len(left.intersection(right)) / len(left)
