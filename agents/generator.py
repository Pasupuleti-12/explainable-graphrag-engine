from __future__ import annotations

import logging

import requests

from api.schemas import DocumentChunk, GraphTriple, ReasoningPath

logger = logging.getLogger(__name__)


class GeneratorAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self.ollama_url = ollama_url.rstrip("/")

    def generate(self, query: str, paths: list[ReasoningPath], model: str = "llama3") -> str:
        if not paths:
            return (
                "I cannot provide a grounded answer because the knowledge graph did not contain a validated, "
                "source-backed reasoning path for this query."
            )
        evidence = self._evidence_block(paths)
        references = {step.reference for path in paths for step in path.steps}
        prompt = (
            "You are a cautious biomedical literature research assistant, not a clinician. Answer only with claims "
            "directly supported by the evidence below. Do not diagnose, prescribe, or give personalized treatment advice. "
            "Preserve negation and uncertainty. Cite every factual sentence with an exact source reference in square "
            "brackets, copied from the evidence. If the evidence is incomplete or conflicting, say so and abstain "
            "from a definitive conclusion.\n\n"
            f"Question: {query}\n\nValidated graph evidence:\n{evidence}\n\n"
            "Return: 1) concise answer, 2) reasoning chain, 3) evidence limitations."
        )
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=60,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if text and self._has_valid_citations(text, references):
                return text
            if text:
                logger.warning("Ollama response lacked valid source citations; returning evidence template")
        except Exception as exc:
            logger.warning("Ollama generation unavailable; using deterministic template: %s", exc)
        return self._template_answer(query, paths)

    def generate_llm_only(self, query: str, model: str = "llama3") -> str:
        prompt = (
            "Answer this biomedical literature question as a research assistant. Do not diagnose or prescribe, "
            "and state that the answer has not been checked against retrieved evidence.\n\nQuestion: " + query
        )
        return self._ollama_or_fallback(prompt, model, "LLM-only baseline unavailable; no answer generated.")

    def generate_from_chunks(self, query: str, chunks: list[DocumentChunk], model: str = "llama3") -> str:
        if not chunks:
            return "I could not retrieve supporting documents for this question."
        references = []
        rows = []
        for chunk in chunks[:10]:
            reference = f"{chunk.source}#{chunk.chunk_id}"
            references.append(reference)
            rows.append(f"[{reference}] {chunk.text[:1800]}")
        prompt = (
            "Answer the biomedical literature question using only the retrieved passages. Do not diagnose or prescribe. "
            "Cite each factual sentence with the exact bracketed source reference. If the passages do not answer it, "
            "say so.\n\nQuestion: " + query + "\n\nRetrieved passages:\n" + "\n\n".join(rows)
        )
        answer = self._ollama_or_fallback(prompt, model, "")
        if answer and self._has_valid_citations(answer, set(references)):
            return answer
        excerpts = [f"- {chunk.text[:400]} [{chunk.source}#{chunk.chunk_id}]" for chunk in chunks[:3]]
        return "Retrieved passages (not synthesized because a cited answer was unavailable):\n" + "\n".join(excerpts)

    def generate_from_relationships(
        self, query: str, triples: list[GraphTriple], model: str = "llama3"
    ) -> str:
        if not triples:
            return "No extracted graph relationships were available for this question."
        rows = []
        references = set()
        for triple in triples[:30]:
            reference = f"{triple.source}#{triple.chunk_id}"
            references.add(reference)
            rows.append(
                f"({triple.subject}) --{triple.predicate} [{triple.polarity}]--> ({triple.object}); "
                f"evidence: {triple.evidence[:500]} [{reference}]"
            )
        prompt = (
            "Answer this biomedical literature question using the extracted relationships and their evidence. "
            "These relationships have not been checked by deterministic path validation. Do not diagnose or prescribe. "
            "Cite every factual sentence with an exact source reference in square brackets, and report uncertainty or "
            "contradictions when visible.\n\nQuestion: " + query + "\n\nExtracted relationships:\n" + "\n".join(rows)
        )
        answer = self._ollama_or_fallback(prompt, model, "")
        if answer and self._has_valid_citations(answer, references):
            return answer
        excerpts = [f"- {row}" for row in rows[:5]]
        return "Extracted graph evidence (not synthesized because a cited answer was unavailable):\n" + "\n".join(excerpts)

    def _ollama_or_fallback(self, prompt: str, model: str, fallback: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=60,
            )
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("Ollama baseline generation unavailable: %s", exc)
        return fallback

    def _template_answer(self, query: str, paths: list[ReasoningPath]) -> str:
        lines = [f"Grounded answer for: {query}", "", "Validated reasoning chain:"]
        for path in paths[:3]:
            chain = " -> ".join(path.entities)
            lines.append(f"- {chain}")
            for step in path.steps:
                lines.append(f"  Evidence: {step.source} {step.relation} {step.target} [{step.reference}]")
        lines.append("")
        lines.append("Limitation: this answer is restricted to validated graph paths and should not be treated as clinical advice.")
        return "\n".join(lines)

    @staticmethod
    def _evidence_block(paths: list[ReasoningPath]) -> str:
        rows: list[str] = []
        for idx, path in enumerate(paths[:5], start=1):
            rows.append(f"Path {idx}: {' -> '.join(path.entities)}")
            for step in path.steps:
                rows.append(f"- {step.source} --{step.relation}--> {step.target}")
                rows.append(f"  Source: {step.reference}")
                rows.append(f"  Evidence: {step.evidence[:350]}")
        return "\n".join(rows)

    @staticmethod
    def _has_valid_citations(text: str, references: set[str]) -> bool:
        cited = {ref for ref in references if f"[{ref}]" in text}
        factual_sentences = [
            sentence.strip()
            for sentence in text.replace("?", ".").replace("!", ".").split(".")
            if len(sentence.split()) >= 5
        ]
        return bool(cited) and bool(factual_sentences) and all(
            any(f"[{ref}]" in sentence for ref in references) for sentence in factual_sentences
        )
