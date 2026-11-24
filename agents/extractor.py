from __future__ import annotations

import logging
import re
from collections import Counter

import requests

from api.schemas import DocumentChunk, GraphTriple

logger = logging.getLogger(__name__)

MEDICAL_RELATIONS = {
    "affects": ["affects", "impairs", "alters", "modulates", "influences"],
    "causes": ["causes", "induces", "leads to", "contributes to", "results in"],
    "associated_with": ["associated with", "linked to", "correlated with", "related to"],
    "treats": [
        "treats",
        "treatment for",
        "therapy for",
        "used to treat",
        "recommended for",
        "first-line",
        "reduces",
        "improves",
        "prevents",
        "inhibits",
        "managed with",
    ],
    "regulates": ["regulates", "controls", "mediates", "activates", "suppresses"],
    "expressed_in": ["expressed in", "found in", "observed in"],
}

ENTITY_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z]?[A-Za-z0-9\-]+){0,3}|"
    r"type\s+[12]\s+diabetes|blood glucose|insulin resistance|cardiovascular disease)\b",
    re.IGNORECASE,
)

BIOMEDICAL_TERMS = [
    "type 2 diabetes",
    "type 1 diabetes",
    "diabetes mellitus",
    "non-insulin-dependent diabetes mellitus",
    "diabetes",
    "insulin resistance",
    "insulin secretion",
    "impaired insulin secretion",
    "insulin production",
    "insulin therapy",
    "insulin treatment",
    "insulin administration",
    "insulin",
    "metformin",
    "sulfonylureas",
    "glp-1 receptor agonists",
    "sglt2 inhibitors",
    "dpp-4 inhibitors",
    "thiazolidinediones",
    "lifestyle intervention",
    "diet",
    "exercise",
    "glycemic control",
    "diabetes treatment",
    "diabetes management",
    "blood glucose regulation",
    "blood glucose",
    "plasma glucose",
    "glucose level",
    "glucose monitoring",
    "glucose",
    "fasting hyperglycemia",
    "hyperglycemia",
    "hypoglycemia",
    "hepatic glucose production",
    "increased hepatic glucose production",
    "pancreatic beta-cell responsiveness",
    "beta-cell responsiveness",
    "beta-cell function",
    "cardiovascular disease",
    "kidney disease",
    "hypertension",
    "inflammation",
]


class ExtractorAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434") -> None:
        self.ollama_url = ollama_url.rstrip("/")
        try:
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
            logger.info("spaCy model unavailable; using deterministic biomedical phrase extraction fallback")

    def extract(self, chunks: list[DocumentChunk], use_llm: bool = False, model: str = "llama3") -> list[GraphTriple]:
        triples: list[GraphTriple] = []
        for chunk in chunks:
            triples.extend(self._rule_triples(chunk))
            if use_llm:
                triples.extend(self._llm_triples(chunk, model=model))
        return self._deduplicate(triples)

    def entities(self, text: str) -> list[str]:
        entities: list[str] = []
        lowered = text.lower()
        for term in BIOMEDICAL_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                entities.append(term)
        if self.nlp:
            doc = self.nlp(text)
            entities.extend(ent.text.strip() for ent in doc.ents if ent.label_ in {"DISEASE", "ORG", "PRODUCT", "GPE"})
            entities.extend(chunk.text.strip() for chunk in doc.noun_chunks if 2 <= len(chunk.text) <= 60)
        entities.extend(match.group(1).strip() for match in ENTITY_PATTERN.finditer(text))
        cleaned = [self._normalize_entity(entity) for entity in entities if self._valid_entity(entity)]
        counts = Counter(cleaned)
        return [entity for entity, _ in counts.most_common(20)]

    def _rule_triples(self, chunk: DocumentChunk) -> list[GraphTriple]:
        triples: list[GraphTriple] = []
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            lowered = sentence.lower()
            for predicate, triggers in MEDICAL_RELATIONS.items():
                trigger = next((item for item in triggers if item in lowered), None)
                if not trigger:
                    continue
                before, after = lowered.split(trigger, 1)
                subject_candidates = self.entities(before[-160:])
                object_candidates = self.entities(after[:180])
                if subject_candidates and object_candidates:
                    subject = self._best_subject(subject_candidates)
                    object_ = self._best_object(object_candidates)
                    predicate = self._normalize_predicate(predicate, subject, object_)
                    polarity = self._statement_polarity(sentence, trigger)
                    triples.append(
                        GraphTriple(
                            subject=subject,
                            predicate=predicate,
                            object=object_,
                            source=chunk.source,
                            chunk_id=chunk.chunk_id,
                            evidence=sentence[:500],
                            confidence=0.82,
                            polarity=polarity,
                            qualifiers=["rule_extracted"],
                        )
                    )
        return triples

    @staticmethod
    def _statement_polarity(sentence: str, trigger: str) -> str:
        """Conservatively label simple negation and uncertainty near a relation trigger."""
        lowered = sentence.lower()
        prefix = lowered.split(trigger, 1)[0][-100:]
        if re.search(r"\b(no|not|never|neither|failed to|did not|does not|was not|were not)\b", prefix):
            return "negative"
        if re.search(r"\b(may|might|could|possibly|potentially|suggests?|appears? to|uncertain)\b", lowered):
            return "uncertain"
        return "positive"

    def _llm_triples(self, chunk: DocumentChunk, model: str) -> list[GraphTriple]:
        prompt = (
            "Extract biomedical graph triples from this text. Return one triple per line as "
            "subject | predicate | object. Use only explicit relationships in the text.\n\n"
            f"TEXT:\n{chunk.text[:1800]}"
        )
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=45,
            )
            response.raise_for_status()
            raw = response.json().get("response", "")
        except Exception as exc:
            logger.warning("Ollama extraction skipped: %s", exc)
            return []
        triples: list[GraphTriple] = []
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) == 3 and all(parts):
                triples.append(
                    GraphTriple(
                        subject=self._normalize_entity(parts[0]),
                        predicate=self._normalize_relation(parts[1]),
                        object=self._normalize_entity(parts[2]),
                        source=chunk.source,
                        chunk_id=chunk.chunk_id,
                        evidence=chunk.text[:500],
                        confidence=0.7,
                        polarity="uncertain",
                        qualifiers=["llm_extracted", "polarity_unverified"],
                    )
                )
        return triples

    def _deduplicate(self, triples: list[GraphTriple]) -> list[GraphTriple]:
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[GraphTriple] = []
        for triple in triples:
            key = (triple.subject.lower(), triple.predicate.lower(), triple.object.lower(), triple.chunk_id)
            if key not in seen and triple.subject.lower() != triple.object.lower():
                seen.add(key)
                unique.append(triple)
        return unique

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        entity = re.sub(r"\s+", " ", entity.strip(" .,:;()[]"))
        entity = re.sub(r"\b(is|are|was|were|closely|related|associated|degree|given patient|such findings)\b", "", entity, flags=re.I)
        entity = re.sub(r"\s+", " ", entity).strip(" .,:;()[]")
        replacements = {
            "Non-Insulin-Dependent Diabetes Mellitus": "Type 2 Diabetes",
            "Diabetes Mellitus": "Diabetes",
            "Plasma Glucose": "Blood Glucose",
            "Glucose Level": "Blood Glucose",
            "Increased Hepatic Glucose Production": "Hepatic Glucose Production",
            "Insulin Treatment": "Insulin Therapy",
            "Insulin Administration": "Insulin Therapy",
            "Glp-1 Receptor Agonists": "GLP-1 Receptor Agonists",
            "Sglt2 Inhibitors": "SGLT2 Inhibitors",
            "Dpp-4 Inhibitors": "DPP-4 Inhibitors",
        }
        titled = entity.title()
        return replacements.get(titled, titled)

    @staticmethod
    def _normalize_relation(relation: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", relation.strip().lower()).strip("_") or "related_to"

    @staticmethod
    def _valid_entity(entity: str) -> bool:
        text = entity.strip()
        if len(text) < 3 or len(text) > 80:
            return False
        blocked = {"the", "a", "an", "such", "several", "recent", "patients", "patient"}
        return text.lower() not in blocked

    @staticmethod
    def _best_subject(candidates: list[str]) -> str:
        priority = [
            "Metformin",
            "Insulin Therapy",
            "Insulin",
            "GLP-1 Receptor Agonists",
            "SGLT2 Inhibitors",
            "DPP-4 Inhibitors",
            "Lifestyle Intervention",
            "Diet",
            "Exercise",
            "Diabetes",
            "Type 2 Diabetes",
            "Insulin Secretion",
            "Impaired Insulin Secretion",
        ]
        for item in priority:
            if item in candidates:
                return item
        return candidates[-1]

    @staticmethod
    def _best_object(candidates: list[str]) -> str:
        priority = ["Type 2 Diabetes", "Diabetes", "Blood Glucose", "Hepatic Glucose Production", "Insulin Secretion", "Insulin Therapy", "Glucose"]
        for item in priority:
            if item in candidates:
                return item
        return candidates[0]

    @staticmethod
    def _normalize_predicate(predicate: str, subject: str, object_: str) -> str:
        if subject in {"Insulin Therapy", "Insulin"} and object_ in {"Diabetes", "Type 2 Diabetes"}:
            return "treats"
        if "glucose" in object_.lower() and predicate in {"affects", "causes", "regulates"}:
            return predicate
        return predicate
