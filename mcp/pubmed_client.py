from __future__ import annotations

import logging
import urllib.parse
import xml.etree.ElementTree as ET

import requests

from api.schemas import DocumentChunk
from utils.document_processing import DocumentProcessor

logger = logging.getLogger(__name__)


class PubMedClient:
    """Free PubMed API adapter with the same role an MCP server would play locally."""

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, processor: DocumentProcessor | None = None) -> None:
        self.processor = processor or DocumentProcessor()

    def search(self, query: str, max_results: int = 5) -> list[str]:
        params = urllib.parse.urlencode(
            {"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results, "sort": "relevance"}
        )
        response = requests.get(f"{self.base_url}/esearch.fcgi?{params}", timeout=20)
        response.raise_for_status()
        return response.json().get("esearchresult", {}).get("idlist", [])

    def fetch_abstracts(self, pmids: list[str]) -> list[dict]:
        if not pmids:
            return []
        params = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"})
        response = requests.get(f"{self.base_url}/efetch.fcgi?{params}", timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        articles: list[dict] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or "unknown"
            title = article.findtext(".//ArticleTitle") or "Untitled"
            abstract_parts = [node.text or "" for node in article.findall(".//AbstractText")]
            abstract = " ".join(part.strip() for part in abstract_parts if part.strip())
            journal = article.findtext(".//Journal/Title") or "PubMed"
            if abstract:
                articles.append({"pmid": pmid, "title": title, "abstract": abstract, "journal": journal})
        return articles

    def ingest(self, query: str, max_results: int = 5) -> list[DocumentChunk]:
        pmids = self.search(query, max_results=max_results)
        articles = self.fetch_abstracts(pmids)
        chunks: list[DocumentChunk] = []
        for article in articles:
            source = f"pubmed:{article['pmid']}"
            metadata = {
                "source_type": "pubmed_article",
                "pmid": article["pmid"],
                "title": article["title"],
                "journal": article["journal"],
            }
            text = f"{article['title']}. {article['abstract']}"
            chunks.extend(self.processor.chunks_from_text(text, source=source, metadata=metadata))
        logger.info("Ingested %s PubMed chunks for query=%s", len(chunks), query)
        return chunks
