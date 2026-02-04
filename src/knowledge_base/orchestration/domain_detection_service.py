"""Domain detection service."""

import logging
from typing import Optional
from pathlib import Path
from knowledge_base.domain.ontology_snippets import DOMAIN_ONTOLOGIES
from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.persistence.v1.schema import Document


class DomainDetectionService(BaseService):
    """Service for detecting document domains."""

    DOMAIN_KEYWORDS = {
        "technology": [
            "software",
            "code",
            "api",
            "algorithm",
            "database",
            "server",
            "cloud",
            "programming",
            "developer",
            "framework",
            "library",
            "function",
            "class",
            "module",
            "interface",
            "protocol",
            "network",
            "system",
            "data",
            "machine learning",
            "ai",
            "neural",
            "model",
            "training",
            "inference",
        ],
        "healthcare": [
            "patient",
            "doctor",
            "hospital",
            "clinical",
            "diagnosis",
            "treatment",
            "therapy",
            "medication",
            "medical",
            "health",
            "disease",
            "symptom",
            "prescription",
            "surgery",
            "procedure",
            "lab",
            "test",
            "blood",
            "pressure",
            "heart",
            "cancer",
            "diabetes",
            "mental",
        ],
        "finance": [
            "finance",
            "bank",
            "investment",
            "stock",
            "market",
            "trading",
            "portfolio",
            "asset",
            "liability",
            "revenue",
            "profit",
            "loss",
            "equity",
            "bond",
            "loan",
            "credit",
            "debt",
            "cryptocurrency",
            "bitcoin",
            "dollar",
            "euro",
            "yen",
            "forex",
            "capital",
            "income",
            "expense",
            "budget",
            "accounting",
        ],
        "legal": [
            "law",
            "legal",
            "contract",
            "agreement",
            "court",
            "judge",
            "attorney",
            "lawyer",
            "litigation",
            "lawsuit",
            "regulation",
            "compliance",
            "policy",
            "clause",
            "term",
            "breach",
            "liability",
            "damages",
            "settlement",
            "verdict",
            "testimony",
            "evidence",
        ],
        "science": [
            "research",
            "experiment",
            "hypothesis",
            "theory",
            "analysis",
            "data",
            "study",
            "paper",
            "publication",
            "laboratory",
            "scientist",
            "physics",
            "chemistry",
            "biology",
            "molecule",
            "cell",
            "gene",
            "protein",
            "atom",
            "energy",
            "force",
            "quantum",
        ],
    }

    def __init__(self):
        super().__init__()

    async def initialize(self) -> None:
        """Initialize the service."""
        self._logger.info("DomainDetectionService initialized")

    async def shutdown(self) -> None:
        """Shutdown the service."""
        self._logger.info("DomainDetectionService shutdown")

    async def detect_domain(
        self, document: Document, content_text: str | None = None
    ) -> str:
        """Determine domain for a document based on its content or metadata.

        Uses keyword-based scoring to classify documents into domains.

        Args:
            document: The document to determine domain for.
            content_text: Optional document content text for content-based classification.

        Returns:
            Domain string (e.g., "TECHNOLOGY", "FINANCIAL", "MEDICAL", "LEGAL",
            "SCIENTIFIC", "GENERAL").
        """
        if document.doc_metadata and "domain" in document.doc_metadata:
            return str(document.doc_metadata["domain"]).upper()

        if content_text:
            scores = self._calculate_domain_scores(content_text)
            best_domain = max(scores, key=lambda k: scores.get(k, 0.0))
            if scores.get(best_domain, 0) >= 0.1:
                return best_domain.upper()

        name_lower = document.name.lower()
        if any(term in name_lower for term in ["tech", "software", "code", "api"]):
            return "TECHNOLOGY"
        elif any(
            term in name_lower for term in ["health", "medical", "patient", "doctor"]
        ):
            return "MEDICAL"
        elif any(
            term in name_lower for term in ["finance", "bank", "money", "investment"]
        ):
            return "FINANCIAL"
        elif any(term in name_lower for term in ["legal", "law", "contract", "court"]):
            return "LEGAL"
        elif any(
            term in name_lower
            for term in ["research", "science", "study", "experiment"]
        ):
            return "SCIENTIFIC"
        else:
            return "GENERAL"

    def _calculate_domain_scores(self, text: str) -> dict[str, float]:
        """Calculate domain scores based on keyword frequency."""
        if not text or not text.strip():
            return {
                "technology": 0.0,
                "healthcare": 0.0,
                "finance": 0.0,
                "legal": 0.0,
                "science": 0.0,
                "general": 0.0,
            }

        text_lower = text.lower()
        scores = {
            "technology": 0.0,
            "healthcare": 0.0,
            "finance": 0.0,
            "legal": 0.0,
            "science": 0.0,
            "general": 0.0,
        }

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                count = text_lower.count(keyword)
                weight = len(keyword) / 10
                scores[domain] += count * weight

        total = sum(scores.values())
        if total > 0:
            for domain in scores:
                scores[domain] /= total

        return scores
