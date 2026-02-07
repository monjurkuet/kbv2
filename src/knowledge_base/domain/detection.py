"""Domain detection module for KBV2."""

import re
import time
from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from knowledge_base.domain.domain_models import (
    DomainPrediction,
    DomainDetectionResult,
    DomainConfig,
)
from knowledge_base.domain.ontology_snippets import DOMAIN_ONTOLOGIES


class DomainDetector:
    """Automatic domain detection using keyword screening + LLM analysis."""

    def __init__(
        self,
        llm_client: Any = None,
        ontology_snippets: Optional[Dict[str, Dict]] = None,
        config: Optional[DomainConfig] = None,
    ):
        """Initialize the domain detector.

        Args:
            llm_client: Optional LLM client for deep analysis.
            ontology_snippets: Optional ontology snippets to use.
            config: Optional configuration for detection.
        """
        self.llm = llm_client
        self.ontology = ontology_snippets or DOMAIN_ONTOLOGIES
        self.config = config or DomainConfig()

    async def detect_domain(
        self, document_text: str, top_k: int = 3
    ) -> DomainDetectionResult:
        """Detect domain(s) of a document.

        Args:
            document_text: The text content to analyze.
            top_k: Maximum number of predictions to return.

        Returns:
            DomainDetectionResult with primary domain and all predictions.
        """
        start_time = time.time()

        if not document_text or not document_text.strip():
            return DomainDetectionResult(
                primary_domain="GENERAL",
                all_domains=[
                    DomainPrediction(
                        domain="GENERAL",
                        confidence=1.0,
                        reasoning="Empty or whitespace-only document",
                    )
                ],
                is_multi_domain=False,
                detection_method="fallback",
                processing_time_ms=0.0,
                confidence_threshold=self.config.min_confidence,
            )

        keyword_predictions = []
        if self.config.enable_keyword_screening:
            keyword_predictions = await self._keyword_screening(document_text)

        llm_predictions = []
        if self.config.enable_llm_analysis and self.llm:
            candidates = [p.domain for p in keyword_predictions[:3]]
            if not candidates:
                candidates = list(self.ontology.keys())[:3]
            llm_predictions = await self._llm_analysis(document_text, candidates)

        all_predictions = self._calibrate_confidence(
            keyword_predictions, llm_predictions
        )

        if not all_predictions:
            all_predictions = [
                DomainPrediction(
                    domain="GENERAL",
                    confidence=0.5,
                    reasoning="No domain detected from analysis",
                )
            ]

        primary = all_predictions[0]

        is_multi = len(all_predictions) > 1 and (
            all_predictions[1].confidence > primary.confidence * 0.7
        )

        return DomainDetectionResult(
            primary_domain=primary.domain,
            all_domains=all_predictions[:top_k],
            is_multi_domain=is_multi,
            detection_method="hybrid"
            if keyword_predictions and llm_predictions
            else (
                "keyword"
                if keyword_predictions
                else ("llm" if llm_predictions else "fallback")
            ),
            processing_time_ms=(time.time() - start_time) * 1000,
            confidence_threshold=self.config.min_confidence,
        )

    async def _keyword_screening(
        self, text: str, max_domains: int = 5
    ) -> List[DomainPrediction]:
        """Perform keyword-based screening to identify potential domains.

        Args:
            text: Document text to analyze.
            max_domains: Maximum number of domains to return.

        Returns:
            List of DomainPrediction objects sorted by confidence.
        """
        text_lower = text.lower()
        domain_scores: Dict[str, float] = {}
        domain_evidence: Dict[str, List[str]] = {}

        for domain, ontology in self.ontology.items():
            score = 0.0
            evidence = []
            keywords = ontology.get("keywords", [])

            for keyword in keywords:
                pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
                matches = list(re.finditer(pattern, text_lower))
                count = len(matches)

                if count > 0:
                    contribution = min(count * 0.1, 0.5)
                    score += contribution
                    evidence.append(f"{keyword}: {count}")

            if score > 0:
                domain_scores[domain] = score
                domain_evidence[domain] = evidence

        if not domain_scores:
            return []

        max_score = max(domain_scores.values())
        if max_score > 0:
            domain_scores = {k: v / max_score for k, v in domain_scores.items()}

        predictions = []
        for domain, score in sorted(domain_scores.items(), key=lambda x: -x[1])[
            :max_domains
        ]:
            predictions.append(
                DomainPrediction(
                    domain=domain,
                    confidence=min(score, 1.0),
                    supporting_evidence=domain_evidence.get(domain, []),
                    keyword_matches=domain_evidence.get(domain, []),
                    reasoning=f"Keyword analysis: {len(domain_evidence.get(domain, []))} matches",
                )
            )

        return predictions

    async def _llm_analysis(
        self, text: str, candidates: List[str]
    ) -> List[DomainPrediction]:
        """Perform LLM-based analysis of document domain.

        Args:
            text: Document text to analyze.
            candidates: List of candidate domains to consider.

        Returns:
            List of DomainPrediction from LLM analysis.
        """
        if not self.llm:
            return []

        available_candidates = [
            domain for domain in candidates if domain in self.ontology
        ]
        if not available_candidates:
            available_candidates = list(self.ontology.keys())[:3]

        ontology_info = "\n".join(
            [
                f"- {domain}: {info.get('description', '')}"
                for domain, info in self.ontology.items()
                if domain in available_candidates
            ]
        )

        prompt = f"""Analyze the following text and determine its domain.

Available domains:
{ontology_info}

Text to analyze (first 1000 chars):
{text[:1000]}

Respond with JSON with keys: domain (string), confidence (0.0-1.0), reasoning (string)"""

        try:
            response = await self.llm.complete(prompt)

            import json

            result = json.loads(response)

            return [
                DomainPrediction(
                    domain=result.get("domain", "GENERAL"),
                    confidence=min(max(result.get("confidence", 0.5), 0.0), 1.0),
                    supporting_evidence=[],
                    reasoning=result.get("reasoning", "LLM analysis completed"),
                )
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return []

    def _calibrate_confidence(
        self, keyword_scores: List[DomainPrediction], llm_scores: List[DomainPrediction]
    ) -> List[DomainPrediction]:
        """Combine keyword and LLM predictions with calibration.

        Args:
            keyword_scores: Predictions from keyword analysis.
            llm_scores: Predictions from LLM analysis.

        Returns:
            Combined and calibrated predictions sorted by confidence.
        """
        combined: Dict[str, DomainPrediction] = {}

        for pred in keyword_scores:
            combined[pred.domain] = pred

        for llm_pred in llm_scores:
            if llm_pred.domain in combined:
                existing = combined[llm_pred.domain]
                combined[llm_pred.domain] = DomainPrediction(
                    domain=llm_pred.domain,
                    confidence=(existing.confidence * 0.4 + llm_pred.confidence * 0.6),
                    supporting_evidence=existing.supporting_evidence,
                    keyword_matches=existing.keyword_matches,
                    reasoning=f"{existing.reasoning}; {llm_pred.reasoning}",
                )
            else:
                combined[llm_pred.domain] = llm_pred

        return sorted(combined.values(), key=lambda x: -x.confidence)

    def _keyword_screening_sync(
        self, text: str, max_domains: int = 5
    ) -> List[DomainPrediction]:
        """Synchronous wrapper for keyword screening (for testing purposes).

        Args:
            text: Document text to analyze.
            max_domains: Maximum number of domains to return.

        Returns:
            List of DomainPrediction objects sorted by confidence.
        """
        import asyncio

        return asyncio.run(self._keyword_screening(text, max_domains))
