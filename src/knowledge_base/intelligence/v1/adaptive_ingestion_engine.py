"""Adaptive ingestion engine with LLM-powered pipeline optimization."""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, TypedDict

from pydantic import BaseModel, Field

from knowledge_base.common.gateway import GatewayClient
from knowledge_base.common.llm_logging_wrapper import (
    LLMCallLogger,
    log_llm_result,
)
from knowledge_base.ingestion.v1.gleaning_service import GleaningService
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    EntityExtractionManager,
)

logger = logging.getLogger(__name__)


class DocumentComplexity(str, Enum):
    """Document complexity levels."""

    SIMPLE = "simple"  # Straightforward text, few entities
    MODERATE = "moderate"  # Structured with multiple entities/relationships
    COMPLEX = "complex"  # Technical, dense with entities and relationships


class RecommendedApproach(str, Enum):
    """Recommended extraction approach."""

    GLEANING = "gleaning"  # Basic entity extraction
    GLEANING_ENHANCED = "gleaning_enhanced"  # Enhanced with context
    MULTI_AGENT = "multi_agent"  # Full multi-agent pipeline


class PipelineRecommendation(BaseModel):
    """Recommendation for pipeline configuration."""

    complexity: DocumentComplexity
    approach: RecommendedApproach
    chunk_size: int = Field(ge=512, le=4096)
    use_multi_agent: bool
    enable_enhancement: bool = True
    max_enhancement_iterations: int = Field(ge=1, le=5, default=2)
    enable_hallucination_detection: bool = True
    confidence_threshold: float = Field(ge=0.3, le=0.95, default=0.7)
    justification: str
    expected_entity_count: int = Field(ge=0, le=500)
    domains: list[str]
    estimated_processing_time: str


class AdaptiveIngestionEngine:
    """Engine that uses LLM to optimize ingestion pipeline."""

    def __init__(self, gateway: GatewayClient):
        """Initialize adaptive engine.

        Args:
            gateway: LLM gateway client
        """
        self._gateway = gateway
        self._gleaning_service = GleaningService(gateway)

    async def analyze_document(
        self,
        document_text: str,
        document_name: str = "",
        file_size_bytes: int = 0,
    ) -> PipelineRecommendation:
        """Analyze document and recommend pipeline configuration.

        Args:
            document_text: Document content
            document_name: Document filename
            file_size_bytes: File size

        Returns:
            Pipeline recommendation
        """
        # Sample first 2000 chars for analysis (avoid token limits)
        sample_text = document_text[:2000]
        if len(document_text) > 2000:
            sample_text += f"\n\n...[Document truncated, total size: {file_size_bytes} bytes]"

        prompt = f"""You are an expert document processing system. Analyze this document and recommend optimal processing parameters.

DOCUMENT SAMPLE:
```
{sample_text}
```

FILENAME: {document_name}
FILE SIZE: {file_size_bytes} bytes

Evaluate based on:
1. **Complexity Score**: Count technical terms, entities, relationships. Rate 1-10.
2. **Structure**: Is it highly structured (headings, sections) or narrative?
3. **Entity Density**: Entities per paragraph (0=none, 5=very dense)
4. **Domain Specificity**: Is this technical/financial/medical/legal or general?

Provide a JSON with these fields:
- complexity: "simple" | "moderate" | "complex"
- chunk_size: 512, 1024, 1536, 2048, 3072, or 4096 (optimal tokens per chunk)
- approach: "gleaning" | "gleaning_enhanced" | "multi_agent"
- max_enhancement_iterations: 1-5 (only if multi_agent)
- expected_entity_count: estimated entities (0-500)
- justification: brief explanation
- domains: list of domains ["tech", "finance", "medical", "legal", "scientific", "general"]
- estimated_processing_time: "fast" (<2min), "medium" (2-10min), "slow" (>10min)

EXAMPLES:
- Simple news article: {{"complexity": "simple", "chunk_size": 1024, "approach": "gleaning", "expected_entity_count": 5}}
- Financial report: {{"complexity": "moderate", "chunk_size": 1536, "approach": "multi_agent", "max_enhancement_iterations": 2}}
- Technical research paper: {{"complexity": "complex", "chunk_size": 2048, "approach": "multi_agent", "max_enhancement_iterations": 3}}

Respond with ONLY the JSON, no markdown or explanations."""

        try:
            # Log the analysis request
            logger.info(
                f"🔍 ADAPTIVE ANALYSIS START for document: {document_name}\n"
                f"   📊 File size: {file_size_bytes} bytes\n"
                f"   📝 Sample text length: {len(document_text)} chars"
            )

            async with LLMCallLogger(
                agent_name="AdaptiveIngestionEngine",
                document_id=document_name,
                step_info="Analysis Phase",
            ):
                response = await self._gateway.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Deterministic for consistency
                )

            log_llm_result(
                "AdaptiveIngestionEngine",
                response,
                document_name,
                metadata={"analysis_type": "document_complexity"},
            )

            # Extract content from response
            content = ""
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].get("message", {}).get("content", "")
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"]

            # Parse JSON response
            try:
                # Try to extract JSON if wrapped in markdown
                json_match = json.loads(content)
            except json.JSONDecodeError:
                # Clean common markdown wrappers
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                json_match = json.loads(content)

            recommendation = PipelineRecommendation(
                complexity=DocumentComplexity(json_match["complexity"]),
                approach=RecommendedApproach(json_match["approach"]),
                chunk_size=json_match.get("chunk_size", 1536),
                use_multi_agent=json_match["approach"] == "multi_agent",
                max_enhancement_iterations=json_match.get(
                    "max_enhancement_iterations", 2
                ),
                enable_hallucination_detection=json_match.get(
                    "enable_hallucination_detection", True
                ),
                confidence_threshold=json_match.get("confidence_threshold", 0.7),
                justification=json_match.get("justification", "AI-optimized pipeline"),
                expected_entity_count=json_match.get("expected_entity_count", 20),
                domains=json_match.get("domains", ["general"]),
                estimated_processing_time=json_match.get(
                    "estimated_processing_time", "medium"
                ),
            )

            logger.info(
                f"Adaptive analysis complete: complexity={recommendation.complexity}, "
                f"approach={recommendation.approach}, chunk_size={recommendation.chunk_size}"
            )

            return recommendation

        except Exception as e:
            logger.warning(f"Adaptive analysis failed: {e}, using defaults")
            # Return moderate defaults as fallback
            return PipelineRecommendation(
                complexity=DocumentComplexity.MODERATE,
                approach=RecommendedApproach.MULTI_AGENT,
                chunk_size=1536,
                use_multi_agent=True,
                max_enhancement_iterations=2,
                enable_hallucination_detection=True,
                confidence_threshold=0.7,
                justification="Fallback to defaults due to analysis error",
                expected_entity_count=20,
                domains=["general"],
                estimated_processing_time="medium",
            )

    def get_ingestion_strategy(self, recommendation: PipelineRecommendation) -> dict:
        """Convert recommendation to ingestion strategy.

        Args:
            recommendation: Pipeline recommendation

        Returns:
            Strategy configuration for orchestrator
        """
        strategy = {
            "use_multi_agent": recommendation.use_multi_agent,
            "chunk_size": recommendation.chunk_size,
            "enable_enhancement": recommendation.enable_enhancement,
            "max_enhancement_iterations": recommendation.max_enhancement_iterations,
            "enable_hallucination_detection": recommendation.enable_hallucination_detection,
            "confidence_threshold": recommendation.confidence_threshold,
            "estimated_calls": self._estimate_llm_calls(recommendation),
        }

        return strategy

    def _estimate_llm_calls(self, recommendation: PipelineRecommendation) -> int:
        """Estimate number of LLM calls based on recommendation.

        Args:
            recommendation: Pipeline recommendation

        Returns:
            Estimated LLM call count
        """
        if recommendation.approach == RecommendedApproach.GLEANING:
            # 1 call per chunk
            return max(1, recommendation.expected_entity_count // 20)

        elif recommendation.approach == RecommendedApproach.GLEANING_ENHANCED:
            # 1-2 calls per chunk (basic + optional enhancement)
            return max(2, (recommendation.expected_entity_count // 20) * 2)

        else:  # MULTI_AGENT
            # Perception + Enhancement (iterative) + Evaluation per chunk
            chunks = max(1, (recommendation.expected_entity_count * 50) // recommendation.chunk_size)
            calls_per_chunk = (
                1  # Perception
                + recommendation.max_enhancement_iterations  # Enhancement iterations
                + 1  # Evaluation
            )
            return chunks * calls_per_chunk + 5  # +5 for cross-domain detection, etc.
