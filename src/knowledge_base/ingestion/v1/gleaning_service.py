"""Adaptive gleaning extraction service."""

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.common.gateway import GatewayClient
import time
from knowledge_base.common.temporal_utils import (
    TemporalClaim,
    TemporalNormalizer,
    TemporalType,
)
from knowledge_base.persistence.v1.schema import EdgeType

logger = logging.getLogger(__name__)


class GleaningConfig(BaseSettings):
    """Gleaning configuration."""

    model_config = SettingsConfigDict()

    max_density_threshold: float = 0.8
    min_density_threshold: float = 0.3
    max_passes: int = 2
    diminishing_returns_threshold: float = 0.05  # Less than 5% new information
    stability_threshold: float = 0.90  # 90% stable


class ExtractedEntity(BaseModel):
    """Extracted entity."""

    name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    description: str | None = Field(None, description="Entity description")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )
    confidence: float = Field(default=1.0, description="Extraction confidence")


class ExtractedEdge(BaseModel):
    """Extracted relationship edge."""

    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    edge_type: EdgeType = Field(..., description="Relationship type")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Edge properties"
    )
    confidence: float = Field(default=1.0, description="Extraction confidence")

    # Additional fields for enhanced knowledge graph capabilities
    source_text: str | None = Field(
        None, description="Original text that describes this relationship"
    )
    provenance: str | None = Field(None, description="Source of the relationship")
    temporal_validity_start: datetime | None = Field(
        None, description="When relationship became valid"
    )
    temporal_validity_end: datetime | None = Field(
        None, description="When relationship ended (if applicable)"
    )
    original_edge_type: str | None = Field(
        None, description="Original edge type from LLM before standardization"
    )


class ExtractionResult(BaseModel):
    """Extraction result from a pass."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    temporal_claims: list[TemporalClaim] = Field(default_factory=list)
    information_density: float = Field(
        default=0.0, description="Remaining info density"
    )


class GleaningService:
    """Service for adaptive 2-pass gleaning extraction."""

    def __init__(
        self,
        gateway: GatewayClient,
        config: GleaningConfig | None = None,
    ) -> None:
        """Initialize gleaning service.

        Args:
            gateway: LLM gateway client.
            config: Gleaning configuration.
        """
        self._gateway = gateway
        self._config = config or GleaningConfig()
        self._temporal_normalizer = TemporalNormalizer()

    def calculate_new_information_gain(
        self,
        previous: ExtractionResult,
        current: ExtractionResult,
    ) -> float:
        """Calculate percentage of new information in current pass."""
        previous_entities = {e.name for e in previous.entities}
        current_entities = {e.name for e in current.entities}

        if not current_entities:
            return 0.0

        new_entities = current_entities - previous_entities
        return len(new_entities) / len(current_entities)

    def calculate_cross_pass_stability(
        self,
        results: list[ExtractionResult],
    ) -> float:
        """Calculate stability of extractions across passes."""
        if len(results) < 2:
            return 1.0

        recent = results[-1]
        previous = results[-2]

        recent_entities = {e.name for e in recent.entities}
        previous_entities = {e.name for e in previous.entities}

        intersection = recent_entities & previous_entities
        union = recent_entities | previous_entities

        if not union:
            return 1.0

        return len(intersection) / len(union)

    def should_continue_extraction(
        self,
        pass_num: int,
        result: ExtractionResult,
        previous_results: list[ExtractionResult],
    ) -> tuple[bool, str]:
        """Determine if extraction should continue to next pass.

        Args:
            pass_num: The number of the next pass we are considering (e.g., if we just
                     finished pass 1 and are considering pass 2, pass_num=2)
            result: The result from the most recent completed pass
            previous_results: All results collected so far (including the most recent)

        Returns:
            Tuple of (should_continue, reason)
        """

        # Check if the next pass number would exceed max passes
        if pass_num > self._config.max_passes:
            return False, "max_passes_reached"

        # Check if this is actually an LLM failure
        if result.information_density < 0:
            return False, "llm_failure"
        
        # Check information density of the most recent result
        if result.information_density < self._config.min_density_threshold:
            return False, "low_information_density"

        # Check diminishing returns - comparing previous result to current
        # To compare, we need at least 2 results total
        if len(previous_results) >= 2:
            # Compare the last 2 results we have
            prev_result = previous_results[-2]  # The result before the current one
            new_info_gain = self.calculate_new_information_gain(
                prev_result,  # Previous result
                result,  # Current result (the most recent)
            )
            if new_info_gain < self._config.diminishing_returns_threshold:
                return False, "diminishing_returns"

        # Check stability - requires at least 2 previous results to calculate
        if len(previous_results) >= 2:
            stability = self.calculate_cross_pass_stability(
                previous_results[-2:]  # The two most recent results
            )
            if stability >= self._config.stability_threshold:
                return False, "high_stability"

        # Additional check for long-tail distribution handling:
        # If information density is high but new information gain is low,
        # this might indicate we're finding mostly rare/long-tail relations
        if (
            result.information_density >= self._config.min_density_threshold
            and len(previous_results) >= 1
            and self.calculate_new_information_gain(previous_results[-1], result) < 0.1
        ):
            logger.info(
                f"High density but low new info gain - may be extracting long-tail relations"
            )

        # Additional check for long-tail distribution handling:
        # If information density is high but new information gain is low,
        # this might indicate we're finding mostly rare/long-tail relations
        if (
            result.information_density >= self._config.min_density_threshold
            and len(previous_results) >= 1
            and self.calculate_new_information_gain(previous_results[-1], result) < 0.1
        ):
            logger.info(
                f"High density but low new info gain - may be extracting long-tail relations"
            )

        # Additional check for long-tail distribution handling:
        # If information density is high but new information gain is low,
        # this might indicate we're finding mostly rare/long-tail relations
        if (
            result.information_density >= self._config.min_density_threshold
            and len(previous_results) >= 1
            and self.calculate_new_information_gain(previous_results[-1], result) < 0.1
        ):
            logger.info(
                f"High density but low new info gain - may be extracting long-tail relations"
            )

        return True, "continue"

    def _analyze_relation_distribution(self, edges: list[ExtractedEdge]) -> None:
        """Analyze the distribution of relation types to identify long-tail patterns.

        Based on 2026 research (DOREMI) on optimizing long-tail predictions in
        document-level relation extraction, where many relations have scarce examples.
        """
        if not edges:
            return

        # Count occurrence of each relation type
        relation_counts = {}
        for edge in edges:
            rel_type = edge.edge_type.value
            relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1

        # Identify potential long-tail relations (those with few occurrences)
        total_relations = len(edges)
        long_tail_threshold = 0.1  # Relations that appear in <10% of the total

        long_tail_relations = {
            rel_type: count
            for rel_type, count in relation_counts.items()
            if count / total_relations < long_tail_threshold
        }

        if long_tail_relations:
            logger.info(
                f"Long-tail relation analysis: {len(long_tail_relations)} rare relation types found out of {len(relation_counts)} total"
            )
            for rel_type, count in long_tail_relations.items():
                logger.debug(f"  - {rel_type}: {count} occurrences")

    async def extract(
        self,
        text: str,
        context: str | None = None,
    ) -> ExtractionResult:
        """Perform adaptive gleaning extraction.

        Args:
            text: Text to extract from.
            context: Optional surrounding context.

        Returns:
            Complete extraction result.
        """
        logger.info(f"🔍 EXTRACT: Starting extraction for text length {len(text)}")
        logger.info(f"🔍 EXTRACT: Text preview: {text[:100]}...")
        
        results: list[ExtractionResult] = []

        # Pass 1 always runs
        logger.info(f"🔍 EXTRACT: Starting Pass 1")
        pass_result = await self._extract_pass(text, 1, context, [])
        logger.info(f"🔍 EXTRACT: Pass 1 complete - entities: {len(pass_result.entities)}, edges: {len(pass_result.edges)}")
        logger.info(f"🔍 EXTRACT: Pass 1 info density: {pass_result.information_density}")
        results.append(pass_result)

        # Only run pass 2 if pass 1 indicates sufficient information density
        should_continue, reason = self.should_continue_extraction(
            pass_num=2, result=pass_result, previous_results=results
        )

        if should_continue:
            logger.info(f"Continuing to pass 2: {reason}")
            pass_result = await self._extract_pass(text, 2, context, results)
            results.append(pass_result)
        else:
            logger.info(f"Stopping after pass 1: {reason}")

        return self._merge_results(results)

    async def _extract_pass(
        self,
        text: str,
        pass_num: int,
        context: str | None,
        previous_results: list[ExtractionResult],
    ) -> ExtractionResult:
        logger.info(f"🔍 PASS {pass_num}: Starting extraction pass")
        logger.info(f"🔍 PASS {pass_num}: Text length: {len(text)}")
        if context:
            logger.info(f"🔍 PASS {pass_num}: Context provided: {context[:50]}...")
        """Perform single extraction pass.

        Args:
            text: Text to extract from.
            pass_num: Pass number (1 or 2).
            context: Optional context.
            previous_results: Results from previous passes.

        Returns:
            Extraction result for this pass.
        """
        logger.info(f"Starting pass {pass_num} for text of length {len(text)}")

        if pass_num == 1:
            system_prompt = self._get_discovery_prompt()
            user_prompt = self._get_discovery_user_prompt(text, context)
        else:
            system_prompt = self._get_gleaning_prompt()
            user_prompt = self._get_gleaning_user_prompt(
                text,
                context,
                previous_results[0] if previous_results else ExtractionResult(),
            )

        try:
            response = await self._gateway.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                json_mode=True,
            )
        except Exception as e:
            logger.error(f"Failed to generate text for pass {pass_num}: {e}")
            return ExtractionResult()

        logger.info(f"Completed pass {pass_num}, response length: {len(response)}")

        return self._parse_extraction_result(response, text)

    def _get_discovery_prompt(self) -> str:
        """Get discovery pass system prompt."""
        return """You are an expert information extraction system. Your task is to extract entities and relationships from the provided text.

Focus on:
1. Clearly named entities (people, organizations, locations, concepts)
2. Explicit relationships between entities
3. Temporal information (dates, times, durations) - ESPECIALLY from timeline entries

CRITICAL: You must respond with valid JSON only. Do not include markdown code blocks, explanations, or any text outside the JSON structure.

Output in the following JSON schema:
{
  "entities": [
    {
      "name": "string (entity name)",
      "type": "string (entity type)",
      "description": "string (optional description)",
      "confidence": 0.9
    }
  ],
  "edges": [
    {
      "source": "string (source entity name)",
      "target": "string (target entity name)",
      "type": "string (relationship type)",
      "confidence": 0.9
    }
  ],
  "temporal_claims": [
    {
      "text": "string (temporal text)",
      "type": "atemporal|static|dynamic",
      "date": "string (optional date)"
    }
  ],
  "information_density": 0.7
}

IMPORTANT FOR TEMPORAL CLAIMS:
- When you see timeline entries like "August 2021: Project initiated, status 'Active'", extract the FULL phrase as a temporal claim
- Include BOTH the date AND the status in the "text" field
- Set "type" to "static" for dated events
- Set "date" to the actual date mentioned

Be precise and factual. Only extract information explicitly stated in the text."""

    def _get_discovery_user_prompt(
        self,
        text: str,
        context: str | None,
    ) -> str:
        """Get discovery pass user prompt."""
        prompt = (
            f"Extract entities and relationships from the following text:\n\n{text}\n\n"
        )
        if context:
            prompt += f"Context: {context}\n\n"
        prompt += """IMPORTANT: For any timeline entries (lines starting with dates like "August 2021:" or "May 2023:"), extract the ENTIRE line including the status information as a temporal claim. For example:
- "August 2021: Project initiated, status 'Active'" should be extracted as a temporal claim with the full text
- "May 2023: Technical challenges encountered, status 'Failed'" should be extracted as a temporal claim

Provide your analysis in JSON format."""
        return prompt

    def _get_gleaning_prompt(self) -> str:
        """Get gleaning pass system prompt."""
        return """You are an expert information extraction system performing a second pass analysis.

Your task is to find subtle, nested, or technical relationships that were missed in the first pass.

Focus on:
1. Implicit relationships
2. Nested or hierarchical structures
3. Technical or domain-specific connections
4. Temporal relationships and dependencies - especially timeline entries with status changes

CRITICAL: You must respond with valid JSON only. Do not include markdown code blocks, explanations, or any text outside the JSON structure.

Output in the same JSON schema as the first pass:
{
  "entities": [{"name": "...", "type": "...", "description": "...", "confidence": 0.9}],
  "edges": [{"source": "...", "target": "...", "type": "...", "confidence": 0.9}],
  "temporal_claims": [{"text": "...", "type": "atemporal|static|dynamic", "date": "..."}],
  "information_density": 0.7
}

IMPORTANT FOR TEMPORAL CLAIMS:
- Look for timeline entries that combine dates with status information
- Extract the FULL phrase including both date and status (e.g., "August 2021: Project initiated, status 'Active'")
- Include status keywords like "Active", "Failed", "Success", "Completed" in the claim text

Be thorough but avoid hallucinating information not present in the text."""

    def _get_gleaning_user_prompt(
        self,
        text: str,
        context: str | None,
        previous_result: ExtractionResult,
    ) -> str:
        """Get gleaning pass user prompt."""
        previous_summary = f"""
Previous extraction found:
- {len(previous_result.entities)} entities
- {len(previous_result.edges)} relationships
- Information density: {previous_result.information_density:.2f}

Entities: {", ".join(e.name for e in previous_result.entities)}
"""

        prompt = (
            f"Perform a second pass extraction focusing on missed information.\n\n"
            f"{previous_summary}\n\n"
            f"Original text:\n{text}\n\n"
        )

        if context:
            prompt += f"Context: {context}\n\n"

        prompt += "Identify additional entities, relationships, and temporal claims in JSON format."

        return prompt

    def _parse_extraction_result(
        self,
        response: str,
        text: str,
    ) -> ExtractionResult:
        """Parse LLM response into extraction result.

        Args:
            response: LLM JSON response.
            text: Original text for temporal extraction.

        Returns:
            Parsed extraction result.
        """
        import json
        import re

        # Preprocess response to handle markdown code blocks
        cleaned_response = response.strip()

        # Remove markdown code block markers if present
        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(markdown_pattern, cleaned_response)
        if match:
            cleaned_response = match.group(1)

        # Also handle case where JSON might be surrounded by other text
        # Try to find the first { and last }
        first_brace = cleaned_response.find("{")
        last_brace = cleaned_response.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned_response = cleaned_response[first_brace : last_brace + 1]

        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response that failed parsing: {response}")
            return ExtractionResult()

        entities = []
        for e in data.get("entities", []):
            try:
                entity = ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "UNKNOWN"),
                    description=e.get("description"),
                    properties=e.get("properties", {}),
                    confidence=e.get("confidence", 1.0),
                )
                if entity.name:  # Only add if entity has a name
                    entities.append(entity)
            except Exception as ex:
                logger.warning(f"Failed to create entity from data {e}: {ex}")

        edges = []
        for edge in data.get("edges", []):
            try:
                edge_type = EdgeType(edge.get("type", "RELATED_TO"))
            except ValueError:
                # Handle unknown edge types by falling back to RELATED_TO or NOTA
                # This addresses the long-tail distribution problem where many relation types are rare
                raw_type = edge.get("type", "RELATED_TO")
                logger.warning(f"Invalid edge type: {raw_type}, using fallback")

                # For unknown types, use either RELATED_TO as general fallback or NOTA as "none-of-the-above"
                # This follows recent research on handling long-tail distributions in relation extraction
                if raw_type and raw_type.lower() not in ["unknown", "none", "null", ""]:
                    edge_type = EdgeType.RELATED_TO
                else:
                    edge_type = EdgeType.NOTA

            # Enhance edge properties with context to help with long-tail relation identification
            enhanced_properties = edge.get("properties", {}).copy()
            # Add source text context for better relation understanding
            if "source_text" in edge:
                enhanced_properties["source_text"] = edge["source_text"]
            if "relation_context" in edge:
                enhanced_properties["relation_context"] = edge["relation_context"]

            # Extract temporal information from the LLM response directly
            # The LLM should provide temporal information as part of the relationship
            temporal_start = None
            temporal_end = None

            # Check if the LLM provided temporal validity information
            if "temporal_validity_start" in edge:
                try:
                    from datetime import datetime

                    temp_start = edge["temporal_validity_start"]
                    if isinstance(temp_start, str):
                        temporal_start = (
                            datetime.fromisoformat(temp_start.replace("Z", "+00:00"))
                            if "Z" in temp_start
                            else datetime.fromisoformat(
                                temp_start.split("T")[0] + "T00:00:00"
                            )
                        )
                    elif isinstance(temp_start, datetime):
                        temporal_start = temp_start
                except (ValueError, TypeError):
                    pass

            if "temporal_validity_end" in edge:
                try:
                    from datetime import datetime

                    temp_end = edge["temporal_validity_end"]
                    if isinstance(temp_end, str):
                        temporal_end = (
                            datetime.fromisoformat(temp_end.replace("Z", "+00:00"))
                            if "Z" in temp_end
                            else datetime.fromisoformat(
                                temp_end.split("T")[0] + "T00:00:00"
                            )
                        )
                    elif isinstance(temp_end, datetime):
                        temporal_end = temp_end
                except (ValueError, TypeError):
                    pass

            # Default source text if not provided
            source_text = edge.get(
                "source_text",
                f"{edge.get('source', '')} {edge.get('type', '')} {edge.get('target', '')}",
            )

            try:
                edge_obj = ExtractedEdge(
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    edge_type=edge_type,
                    properties=enhanced_properties,
                    confidence=edge.get("confidence", 1.0),
                    source_text=source_text,
                    provenance=edge.get("provenance"),
                    temporal_validity_start=temporal_start,
                    temporal_validity_end=temporal_end,
                    original_edge_type=edge.get(
                        "type"
                    ),  # Store the original type before standardization
                )
                if (
                    edge_obj.source and edge_obj.target
                ):  # Only add if both source and target exist
                    edges.append(edge_obj)
            except Exception as ex:
                logger.warning(f"Failed to create edge from data {edge}: {ex}")

        temporal_claims: list[TemporalClaim] = []
        for claim_data in data.get("temporal_claims", []):
            try:
                claim = self._temporal_normalizer.extract_temporal_info(
                    claim_data.get("text", "")
                )

                if claim_data.get("type"):
                    try:
                        claim.temporal_type = TemporalType(claim_data["type"])
                    except ValueError:
                        logger.warning(f"Invalid temporal type: {claim_data['type']}")

                temporal_claims.append(claim)
            except Exception as ex:
                logger.warning(
                    f"Failed to create temporal claim from data {claim_data}: {ex}"
                )

        information_density = data.get("information_density", 0.5)
        # Ensure information density is within bounds
        information_density = max(0.0, min(1.0, information_density))

        return ExtractionResult(
            entities=entities,
            edges=edges,
            temporal_claims=temporal_claims,
            information_density=information_density,
        )

    def _merge_results(self, results: list[ExtractionResult]) -> ExtractionResult:
        """Merge results from multiple passes.

        Args:
            results: List of extraction results.

        Returns:
            Merged extraction result.
        """
        entities: dict[str, ExtractedEntity] = {}
        edges: dict[str, ExtractedEdge] = {}
        all_temporal_claims: list[TemporalClaim] = []

        for result in results:
            # Merge entities
            for entity in result.entities:
                if entity.name not in entities:
                    entities[entity.name] = entity
                else:
                    existing = entities[entity.name]
                    # Update confidence to the maximum of both passes
                    existing.confidence = max(existing.confidence, entity.confidence)
                    # Merge properties from both passes
                    existing.properties.update(entity.properties)

            # Merge edges
            for edge in result.edges:
                edge_key = f"{edge.source}|{edge.target}|{edge.edge_type}"
                if edge_key not in edges:
                    edges[edge_key] = edge
                else:
                    existing = edges[edge_key]
                    # Update confidence to the maximum of both passes
                    existing.confidence = max(existing.confidence, edge.confidence)
                    # Merge properties from both passes
                    existing.properties.update(edge.properties)

            all_temporal_claims.extend(result.temporal_claims)

        # Deduplicate temporal claims
        seen_claims = set()
        unique_temporal_claims = []
        for claim in all_temporal_claims:
            claim_key = claim.text.lower().strip()
            if claim_key and claim_key not in seen_claims:
                seen_claims.add(claim_key)
                unique_temporal_claims.append(claim)

        # Calculate average information density
        if results:
            # Use the information density from the last pass as it reflects the remaining density
            final_density = results[-1].information_density
        else:
            final_density = 0.0

        # Analyze relation distribution to identify potential long-tail patterns
        # Based on 2026 research on optimizing long-tail predictions in document-level relation extraction
        self._analyze_relation_distribution(list(edges.values()))

        return ExtractionResult(
            entities=list(entities.values()),
            edges=list(edges.values()),
            temporal_claims=unique_temporal_claims,
            information_density=final_density,
        )
