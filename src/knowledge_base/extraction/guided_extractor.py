"""Guided extraction with domain-specific prompts and user goals."""

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from knowledge_base.common.gateway import GatewayClient
from knowledge_base.extraction.template_registry import (
    ExtractionGoal,
    TemplateRegistry,
)
from knowledge_base.persistence.v1.schema import EdgeType

logger = logging.getLogger(__name__)


class ExtractionPrompt(BaseModel):
    """Single extraction prompt for a goal."""

    goal_name: str = Field(..., description="Name of the extraction goal")
    system_prompt: str = Field(..., description="System prompt for extraction")
    user_prompt: str = Field(..., description="User prompt with document text")
    target_entities: List[str] = Field(..., description="Entity types to extract")
    target_relationships: List[str] = Field(
        ..., description="Relationship types to extract"
    )
    priority: int = Field(..., description="Priority level")


class ExtractionPrompts(BaseModel):
    """Collection of extraction prompts for a document."""

    domain: str = Field(..., description="Detected or specified domain")
    prompts: List[ExtractionPrompt] = Field(
        default_factory=list, description="Extraction prompts"
    )
    user_goals_used: bool = Field(
        default=False, description="Whether user-provided goals were used"
    )
    original_user_goals: List[str] = Field(
        default_factory=list, description="Original user-provided goals"
    )


class GuidedExtractor:
    """Guided extraction with domain-specific prompts.

    This class generates extraction prompts based on either:
    1. Domain-specific default goals (fully automated mode when user_goals is None)
    2. User-provided custom goals (interpreted and applied)

    Attributes:
        llm: LLM gateway client for extraction.
        templates: Template registry for domain-specific goals.
    """

    def __init__(
        self,
        llm_client: GatewayClient,
        template_registry: Optional[TemplateRegistry] = None,
    ) -> None:
        """Initialize guided extractor.

        Args:
            llm_client: LLM gateway client for extraction.
            template_registry: Optional template registry. Creates default if None.
        """
        self.llm = llm_client
        self.templates = template_registry or TemplateRegistry()

    async def generate_extraction_prompts(
        self,
        document_text: str,
        user_goals: Optional[List[str]] = None,
        domain: Optional[str] = None,
        previous_extraction: Optional["ExtractionResult"] = None,
    ) -> ExtractionPrompts:
        """Generate domain-specific extraction prompts.

        This is the main entry point for guided extraction. It determines the mode
        (automated or user-guided) and generates appropriate prompts.

        Args:
            document_text: Text content to extract from.
            user_goals: Optional list of user-specified extraction goals.
                        If None, uses domain-based default goals (fully automated).
            domain: Optional domain hint. Detected from content if not provided.
            previous_extraction: Optional previous extraction result for iterative extraction.

        Returns:
            ExtractionPrompts containing all prompts to use for extraction.
        """
        if user_goals is None:
            detected_domain = domain or await self._detect_domain(document_text)
            logger.info(f"Using automated mode with domain: {detected_domain}")
            return await self._auto_mode(
                document_text, detected_domain, previous_extraction
            )
        else:
            logger.info(f"Using user-guided mode with goals: {user_goals}")
            return await self._interpret_user_goals(
                document_text, user_goals, previous_extraction
            )

    async def _auto_mode(
        self,
        document_text: str,
        domain: str,
        previous_extraction: Optional["ExtractionResult"] = None,
    ) -> ExtractionPrompts:
        """Automated mode: Use domain-based default goals.

        Args:
            document_text: Document text to extract from.
            domain: Detected domain.
            previous_extraction: Optional previous extraction result.

        Returns:
            ExtractionPrompts with domain-specific prompts.
        """
        goals = self.templates.get_prioritized_goals(domain)

        prompts = []
        for goal in goals:
            prompt = self._build_goal_prompt(
                document_text, goal, domain, previous_extraction
            )
            prompts.append(prompt)

        return ExtractionPrompts(
            domain=domain,
            prompts=prompts,
            user_goals_used=False,
            original_user_goals=[],
        )

    async def _interpret_user_goals(
        self,
        document_text: str,
        user_goals: List[str],
        previous_extraction: Optional["ExtractionResult"] = None,
    ) -> ExtractionPrompts:
        """Interpret and apply user-provided goals.

        Args:
            document_text: Document text to extract from.
            user_goals: List of user-specified goal names or descriptions.
            previous_extraction: Optional previous extraction result.

        Returns:
            ExtractionPrompts with user-guided prompts.
        """
        domain = await self._detect_domain(document_text)
        domain_goals = self.templates.get_goals(domain)

        matched_goals: List[ExtractionGoal] = []
        unmatched_goals: List[str] = []

        for user_goal in user_goals:
            user_goal_lower = user_goal.lower().strip()

            matched = False
            for goal in domain_goals:
                if (
                    goal.name.lower() == user_goal_lower
                    or user_goal_lower in goal.name.lower()
                    or user_goal_lower in goal.description.lower()
                ):
                    matched_goals.append(goal)
                    matched = True
                    break

            if not matched:
                unmatched_goals.append(user_goal)

        if not matched_goals:
            logger.warning(
                f"No goals matched for user input: {user_goals}. "
                f"Falling back to domain defaults for {domain}."
            )
            matched_goals = self.templates.get_prioritized_goals(domain)[:3]

        if unmatched_goals:
            logger.info(
                f"Unmatched goals will use general entity extraction: {unmatched_goals}"
            )

        prompts = []
        for goal in matched_goals:
            prompt = self._build_goal_prompt(
                document_text, goal, domain, previous_extraction, user_goals
            )
            prompts.append(prompt)

        if unmatched_goals:
            general_prompt = self._build_general_extraction_prompt(
                document_text, unmatched_goals, previous_extraction
            )
            prompts.append(general_prompt)

        return ExtractionPrompts(
            domain=domain,
            prompts=prompts,
            user_goals_used=True,
            original_user_goals=user_goals,
        )

    def _build_goal_prompt(
        self,
        document_text: str,
        goal: ExtractionGoal,
        domain: str,
        previous_extraction: Optional["ExtractionResult"] = None,
        user_goals: Optional[List[str]] = None,
    ) -> ExtractionPrompt:
        """Build extraction prompt for a specific goal.

        Args:
            document_text: Document text to extract from.
            goal: Extraction goal to build prompt for.
            domain: Domain context.
            previous_extraction: Optional previous extraction result.
            user_goals: Optional user-provided goals for context.

        Returns:
            ExtractionPrompt for the goal.
        """
        system_prompt = self._build_system_prompt(goal, domain)
        user_prompt = self._build_user_prompt(document_text, goal, previous_extraction)

        return ExtractionPrompt(
            goal_name=goal.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_entities=goal.target_entities,
            target_relationships=goal.target_relationships,
            priority=goal.priority,
        )

    def _build_system_prompt(self, goal: ExtractionGoal, domain: str) -> str:
        """Build system prompt from extraction goal.

        Args:
            goal: Extraction goal to build prompt for.
            domain: Domain context.

        Returns:
            System prompt string.
        """
        entity_list = ", ".join(goal.target_entities)
        relationship_list = ", ".join(goal.target_relationships)
        examples = ", ".join(goal.examples) if goal.examples else "N/A"

        prompt = f"""You are an expert information extraction system specialized in the {domain} domain.

Your task is to extract information for the goal: {goal.name}

GOAL DESCRIPTION:
{goal.description}

EXTRACT THE FOLLOWING ENTITY TYPES:
{entity_list}

EXTRACT THE FOLLOWING RELATIONSHIP TYPES:
{relationship_list}

EXAMPLE ENTITIES TO GUIDE EXTRACTION:
{examples}

OUTPUT REQUIREMENTS:
1. Respond with valid JSON only - no markdown code blocks, no explanations
2. Focus strictly on entities and relationships matching the goal
3. Use the exact entity and relationship types specified above
4. Provide confidence scores (0.0-1.0) for each extraction
5. Extract only information explicitly stated in the text

JSON OUTPUT SCHEMA:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "entity type from the specified list",
      "description": "brief description",
      "confidence": 0.95
    }}
  ],
  "edges": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "type": "relationship type from the specified list",
      "confidence": 0.9
    }}
  ],
  "information_density": 0.8
}}

REMEMBER: Output only valid JSON. Do not include any text outside the JSON structure."""

        return prompt

    def _build_user_prompt(
        self,
        document_text: str,
        goal: ExtractionGoal,
        previous_extraction: Optional["ExtractionResult"] = None,
    ) -> str:
        """Build user prompt with document text.

        Args:
            document_text: Document text to extract from.
            goal: Extraction goal context.
            previous_extraction: Optional previous extraction result.

        Returns:
            User prompt string.
        """
        prompt_parts = [
            f"Extract {goal.name} from the following text:\n\n{document_text}",
        ]

        if previous_extraction and previous_extraction.entities:
            previous_entities = [e.name for e in previous_extraction.entities]
            prompt_parts.append(
                f"\n\nPreviously extracted entities: {', '.join(previous_entities[:20])}"
            )
            prompt_parts.append(
                "\nFocus on finding additional entities and relationships that were missed."
            )

        prompt_parts.append("\n\nProvide your extraction in JSON format.")
        prompt_parts.append("\n\nIMPORTANT: Respond with valid JSON only.")

        return "".join(prompt_parts)

    def _build_general_extraction_prompt(
        self,
        document_text: str,
        unmatched_goals: List[str],
        previous_extraction: Optional["ExtractionResult"] = None,
    ) -> ExtractionPrompt:
        """Build a general extraction prompt for unmatched user goals.

        Args:
            document_text: Document text to extract from.
            unmatched_goals: List of unmatched user goals.
            previous_extraction: Optional previous extraction result.

        Returns:
            ExtractionPrompt for general extraction.
        """
        system_prompt = """You are an expert information extraction system.

Your task is to extract named entities and relationships from the text based on user-specified requirements.

EXTRACT THE FOLLOWING ENTITY TYPES:
- Person
- Organization
- Location
- Event
- Concept
- Any other entity types relevant to the user's goals

EXTRACT THE FOLLOWING RELATIONSHIP TYPES:
- related_to
- located_in
- participated_in
- produces
- known_for
- Any other relationships relevant to the user's goals

OUTPUT REQUIREMENTS:
1. Respond with valid JSON only - no markdown code blocks, no explanations
2. Focus on entities and relationships matching the user's goals
3. Provide confidence scores (0.0-1.0) for each extraction
4. Extract only information explicitly stated in the text

JSON OUTPUT SCHEMA:
{
  "entities": [
    {
      "name": "entity name",
      "type": "entity type",
      "description": "brief description",
      "confidence": 0.95
    }
  ],
  "edges": [
    {
      "source": "source entity name",
      "target": "target entity name",
      "type": "relationship type",
      "confidence": 0.9
    }
  ],
  "information_density": 0.8
}

REMEMBER: Output only valid JSON. Do not include any text outside the JSON structure."""

        user_goal_text = ", ".join(unmatched_goals)
        user_prompt = f"""Extract information related to the following goals from the text:
{user_goal_text}

Text:
{document_text}
{"" if not previous_extraction or not previous_extraction.entities else f"\n\nPreviously extracted: {', '.join([e.name for e in previous_extraction.entities[:20]])}\nFocus on additional entities."}

Provide your extraction in JSON format.
IMPORTANT: Respond with valid JSON only."""

        return ExtractionPrompt(
            goal_name="general_extraction",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_entities=[
                "Person",
                "Organization",
                "Location",
                "Event",
                "Concept",
            ],
            target_relationships=[
                "related_to",
                "located_in",
                "participated_in",
            ],
            priority=5,
        )

    def _build_extraction_schema(self, goals: List[ExtractionGoal]) -> Dict[str, Any]:
        """Build JSON schema from extraction goals.

        Args:
            goals: List of extraction goals.

        Returns:
            JSON schema dictionary for extraction output.
        """
        all_entities = set()
        all_relationships = set()

        for goal in goals:
            all_entities.update(goal.target_entities)
            all_relationships.update(goal.target_relationships)

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": list(all_entities)},
                            "description": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": ["name", "type", "confidence"],
                    },
                },
                "edges": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "type": {"type": "string", "enum": list(all_relationships)},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": ["source", "target", "type", "confidence"],
                    },
                },
                "information_density": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": ["entities", "edges", "information_density"],
        }

        return schema

    async def _detect_domain(self, document_text: str) -> str:
        """Detect domain from document content.

        Args:
            document_text: Document text to analyze.

        Returns:
            Detected domain string.
        """
        domain_indicators = {
            "TECHNOLOGY": [
                "software",
                "api",
                "code",
                "database",
                "server",
                "cloud",
                "kubernetes",
                "python",
                "programming",
                "framework",
                "library",
            ],
            "FINANCIAL": [
                "revenue",
                "profit",
                "investment",
                "stock",
                "market",
                "billion",
                "quarter",
                "earnings",
                "IPO",
                "merger",
            ],
            "MEDICAL": [
                "patient",
                "diagnosis",
                "treatment",
                "symptom",
                "disease",
                "hospital",
                "clinical",
                "drug",
                "therapy",
                "health",
            ],
            "LEGAL": [
                "court",
                "plaintiff",
                "defendant",
                "attorney",
                "ruling",
                "judgment",
                "statute",
                "law",
                "legal",
                "contract",
            ],
            "SCIENTIFIC": [
                "research",
                "study",
                "experiment",
                "hypothesis",
                "data",
                "analysis",
                "methodology",
                "journal",
                "peer-reviewed",
                "hypothesis",
            ],
        }

        text_lower = document_text.lower()
        domain_scores: Dict[str, int] = {}

        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            domain_scores[domain] = score

        if domain_scores:
            detected_domain = max(domain_scores, key=domain_scores.get)  # type: ignore
            if domain_scores[detected_domain] > 0:
                return detected_domain

        return "GENERAL"

    async def extract_with_prompts(
        self,
        prompts: ExtractionPrompts,
        context: Optional[str] = None,
    ) -> List["ExtractionResult"]:
        """Execute extraction using generated prompts.

        Args:
            prompts: ExtractionPrompts containing prompts to execute.
            context: Optional context for extraction.

        Returns:
            List of extraction results from each prompt.
        """
        from knowledge_base.ingestion.v1.gleaning_service import ExtractionResult

        results = []

        for prompt in sorted(prompts.prompts, key=lambda p: p.priority):
            try:
                response = await self.llm.generate_text(
                    prompt=prompt.user_prompt,
                    system_prompt=prompt.system_prompt,
                    temperature=0.0,
                    json_mode=True,
                )

                result = self._parse_extraction_result(response, prompt.goal_name)
                results.append(result)

                logger.info(
                    f"Extraction for goal '{prompt.goal_name}': "
                    f"{len(result.entities)} entities, {len(result.edges)} edges"
                )
            except Exception as e:
                logger.error(f"Extraction failed for goal '{prompt.goal_name}': {e}")

        return results

    def _parse_extraction_result(
        self, response: str, goal_name: str
    ) -> "ExtractionResult":
        """Parse LLM response into ExtractionResult.

        Args:
            response: LLM JSON response.
            goal_name: Name of the goal that generated this response.

        Returns:
            Parsed ExtractionResult.
        """
        from knowledge_base.ingestion.v1.gleaning_service import (
            ExtractionResult as LegacyExtractionResult,
        )
        from knowledge_base.ingestion.v1.gleaning_service import (
            ExtractedEntity,
            ExtractedEdge,
        )

        import re

        cleaned_response = response.strip()

        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(markdown_pattern, cleaned_response)
        if match:
            cleaned_response = match.group(1)

        first_brace = cleaned_response.find("{")
        last_brace = cleaned_response.rfind("}")
        if first_brace != -1 and last_brace != -1:
            cleaned_response = cleaned_response[first_brace : last_brace + 1]

        try:
            data = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction response for {goal_name}: {e}")
            return LegacyExtractionResult()

        entities = []
        for e in data.get("entities", []):
            try:
                entity = ExtractedEntity(
                    name=e.get("name", ""),
                    entity_type=e.get("type", "UNKNOWN"),
                    description=e.get("description"),
                    properties={"extraction_goal": goal_name},
                    confidence=e.get("confidence", 1.0),
                )
                if entity.name:
                    entities.append(entity)
            except Exception as ex:
                logger.warning(f"Failed to create entity: {ex}")

        edges = []
        for edge in data.get("edges", []):
            try:
                edge_type = EdgeType(edge.get("type", "RELATED_TO"))
            except ValueError:
                edge_type = EdgeType.RELATED_TO

            edge_obj = ExtractedEdge(
                source=edge.get("source", ""),
                target=edge.get("target", ""),
                edge_type=edge_type,
                properties={"extraction_goal": goal_name},
                confidence=edge.get("confidence", 1.0),
            )
            if edge_obj.source and edge_obj.target:
                edges.append(edge_obj)

        information_density = data.get("information_density", 0.5)
        information_density = max(0.0, min(1.0, information_density))

        return LegacyExtractionResult(
            entities=entities,
            edges=edges,
            information_density=information_density,
        )


ExtractionResult = None
