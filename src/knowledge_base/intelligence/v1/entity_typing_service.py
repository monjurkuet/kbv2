"""Entity typing service with few-shot prompting and domain awareness."""

import logging
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.common.gateway import GatewayClient
from knowledge_base.common.llm_logging_wrapper import (
    LLMCallLogger,
    log_llm_result,
)

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity type taxonomy."""

    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    PRODUCT = "Product"
    OTHER = "Other"


class DomainType(str, Enum):
    """Domain types for domain-aware entity typing."""

    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    ACADEMIC = "academic"
    SCIENTIFIC = "scientific"
    GOVERNMENT = "government"


class EntityTypingConfig(BaseSettings):
    """Entity typing configuration."""

    model_config = SettingsConfigDict()

    confidence_threshold: float = 0.6
    max_few_shot_examples: int = 5
    temperature: float = 0.2
    enable_domain_awareness: bool = True


class EntityTypingResult(BaseModel):
    """Result of entity typing."""

    entity_id: UUID = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: EntityType = Field(..., description="Predicted entity type")
    confidence_score: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Reasoning for the type prediction")
    alternative_types: list[tuple[EntityType, float]] = Field(
        default_factory=list, description="Alternative type candidates with scores"
    )
    domain: DomainType | None = Field(None, description="Detected domain")
    human_review_required: bool = Field(
        default=False, description="Flag for manual review"
    )


class TypedEntity(BaseModel):
    """Entity with typing information for batch processing."""

    entity_id: UUID = Field(..., description="Entity ID")
    name: str = Field(..., description="Entity name")
    description: str | None = Field(None, description="Entity description")
    source_text: str | None = Field(None, description="Source text context")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )


class FewShotExample(BaseModel):
    """Few-shot example for prompting."""

    name: str = Field(..., description="Entity name")
    entity_type: EntityType = Field(..., description="Correct entity type")
    context: str = Field(..., description="Context for disambiguation")


class PromptTemplate(BaseModel):
    """Prompt template for entity typing."""

    system_prompt: str = Field(..., description="System prompt")
    user_prompt_template: str = Field(
        ..., description="User prompt template with placeholders"
    )
    domain_suffix: dict[str, str] = Field(
        default_factory=dict, description="Domain-specific prompt additions"
    )

    def format_user_prompt(
        self,
        entity_name: str,
        entity_description: str | None,
        source_text: str | None,
        properties: dict[str, Any] | None,
        domain: DomainType | None,
        examples: list[FewShotExample],
    ) -> str:
        """Format user prompt with entity information."""
        context = source_text or "No source context available"
        props_str = (
            "\n".join(f"  {k}: {v}" for k, v in (properties or {}).items())
            or "  No properties"
        )
        examples_str = self._format_examples(examples)
        domain_info = f"\nDomain: {domain.value}" if domain else ""

        return self.user_prompt_template.format(
            entity_name=entity_name,
            entity_description=entity_description or "No description",
            source_text=context,
            properties=props_str,
            examples=examples_str,
            domain_info=domain_info,
        )

    def _format_examples(self, examples: list[FewShotExample]) -> str:
        """Format few-shot examples."""
        if not examples:
            return "No examples provided."

        formatted = []
        for ex in examples[:5]:
            formatted.append(
                f"- Entity: {ex.name}\n"
                f"  Context: {ex.context}\n"
                f"  Type: {ex.entity_type.value}"
            )
        return "\n".join(formatted)


class PromptTemplateRegistry:
    """Registry of prompt templates for entity typing."""

    def __init__(self) -> None:
        """Initialize prompt templates."""
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default prompt templates."""
        self._templates["default"] = PromptTemplate(
            system_prompt="""You are an expert entity classification system. Your task is to classify named entities into their correct type categories.

Entity Types:
- Person: Individual people, including fictional characters, historical figures, and public individuals
- Organization: Companies, institutions, government agencies, non-profits, and other organized groups
- Location: Geographic places including cities, countries, landmarks, and facilities
- Event: Occurrences including conferences, wars, natural disasters, and scheduled events
- Concept: Abstract ideas, theories, beliefs, and philosophical concepts
- Product: Physical or digital products, software, and services
- Other: Entities that don't fit into other categories

Rules:
1. Consider the entity name, description, and context
2. Use domain knowledge for disambiguation
3. Assign a confidence score based on evidence strength
4. Consider alternative classifications if ambiguous
5. Provide clear reasoning for your decision

Output your analysis in JSON format with the following structure:
{
  "entity_type": "<EntityType>",
  "confidence_score": 0.85,
  "reasoning": "<detailed explanation>",
  "alternative_types": [
    {"type": "<AlternativeType>", "score": 0.3}
  ],
  "detected_domain": "<DomainType>"
}""",
            user_prompt_template="""Classify the following entity:

Entity Name: {entity_name}
Description: {entity_description}
Source Context: {source_text}
Properties:
{properties}
{domain_info}

Examples of previous classifications:
{examples}

Provide your classification in JSON format.""",
            domain_suffix={
                "medical": " Pay special attention to medical terminology, drug names, diseases, and medical procedures.",
                "legal": " Consider legal terminology, case law, statutes, and legal institutions.",
                "financial": " Focus on financial instruments, markets, institutions, and economic concepts.",
                "technology": " Consider tech companies, software, protocols, and technical standards.",
                "academic": " Pay attention to academic institutions, research papers, and scholarly concepts.",
                "scientific": " Focus on scientific terminology, discoveries, and research entities.",
                "government": " Consider government agencies, political entities, and policy-related terms.",
            },
        )

    def get_template(self, domain: DomainType | None = None) -> PromptTemplate:
        """Get prompt template for domain."""
        if domain and domain != DomainType.GENERAL:
            template = self._templates.get(domain.value)
            if template:
                return template
        return self._templates["default"]

    def register_template(self, name: str, template: PromptTemplate) -> None:
        """Register a custom prompt template."""
        self._templates[name] = template


class FewShotExampleBank:
    """Bank of few-shot examples for entity typing."""

    def __init__(self) -> None:
        """Initialize example bank."""
        self._examples: dict[EntityType, list[FewShotExample]] = {}
        self._domain_examples: dict[str, dict[EntityType, list[FewShotExample]]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default few-shot examples."""
        person_examples = [
            FewShotExample(
                name="Albert Einstein",
                entity_type=EntityType.PERSON,
                context="Theoretical physicist who developed the theory of relativity",
            ),
            FewShotExample(
                name="OpenAI",
                entity_type=EntityType.ORGANIZATION,
                context="AI research laboratory developing artificial intelligence",
            ),
            FewShotExample(
                name="Paris",
                entity_type=EntityType.LOCATION,
                context="Capital city of France known for the Eiffel Tower",
            ),
            FewShotExample(
                name="World War II",
                entity_type=EntityType.EVENT,
                context="Global conflict from 1939 to 1945 involving major world powers",
            ),
            FewShotExample(
                name="Democracy",
                entity_type=EntityType.CONCEPT,
                context="System of government where citizens exercise power through voting",
            ),
            FewShotExample(
                name="iPhone",
                entity_type=EntityType.PRODUCT,
                context="Smartphone developed and sold by Apple Inc.",
            ),
            FewShotExample(
                name="Microsoft Corporation",
                entity_type=EntityType.ORGANIZATION,
                context="Multinational technology company headquartered in Redmond, Washington",
            ),
            FewShotExample(
                name="Google",
                entity_type=EntityType.ORGANIZATION,
                context="Multinational technology company specializing in search and advertising",
            ),
            FewShotExample(
                name="New York",
                entity_type=EntityType.LOCATION,
                context="Largest city in the United States located in the state of New York",
            ),
            FewShotExample(
                name="Shakespeare",
                entity_type=EntityType.PERSON,
                context="English playwright and poet widely regarded as the greatest writer in the English language",
            ),
        ]

        medical_examples = [
            FewShotExample(
                name="Metformin",
                entity_type=EntityType.PRODUCT,
                context="Medication used to treat type 2 diabetes by controlling blood sugar levels",
            ),
            FewShotExample(
                name="COVID-19",
                entity_type=EntityType.EVENT,
                context="Global pandemic caused by the SARS-CoV-2 coronavirus",
            ),
        ]

        self._examples[EntityType.PERSON] = person_examples[:2]
        self._examples[EntityType.ORGANIZATION] = person_examples[1:3]
        self._examples[EntityType.LOCATION] = [person_examples[2]]
        self._examples[EntityType.EVENT] = [person_examples[3]]
        self._examples[EntityType.CONCEPT] = [person_examples[4]]
        self._examples[EntityType.PRODUCT] = [person_examples[5], person_examples[6]]

        self._domain_examples["medical"] = {
            EntityType.PERSON: [
                FewShotExample(
                    name="Dr. Anthony Fauci",
                    entity_type=EntityType.PERSON,
                    context="American immunologist and director of the NIAID",
                )
            ],
            EntityType.PRODUCT: medical_examples[:1],
            EntityType.EVENT: medical_examples[1:],
        }

    def get_examples(
        self,
        entity_type: EntityType | None = None,
        domain: DomainType | None = None,
        limit: int = 3,
    ) -> list[FewShotExample]:
        """Get few-shot examples for entity typing."""
        examples: list[FewShotExample] = []

        if domain and domain != DomainType.GENERAL:
            domain_key = domain.value
            if domain_key in self._domain_examples:
                domain_dict = self._domain_examples[domain_key]
                if entity_type and entity_type in domain_dict:
                    examples = domain_dict[entity_type]
                else:
                    for ex_list in domain_dict.values():
                        examples.extend(ex_list)

        if not examples:
            if entity_type and entity_type in self._examples:
                examples = self._examples[entity_type]
            else:
                for ex_list in self._examples.values():
                    examples.extend(ex_list)

        return examples[:limit]

    def add_example(self, example: FewShotExample, domain: str | None = None) -> None:
        """Add a few-shot example."""
        if domain:
            if domain not in self._domain_examples:
                self._domain_examples[domain] = {}
            if example.entity_type not in self._domain_examples[domain]:
                self._domain_examples[domain][example.entity_type] = []
            self._domain_examples[domain][example.entity_type].append(example)
        else:
            if example.entity_type not in self._examples:
                self._examples[example.entity_type] = []
            self._examples[example.entity_type].append(example)


class EntityTyper:
    """Entity typing service with few-shot prompting and domain awareness."""

    def __init__(
        self,
        gateway: GatewayClient,
        config: EntityTypingConfig | None = None,
    ) -> None:
        """Initialize entity typer.

        Args:
            gateway: LLM gateway client.
            config: Entity typing configuration.
        """
        self._gateway = gateway
        self._config = config or EntityTypingConfig()
        self._template_registry = PromptTemplateRegistry()
        self._example_bank = FewShotExampleBank()

    async def type_entity(
        self,
        entity_id: UUID,
        name: str,
        description: str | None = None,
        source_text: str | None = None,
        properties: dict[str, Any] | None = None,
        domain: DomainType | None = None,
    ) -> EntityTypingResult:
        """Type a single entity.

        Args:
            entity_id: Entity ID.
            name: Entity name.
            description: Entity description.
            source_text: Source text context.
            properties: Entity properties.
            domain: Domain for domain-aware typing.

        Returns:
            Entity typing result with predicted type and confidence.
        """
        entity = TypedEntity(
            entity_id=entity_id,
            name=name,
            description=description,
            source_text=source_text,
            properties=properties or {},
        )

        results = await self.type_entities([entity], domain)
        return results[0]

    async def type_entities(
        self,
        entities: list[TypedEntity],
        domain: DomainType | None = None,
    ) -> list[EntityTypingResult]:
        """Type multiple entities.

        Args:
            entities: List of entities to type.
            domain: Domain for domain-aware typing.

        Returns:
            List of entity typing results.
        """
        if not entities:
            return []

        effective_domain = domain or DomainType.GENERAL
        template = self._template_registry.get_template(effective_domain)
        examples = self._example_bank.get_examples(domain=effective_domain, limit=3)

        results = await asyncio.gather(
            *[
                self._type_single_entity(entity, template, examples, effective_domain)
                for entity in entities
            ]
        )

        return results

    async def _type_single_entity(
        self,
        entity: TypedEntity,
        template: PromptTemplate,
        examples: list[FewShotExample],
        domain: DomainType,
    ) -> EntityTypingResult:
        """Type a single entity using LLM."""
        user_prompt = template.format_user_prompt(
            entity_name=entity.name,
            entity_description=entity.description,
            source_text=entity.source_text,
            properties=entity.properties,
            domain=domain,
            examples=examples,
        )

        # Log entity typing call
        logger.info(
            f"🎯 ENTITY TYPING: {entity.name[:50]}...\n"
            f"   📄 Description: {entity.description[:100] if entity.description else 'N/A'}\n"
            f"   📝 Types to consider: {list(set(e.entity_type.value for e in examples))[:5]}"
        )

        async with LLMCallLogger(
            agent_name="EntityTyper",
            document_id=str(entity.entity_id),
            step_info=f"Type determination",
        ):
            response = await self._gateway.generate_text(
                prompt=user_prompt,
                system_prompt=template.system_prompt,
                temperature=self._config.temperature,
                json_mode=True,
            )

        log_llm_result(
            "EntityTyper",
            response,
            str(entity.entity_id),
            metadata={
                "entity_name": entity.name,
                "domain": domain.value,
                "examples_count": len(examples),
            },
        )

        result = self._parse_typing_response(response, entity)

        if result.confidence_score < self._config.confidence_threshold:
            result.human_review_required = True

        return result

    def _parse_typing_response(
        self,
        response: str,
        entity: TypedEntity,
    ) -> EntityTypingResult:
        """Parse LLM typing response."""
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return EntityTypingResult(
                entity_id=entity.entity_id,
                entity_name=entity.name,
                entity_type=EntityType.OTHER,
                confidence_score=0.0,
                reasoning=f"Failed to parse LLM response: {response[:100]}",
                human_review_required=True,
            )

        entity_type_str = data.get("entity_type", "Other")
        confidence = data.get("confidence_score", 0.5)

        try:
            entity_type = EntityType(entity_type_str)
        except ValueError:
            entity_type = EntityType.OTHER

        alternative_types = []
        for alt in data.get("alternative_types", []):
            try:
                alt_type = EntityType(alt.get("type", "Other"))
                alt_score = alt.get("score", 0.0)
                alternative_types.append((alt_type, alt_score))
            except ValueError:
                continue

        alternative_types.sort(key=lambda x: x[1], reverse=True)

        domain_str = data.get("detected_domain", "")
        try:
            detected_domain = DomainType(domain_str) if domain_str else None
        except ValueError:
            detected_domain = None

        return EntityTypingResult(
            entity_id=entity.entity_id,
            entity_name=entity.name,
            entity_type=entity_type,
            confidence_score=confidence,
            reasoning=data.get("reasoning", "No reasoning provided"),
            alternative_types=alternative_types[:3],
            domain=detected_domain,
            human_review_required=False,
        )

    def add_few_shot_example(
        self,
        name: str,
        entity_type: EntityType,
        context: str,
        domain: DomainType | None = None,
    ) -> None:
        """Add a custom few-shot example.

        Args:
            name: Entity name.
            entity_type: Correct entity type.
            context: Context for disambiguation.
            domain: Optional domain for domain-specific example.
        """
        example = FewShotExample(name=name, entity_type=entity_type, context=context)
        domain_str = domain.value if domain else None
        self._example_bank.add_example(example, domain=domain_str)

    def register_custom_template(self, name: str, template: PromptTemplate) -> None:
        """Register a custom prompt template.

        Args:
            name: Template name.
            template: Prompt template to register.
        """
        self._template_registry.register_template(name, template)


import asyncio
