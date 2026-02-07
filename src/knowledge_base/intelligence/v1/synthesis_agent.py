"""Map-Reduce recursive summarization agent."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_base.clients import AsyncLLMClient
from knowledge_base.persistence.v1.schema import Community, Edge, Entity


class SynthesisConfig(BaseSettings):
    """Synthesis configuration."""

    model_config = SettingsConfigDict()

    max_tokens_per_report: int = 2000
    include_edge_fidelity: bool = True


class MicroReport(BaseModel):
    """Micro-report for individual community."""

    community_id: UUID = Field(..., description="Community ID")
    summary: str = Field(..., description="Community summary")
    key_entities: list[str] = Field(..., description="Key entity names")
    entity_count: int = Field(..., description="Number of entities")
    key_relationships: list[str] = Field(
        default_factory=list, description="Key relationships"
    )

    @field_validator("key_entities", "key_relationships", mode="before")
    @classmethod
    def stringify_items(cls, v):
        """Ensure all items in lists are strings."""
        if not isinstance(v, list):
            return v

        processed = []
        for item in v:
            if isinstance(item, dict):
                # Convert dict to a readable string representation
                items = []
                for key, value in item.items():
                    items.append(f"{key}: {value}")
                processed.append(" | ".join(items))
            else:
                processed.append(str(item))
        return processed


class MacroReport(BaseModel):
    """Macro-report for higher-level synthesis."""

    community_id: UUID = Field(..., description="Community ID")
    summary: str = Field(..., description="Strategic synthesis")
    child_reports: list[UUID] = Field(..., description="Child report IDs")
    thematic_focus: list[str] = Field(
        default_factory=list, description="Thematic focuses"
    )


class SynthesisAgent:
    """Agent for map-reduce recursive summarization."""

    def __init__(
        self,
        gateway: AsyncLLMClient,
        config: SynthesisConfig | None = None,
    ) -> None:
        """Initialize synthesis agent.

        Args:
            gateway: LLM gateway client.
            config: Synthesis configuration.
        """
        self._gateway = gateway
        self._config = config or SynthesisConfig()

    async def generate_micro_report(
        self,
        community: Community,
        entities: list[Entity],
        edges: list[Edge],
    ) -> MicroReport:
        """Generate micro-report for individual community.

        Args:
            community: Community to summarize.
            entities: Entities in the community.
            edges: High-confidence edges between entities.

        Returns:
            Micro-report.
        """
        entity_summaries = self._prepare_entity_summaries(entities)
        edge_summaries = (
            self._prepare_edge_summaries(edges)
            if self._config.include_edge_fidelity
            else []
        )

        system_prompt = self._get_micro_report_system_prompt()
        user_prompt = self._get_micro_report_user_prompt(
            community,
            entity_summaries,
            edge_summaries,
        )

        response = await self._gateway.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=self._config.max_tokens_per_report,
        )

        return self._parse_micro_report(response, community.id, entities)

    def _prepare_entity_summaries(self, entities: list[Entity]) -> str:
        """Prepare entity summaries for prompt.

        Args:
            entities: List of entities.

        Returns:
            Formatted entity summaries.
        """
        summaries = []
        for entity in entities:
            summary = f"- {entity.name}"
            if entity.entity_type:
                summary += f" ({entity.entity_type})"
            if entity.description:
                summary += f": {entity.description}"
            if entity.properties:
                summary += f" [Props: {entity.properties}]"
            summaries.append(summary)
        return "\n".join(summaries)

    def _prepare_edge_summaries(self, edges: list[Edge]) -> str:
        """Prepare edge summaries for prompt.

        Args:
            edges: List of edges.

        Returns:
            Formatted edge summaries.
        """
        summaries = []
        for edge in edges:
            summary = f"- {edge.source_id} --[{edge.edge_type}]--> {edge.target_id}"
            if edge.properties:
                summary += f" [Props: {edge.properties}]"
            if edge.confidence < 1.0:
                summary += f" (confidence: {edge.confidence:.2f})"
            summaries.append(summary)
        return "\n".join(summaries)

    def _get_micro_report_system_prompt(self) -> str:
        """Get micro-report system prompt."""
        return """You are an expert knowledge synthesist. Your task is to generate a detailed, factual micro-report summarizing a community of entities.

Guidelines:
1. Be comprehensive and factually accurate
2. Include all key entities and their roles
3. Detail specific relationships with confidence scores
4. Focus on what is explicitly known from the data
5. Maintain fidelity to raw relationship data
6. Preserve temporal and contextual information
7. Include all specific technical details, values, dates, and specifications from entity properties and edge properties
8. Output in JSON format:
{
  "summary": "detailed factual summary",
  "key_entities": ["entity1", "entity2"],
  "key_relationships": ["relationship descriptions"],
  "thematic_focus": ["theme1", "theme2"]
}

Avoid generalizations. Be specific and evidence-based."""

    def _get_micro_report_user_prompt(
        self,
        community: Community,
        entity_summaries: str,
        edge_summaries: str,
    ) -> str:
        """Get micro-report user prompt."""
        prompt = f"""Generate a micro-report for the following community:

Community: {community.name}
Level: {community.level}
Entity Count: {community.entity_count}

Entities:
{entity_summaries}
"""

        if edge_summaries:
            prompt += f"\nRelationships (High-Confidence Edges):\n{edge_summaries}\n"

        prompt += "\nProvide your analysis in JSON format."

        return prompt

    def _parse_micro_report(
        self,
        response: str,
        community_id: UUID,
        entities: list[Entity],
    ) -> MicroReport:
        """Parse micro-report response.

        Args:
            response: LLM JSON response.
            community_id: Community ID.
            entities: Community entities.

        Returns:
            Parsed micro-report.
        """
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return MicroReport(
                community_id=community_id,
                summary=response,
                key_entities=[e.name for e in entities],
                entity_count=len(entities),
                key_relationships=[],
            )

        return MicroReport(
            community_id=community_id,
            summary=data.get("summary", response),
            key_entities=data.get("key_entities", [e.name for e in entities]),
            entity_count=len(entities),
            key_relationships=data.get("key_relationships", []),
        )

    async def generate_macro_report(
        self,
        community: Community,
        child_reports: list[MicroReport],
        child_edges: list[Edge],
    ) -> MacroReport:
        """Generate macro-report from child reports.

        Args:
            community: Parent community.
            child_reports: Child micro-reports.
            child_edges: High-confidence edges between children.

        Returns:
            Macro-report.
        """
        report_summaries = self._prepare_child_report_summaries(child_reports)
        edge_summaries = (
            self._prepare_edge_summaries(child_edges)
            if self._config.include_edge_fidelity
            else []
        )

        system_prompt = self._get_macro_report_system_prompt()
        user_prompt = self._get_macro_report_user_prompt(
            community,
            report_summaries,
            edge_summaries,
        )

        response = await self._gateway.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=self._config.max_tokens_per_report,
        )

        return self._parse_macro_report(response, community.id, child_reports)

    def _prepare_child_report_summaries(self, reports: list[MicroReport]) -> str:
        """Prepare child report summaries for prompt.

        Args:
            reports: List of child reports.

        Returns:
            Formatted report summaries.
        """
        summaries = []
        for report in reports:
            summary = f"Community {report.community_id}:\n"
            summary += f"  Summary: {report.summary}\n"
            summary += f"  Key Entities: {', '.join(report.key_entities)}\n"
            if report.key_relationships:
                summary += (
                    f"  Key Relationships: {', '.join(report.key_relationships)}\n"
                )
            summaries.append(summary)
        return "\n\n".join(summaries)

    def _get_macro_report_system_prompt(self) -> str:
        """Get macro-report system prompt."""
        return """You are a strategic knowledge synthesist. Your task is to generate a high-level macro-report that synthesizes multiple child community reports into a coherent strategic overview.

Guidelines:
1. Identify cross-cutting themes and patterns
2. Synthesize relationships between communities
3. Preserve high-confidence edge information
4. Provide strategic insights while maintaining factual accuracy
5. Do not smooth over or omit specific technical details, even if they seem minor
6. Preserve all specific technical details from child reports including exact values, dates, version numbers, and technical specifications
7. Output in JSON format:
{
  "summary": "strategic synthesis",
  "thematic_focus": ["theme1", "theme2", "theme3"],
  "key_insights": ["insight1", "insight2"]
}

Focus on the "big picture" while staying grounded in the provided data."""

    def _get_macro_report_user_prompt(
        self,
        community: Community,
        report_summaries: str,
        edge_summaries: str,
    ) -> str:
        """Get macro-report user prompt."""
        prompt = f"""Generate a macro-report for the following parent community:

Community: {community.name}
Level: {community.level}

Child Community Reports:
{report_summaries}
"""

        if edge_summaries:
            prompt += f"\nCross-Community Relationships:\n{edge_summaries}\n"

        prompt += "\nProvide your strategic synthesis in JSON format."

        return prompt

    def _parse_macro_report(
        self,
        response: str,
        community_id: UUID,
        child_reports: list[MicroReport],
    ) -> MacroReport:
        """Parse macro-report response.

        Args:
            response: LLM JSON response.
            community_id: Community ID.
            child_reports: Child reports.

        Returns:
            Parsed macro-report.
        """
        import json

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return MacroReport(
                community_id=community_id,
                summary=response,
                child_reports=[r.community_id for r in child_reports],
                thematic_focus=[],
            )

        return MacroReport(
            community_id=community_id,
            summary=data.get("summary", response),
            child_reports=[r.community_id for r in child_reports],
            thematic_focus=data.get("thematic_focus", []),
        )

    async def generate_intelligence_report(
        self,
        hierarchy: dict[UUID, Community],
        entities: list[Entity],
        edges: list[Edge],
    ) -> dict[UUID, MicroReport | MacroReport]:
        """Generate complete intelligence report for hierarchy.

        Args:
            hierarchy: Community hierarchy.
            entities: All entities.
            edges: All edges.

        Returns:
            Dictionary of community IDs to reports.
        """
        leaf_communities = [c for c in hierarchy.values() if c.level == 0]
        parent_communities = [c for c in hierarchy.values() if c.level > 0]

        reports: dict[UUID, MicroReport | MacroReport] = {}

        for community in leaf_communities:
            community_entities = [e for e in entities if e.community_id == community.id]
            community_entity_ids = {e.id for e in community_entities}

            community_edges = [
                e
                for e in edges
                if e.source_id in community_entity_ids
                and e.target_id in community_entity_ids
            ]

            report = await self.generate_micro_report(
                community, community_entities, community_edges
            )
            reports[community.id] = report

        for community in parent_communities:
            child_reports = [
                reports[c.id]
                for c in hierarchy.values()
                if c.parent_id == community.id and c.id in reports
            ]

            if child_reports:
                child_ids = {c.id for c in child_reports if isinstance(c, MicroReport)}

                cross_edges = [
                    e
                    for e in edges
                    if e.source_id in child_ids and e.target_id in child_ids
                ]

                report = await self.generate_macro_report(
                    community,
                    [r for r in child_reports if isinstance(r, MicroReport)],
                    cross_edges,
                )
                reports[community.id] = report

        return reports
