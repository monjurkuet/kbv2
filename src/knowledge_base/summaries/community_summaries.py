"""Enhanced community summary functionality with multi-level hierarchy for KBV2.

This module provides classes and functions for generating and managing
community summaries at multiple hierarchical levels: macro, meso, micro,
and nano levels.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import json


class HierarchyLevel(str, Enum):
    """Community hierarchy levels.

    Attributes:
        MACRO: High-level themes and topics representing broad categories.
        MESO: Sub-topics that fall under macro-level themes.
        MICRO: Specific entities within meso-level communities.
        NANO: Individual atomic entities in the knowledge graph.
    """

    MACRO = "macro"
    MESO = "meso"
    MICRO = "micro"
    NANO = "nano"


class CommunitySummary(BaseModel):
    """Enhanced community summary with multi-level hierarchy.

    Attributes:
        community_id: Unique identifier for the community.
        name: Human-readable name for the community.
        level: The hierarchy level of this community.
        summary: Detailed text summary of the community content.
        key_entities: List of key entity names in this community.
        key_relationships: List of key relationship descriptions.
        parent_community_id: ID of the parent community, if any.
        child_community_ids: List of child community IDs.
        entity_count: Number of entities in this community.
        coherence_score: Score indicating how coherent the community is (0.0-1.0).
        generated_at: Timestamp when this summary was generated.
    """

    community_id: str
    name: str
    level: HierarchyLevel
    summary: str
    key_entities: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list)
    parent_community_id: Optional[str] = None
    child_community_ids: List[str] = Field(default_factory=list)
    entity_count: int = 0
    coherence_score: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class CommunityNamingResult(BaseModel):
    """Result from LLM-based community naming.

    Attributes:
        suggested_name: The suggested name for the community.
        description: Brief description of the community.
        key_themes: List of key themes identified in the community.
        confidence: Confidence score for the naming suggestion (0.0-1.0).
    """

    suggested_name: str
    description: str
    key_themes: List[str]
    confidence: float


class MultiLevelSummary(BaseModel):
    """Complete multi-level community summary.

    Attributes:
        document_id: ID of the source document.
        macro_communities: List of macro-level community summaries.
        meso_communities: List of meso-level community summaries.
        micro_communities: List of micro-level community summaries.
        hierarchy_tree: Dictionary representing the hierarchy structure.
    """

    document_id: str
    macro_communities: List[CommunitySummary] = Field(default_factory=list)
    meso_communities: List[CommunitySummary] = Field(default_factory=list)
    micro_communities: List[CommunitySummary] = Field(default_factory=list)
    hierarchy_tree: Dict[str, Any] = Field(default_factory=dict)


class CommunitySummarizer:
    """Enhanced community summarizer with multi-level hierarchy.

    This class provides functionality to generate hierarchical community
    summaries from entities and relationships in the knowledge graph.

    Attributes:
        llm: Optional LLM client for generating intelligent names.
    """

    def __init__(self, llm_client: Any = None):
        """Initialize the community summarizer.

        Args:
            llm_client: Optional LLM client for generating community names.
        """
        self.llm = llm_client

    async def generate_summaries(
        self,
        communities: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        document_id: str = "",
    ) -> MultiLevelSummary:
        """Generate multi-level community summaries from entities and edges.

        Args:
            communities: List of community definitions.
            entities: List of entity dictionaries with their properties.
            edges: List of edge dictionaries representing relationships.
            document_id: ID of the source document.

        Returns:
            MultiLevelSummary containing all hierarchy levels.
        """
        community_entities = self._group_entities_by_community(entities)
        community_edges = self._group_edges_by_community(edges)

        macro_communities = await self._create_macro_communities(
            community_entities, community_edges, communities
        )
        meso_communities = await self._create_meso_communities(
            community_entities, community_edges, communities
        )
        micro_communities = await self._create_micro_communities(
            community_entities, community_edges, communities
        )

        await self._generate_community_names(
            macro_communities, level=HierarchyLevel.MACRO
        )
        await self._generate_community_names(
            meso_communities, level=HierarchyLevel.MESO
        )
        await self._generate_community_names(
            micro_communities, level=HierarchyLevel.MICRO
        )

        hierarchy_tree = self._build_hierarchy_tree(
            macro_communities, meso_communities, micro_communities
        )

        return MultiLevelSummary(
            document_id=document_id,
            macro_communities=macro_communities,
            meso_communities=meso_communities,
            micro_communities=micro_communities,
            hierarchy_tree=hierarchy_tree,
        )

    def _group_entities_by_community(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group entities by their community ID.

        Args:
            entities: List of entity dictionaries.

        Returns:
            Dictionary mapping community IDs to lists of entities.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities:
            community_id = entity.get("community_id") or "unassigned"
            if community_id not in grouped:
                grouped[community_id] = []
            grouped[community_id].append(entity)
        return grouped

    def _group_edges_by_community(
        self, edges: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group edges by their associated community ID.

        Args:
            edges: List of edge dictionaries.

        Returns:
            Dictionary mapping community IDs to lists of edges.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edges:
            community_id = edge.get("community_id") or "unassigned"
            if community_id not in grouped:
                grouped[community_id] = []
            grouped[community_id].append(edge)
        return grouped

    async def _create_macro_communities(
        self,
        community_entities: Dict[str, List[Dict[str, Any]]],
        community_edges: Dict[str, List[Dict[str, Any]]],
        communities: List[Dict[str, Any]],
    ) -> List[CommunitySummary]:
        """Create macro-level communities representing high-level themes.

        Macro communities are formed by grouping entities based on their
        primary entity types and semantic themes.

        Args:
            community_entities: Entities grouped by community.
            community_edges: Edges grouped by community.
            communities: List of community definitions.

        Returns:
            List of macro-level CommunitySummary objects.
        """
        macro_summaries: List[CommunitySummary] = []
        type_groups: Dict[str, List[Dict[str, Any]]] = {}

        for comm_id, ents in community_entities.items():
            if not ents:
                continue
            for entity in ents:
                primary_type = entity.get("entity_type", "Unknown")
                if primary_type not in type_groups:
                    type_groups[primary_type] = []
                type_groups[primary_type].append(entity)

        edge_relationships: Dict[str, List[str]] = {}
        for comm_id, edges_list in community_edges.items():
            for edge in edges_list:
                rel_type = edge.get("relationship_type", "related")
                if rel_type not in edge_relationships:
                    edge_relationships[rel_type] = []
                edge_relationships[rel_type].append(
                    f"{edge.get('source', '')} -> {edge.get('target', '')}"
                )

        for i, (theme_type, type_ents) in enumerate(type_groups.items()):
            key_rels = edge_relationships.get(theme_type, [])[:5]
            macro_summary = CommunitySummary(
                community_id=f"macro-{i}",
                name=f"{theme_type} Cluster",
                level=HierarchyLevel.MACRO,
                summary=self._summarize_entities(type_ents),
                key_entities=[
                    e.get("name", "") for e in type_ents[:10] if e.get("name")
                ],
                key_relationships=key_rels,
                entity_count=len(type_ents),
            )
            macro_summaries.append(macro_summary)

        return macro_summaries

    async def _create_meso_communities(
        self,
        community_entities: Dict[str, List[Dict[str, Any]]],
        community_edges: Dict[str, List[Dict[str, Any]]],
        communities: List[Dict[str, Any]],
    ) -> List[CommunitySummary]:
        """Create meso-level communities representing sub-topics.

        Meso communities are formed by analyzing the structure of entities
        within each original community and identifying sub-themes.

        Args:
            community_entities: Entities grouped by community.
            community_edges: Edges grouped by community.
            communities: List of community definitions.

        Returns:
            List of meso-level CommunitySummary objects.
        """
        meso_summaries: List[CommunitySummary] = []

        for i, (comm_id, ents) in enumerate(community_entities.items()):
            if len(ents) < 2:
                continue

            edges_list = community_edges.get(comm_id, [])
            key_rels = [
                f"{e.get('source', '')} -> {e.get('target', '')}"
                for e in edges_list[:5]
            ]

            ent_names = [e.get("name", f"Entity-{j}") for j, e in enumerate(ents[:5])]
            entity_types = list(set(e.get("entity_type", "Unknown") for e in ents[:5]))

            meso_summary = CommunitySummary(
                community_id=f"meso-{comm_id}",
                name=f"Sub-topic {i + 1}: {entity_types[0] if entity_types else 'Mixed'}",
                level=HierarchyLevel.MESO,
                summary=self._summarize_entities(ents),
                key_entities=ent_names,
                key_relationships=key_rels,
                parent_community_id=None,
                entity_count=len(ents),
            )
            meso_summaries.append(meso_summary)

        return meso_summaries

    async def _create_micro_communities(
        self,
        community_entities: Dict[str, List[Dict[str, Any]]],
        community_edges: Dict[str, List[Dict[str, Any]]],
        communities: List[Dict[str, Any]],
    ) -> List[CommunitySummary]:
        """Create micro-level communities for specific entities.

        Micro communities represent individual entities or small groups
        of closely related entities.

        Args:
            community_entities: Entities grouped by community.
            community_edges: Edges grouped by community.
            communities: List of community definitions.

        Returns:
            List of micro-level CommunitySummary objects.
        """
        micro_summaries: List[CommunitySummary] = []
        entity_counter = 0

        for comm_id, ents in community_entities.items():
            for j, entity in enumerate(ents[:30]):
                entity_id = entity.get("id", f"micro-{entity_counter}")
                entity_counter += 1

                edges_list = community_edges.get(comm_id, [])
                related_edges = [
                    e
                    for e in edges_list
                    if e.get("source") == entity.get("name")
                    or e.get("target") == entity.get("name")
                ]
                related_entities = [
                    e.get("target")
                    if e.get("source") == entity.get("name")
                    else e.get("source")
                    for e in related_edges[:3]
                ]

                micro_summary = CommunitySummary(
                    community_id=f"micro-{entity_id}",
                    name=entity.get("name", f"Entity {entity_counter}"),
                    level=HierarchyLevel.MICRO,
                    summary=entity.get(
                        "description",
                        f"Entity of type {entity.get('entity_type', 'Unknown')}",
                    ),
                    key_entities=[entity.get("name", "")] if entity.get("name") else [],
                    key_relationships=related_entities,
                    entity_count=1,
                    coherence_score=1.0,
                )
                micro_summaries.append(micro_summary)

        return micro_summaries[:100]

    async def _generate_community_names(
        self, communities: List[CommunitySummary], level: HierarchyLevel
    ) -> None:
        """Generate LLM-based names for communities.

        Args:
            communities: List of communities to name.
            level: The hierarchy level of these communities.
        """
        if not self.llm:
            return

        for community in communities:
            prompt = self._create_naming_prompt(community, level)
            try:
                response = await self.llm.complete(prompt)
                result = json.loads(response)
                if result.get("name"):
                    community.name = result["name"]
                if result.get("confidence"):
                    community.coherence_score = float(result["confidence"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    def _create_naming_prompt(
        self, community: CommunitySummary, level: HierarchyLevel
    ) -> str:
        """Create a prompt for generating community names.

        Args:
            community: The community to generate a name for.
            level: The hierarchy level of the community.

        Returns:
            A formatted prompt string for LLM naming.
        """
        level_desc = {
            HierarchyLevel.MACRO: "broad thematic category",
            HierarchyLevel.MESO: "sub-topic or theme",
            HierarchyLevel.MICRO: "specific entity or concept",
            HierarchyLevel.NANO: "atomic entity",
        }.get(level, "category")

        return f"""Generate a concise name (2-5 words) for this {level_desc}:

Summary: {community.summary}
Key entities: {", ".join(community.key_entities[:5])}
Entity count: {community.entity_count}

Respond with a JSON object with keys 'name' (string) and 'confidence' (number 0.0-1.0).
Example: {{"name": "Machine Learning Systems", "confidence": 0.85}}"""

    def _summarize_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Generate a brief summary of a list of entities.

        Args:
            entities: List of entity dictionaries.

        Returns:
            A string summarizing the entities.
        """
        if not entities:
            return "No entities in this community."

        types = set(e.get("entity_type", "Unknown") for e in entities)
        names = [e.get("name", "") for e in entities[:5] if e.get("name")]

        type_str = ", ".join(sorted(types))
        name_str = ", ".join(names) if names else "Unnamed entities"

        return f"Contains {len(entities)} entities of types: {type_str}. Key entities: {name_str}"

    def _build_hierarchy_tree(
        self,
        macro: List[CommunitySummary],
        meso: List[CommunitySummary],
        micro: List[CommunitySummary],
    ) -> Dict[str, Any]:
        """Build hierarchy tree structure linking all levels.

        Args:
            macro: List of macro-level communities.
            meso: List of meso-level communities.
            micro: List of micro-level communities.

        Returns:
            Dictionary representing the hierarchy tree structure.
        """
        tree: Dict[str, Any] = {
            "type": "hierarchy",
            "levels": {"macro": len(macro), "meso": len(meso), "micro": len(micro)},
            "nodes": {},
            "edges": [],
        }

        for comm in macro:
            tree["nodes"][comm.community_id] = {
                "name": comm.name,
                "level": comm.level.value,
                "children": [],
                "entity_count": comm.entity_count,
            }

        macro_map: Dict[str, CommunitySummary] = {c.community_id: c for c in macro}

        for comm in meso:
            parent_id: Optional[str] = None
            if macro:
                best_parent = None
                best_score = 0
                for macro_comm in macro:
                    score = self._calculate_parent_score(comm, macro_comm)
                    if score > best_score:
                        best_score = score
                        best_parent = macro_comm.community_id
                parent_id = best_parent

            comm.parent_community_id = parent_id
            tree["nodes"][comm.community_id] = {
                "name": comm.name,
                "level": comm.level.value,
                "parent": parent_id,
                "children": [],
                "entity_count": comm.entity_count,
            }
            if parent_id and parent_id in tree["nodes"]:
                tree["nodes"][parent_id]["children"].append(comm.community_id)
                tree["edges"].append({"source": parent_id, "target": comm.community_id})

        for comm in micro:
            parent_id: Optional[str] = None
            if meso:
                best_parent = None
                best_score = 0
                for meso_comm in meso:
                    score = self._calculate_parent_score(comm, meso_comm)
                    if score > best_score:
                        best_score = score
                        best_parent = meso_comm.community_id
                parent_id = best_parent

            comm.parent_community_id = parent_id
            tree["nodes"][comm.community_id] = {
                "name": comm.name,
                "level": comm.level.value,
                "parent": parent_id,
                "children": [],
                "entity_count": comm.entity_count,
            }
            if parent_id and parent_id in tree["nodes"]:
                tree["nodes"][parent_id]["children"].append(comm.community_id)
                tree["edges"].append({"source": parent_id, "target": comm.community_id})

        return tree

    def _calculate_parent_score(
        self, child: CommunitySummary, parent: CommunitySummary
    ) -> float:
        """Calculate a score for parent-child relationship.

        Args:
            child: The potential child community.
            parent: The potential parent community.

        Returns:
            Score representing how well they match (0.0-1.0).
        """
        score = 0.0
        child_entities_lower = set(e.lower() for e in child.key_entities)
        parent_entities_lower = set(e.lower() for e in parent.key_entities)
        overlap = child_entities_lower & parent_entities_lower
        score += len(overlap) * 0.3
        score += min(child.entity_count / max(parent.entity_count, 1), 1.0) * 0.3
        score += 0.4
        return min(score, 1.0)

    def get_summary_by_level(
        self, summary: MultiLevelSummary, level: HierarchyLevel
    ) -> List[CommunitySummary]:
        """Get communities at a specific hierarchy level.

        Args:
            summary: The multi-level summary to query.
            level: The desired hierarchy level.

        Returns:
            List of communities at the specified level.
        """
        level_map = {
            HierarchyLevel.MACRO: summary.macro_communities,
            HierarchyLevel.MESO: summary.meso_communities,
            HierarchyLevel.MICRO: summary.micro_communities,
        }
        return level_map.get(level, [])

    def get_community_path(
        self, summary: MultiLevelSummary, community_id: str
    ) -> List[CommunitySummary]:
        """Get the path from a community to the root.

        Args:
            summary: The multi-level summary to query.
            community_id: The ID of the starting community.

        Returns:
            List of communities from the given community to the root.
        """
        all_communities = (
            summary.macro_communities
            + summary.meso_communities
            + summary.micro_communities
        )
        comm_map = {c.community_id: c for c in all_communities}

        path: List[CommunitySummary] = []
        current_id = community_id

        while current_id and current_id in comm_map:
            comm = comm_map[current_id]
            path.append(comm)
            current_id = comm.parent_community_id

        return path
