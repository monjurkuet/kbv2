"""Graph endpoints for KBV2."""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from knowledge_base.routes.dependencies import get_dependencies
from knowledge_base.common.utils import entities_to_dict_list


# Create router
router = APIRouter(tags=["graph"])


@router.get("/graph/entities")
async def list_entities(
    name: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """List entities from knowledge graph."""
    deps = get_dependencies()
    if not deps.kuzu:
        raise RuntimeError("Graph store not initialized")

    entities = await deps.kuzu.search_entities(
        name=name,
        entity_type=entity_type,
        limit=limit,
    )

    return {
        "entities": entities_to_dict_list(entities),
        "count": len(entities),
    }


@router.get("/graph/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get an entity and its relationships."""
    deps = get_dependencies()
    if not deps.kuzu:
        raise RuntimeError("Graph store not initialized")

    entity = await deps.kuzu.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    relationships = await deps.kuzu.get_entity_relationships(entity_id)

    return {
        "entity": {
            "id": entity.id,
            "name": entity.name,
            "type": entity.entity_type,
            "description": entity.description,
            "domain": entity.domain,
        },
        "relationships": relationships,
    }


class CommunitySummaryRequest(BaseModel):
    """Request to generate community summaries."""

    domain: Optional[str] = None
    min_entities: int = Field(default=2, ge=1, le=100)
    max_communities: int = Field(default=20, ge=1, le=100)


@router.get("/graph/communities")
async def get_communities(
    domain: Optional[str] = None,
    level: str = Query("macro", pattern="^(macro|meso|micro)$"),
    limit: int = Query(20, ge=1, le=100),
):
    """Get community summaries from the knowledge graph.

    Requires entities to be extracted first via ingestion.
    """
    deps = get_dependencies()
    if not deps.kuzu or not deps.community_summarizer:
        raise HTTPException(status_code=503, detail="Graph store or summarizer not initialized")

    try:
        # Get all entities
        entities = await deps.kuzu.search_entities(limit=1000)

        if not entities:
            return {
                "message": "No entities found in graph. Ingest documents first to extract entities.",
                "communities": [],
                "count": 0,
            }

        # Get edges/relationships
        edges_result = await deps.kuzu.query(
            "MATCH (a)-[r]->(b) RETURN a.id as source, type(r) as relation, b.id as target LIMIT 500"
        )

        # Convert to dict format using helper function
        entities_dict = entities_to_dict_list(entities, domain_filter=domain)

        edges_dict = [
            {
                "source": row.get("source"),
                "relationship_type": row.get("relation"),
                "target": row.get("target"),
            }
            for row in edges_result.records
        ]

        # Generate summaries
        communities = await deps.community_summarizer.summarize_communities(
            entities=entities_dict,
            edges=edges_dict,
            level=level,
            max_communities=limit,
        )

        return {
            "communities": communities,
            "count": len(communities),
            "level": level,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get communities: {e}")


@router.get("/graph/communities/{community_id}")
async def get_community_details(community_id: str):
    """Get detailed information about a specific community."""
    deps = get_dependencies()
    if not deps.kuzu or not deps.community_summarizer:
        raise HTTPException(status_code=503, detail="Graph store or summarizer not initialized")

    try:
        # Get all entities
        entities = await deps.kuzu.search_entities(limit=1000)

        if not entities:
            raise HTTPException(status_code=404, detail="No communities found")

        # Get edges
        edges_result = await deps.kuzu.query(
            "MATCH (a)-[r]->(b) RETURN a.id as source, type(r) as relation, b.id as target LIMIT 500"
        )

        # Convert to dict format using helper function
        entities_dict = entities_to_dict_list(entities)

        edges_dict = [
            {
                "source": row.get("source"),
                "relationship_type": row.get("relation"),
                "target": row.get("target"),
            }
            for row in edges_result.records
        ]

        # Get community details
        community = await deps.community_summarizer.get_community_details(
            community_id=community_id,
            entities=entities_dict,
            edges=edges_dict,
        )

        if not community:
            raise HTTPException(status_code=404, detail="Community not found")

        # Get path to root
        path = await deps.community_summarizer.get_community_path_to_root(
            community_id=community_id,
            entities=entities_dict,
        )

        return {
            "community": {
                "community_id": community.community_id,
                "name": community.name,
                "level": community.level.value,
                "summary": community.summary,
                "key_entities": community.key_entities,
                "key_relationships": community.key_relationships,
                "entity_count": community.entity_count,
                "coherence_score": community.coherence_score,
                "parent_community_id": community.parent_community_id,
                "child_community_ids": community.child_community_ids,
            },
            "path_to_root": [
                {
                    "community_id": c.community_id,
                    "name": c.name,
                    "level": c.level.value,
                }
                for c in path
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get community details: {e}")
