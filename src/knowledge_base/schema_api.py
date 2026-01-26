"""Schema API endpoints for domain management."""

from fastapi import APIRouter, Depends, HTTPException
from knowledge_base.intelligence import (
    SchemaRegistry,
    EntityTypeDef,
    DomainAttribute,
    InheritanceType,
    DomainLevel,
)
from knowledge_base.intelligence.v1.domain_schema_service import DomainSchemaModel
from knowledge_base.common.dependencies import get_async_session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/schemas", tags=["schemas"])


class SchemaListItem(BaseModel):
    domain_name: str
    level: str
    entity_types: int
    parent_domain_name: Optional[str]


class EntityTypeDefResponse(BaseModel):
    type_name: str
    base_type: Optional[str] = None
    attributes: Dict[str, Dict[str, Any]]


class SchemaDetail(BaseModel):
    domain_name: str
    domain_display_name: str
    level: str
    parent_domain_name: Optional[str]
    inheritance_type: str
    entity_types: List[Dict[str, Any]]
    inherited_types: Optional[List[Dict[str, Any]]] = None


class DomainAttributeRequest(BaseModel):
    name: str
    attribute_type: str
    description: str = ""
    required: bool = False


class EntityTypeDefRequest(BaseModel):
    type_name: str
    parent_type: Optional[str] = None
    description: str = ""
    attributes: Dict[str, DomainAttributeRequest] = {}
    required_attributes: List[str] = []


class RegisterSchemaRequest(BaseModel):
    domain_name: str
    domain_display_name: str
    entity_types: List[EntityTypeDefRequest]
    parent_domain_name: Optional[str] = None
    inheritance_type: str = "extends"


async def get_schema_registry(
    db: AsyncSession = Depends(get_async_session),
) -> SchemaRegistry:
    return SchemaRegistry(db)


@router.get("/", response_model=List[SchemaListItem])
async def list_schemas(
    registry: SchemaRegistry = Depends(get_schema_registry),
) -> List[SchemaListItem]:
    """List all registered domain schemas."""
    schemas = await registry.list_schemas()
    return [
        SchemaListItem(
            domain_name=s.domain_name,
            level=s.domain_level.value
            if hasattr(s.domain_level, "value")
            else str(s.domain_level),
            entity_types=len(s.entity_types),
            parent_domain_name=s.parent_domain_name,
        )
        for s in schemas
    ]


@router.get("/{domain_name}", response_model=SchemaDetail)
async def get_schema(
    domain_name: str,
    include_inherited: bool = False,
    registry: SchemaRegistry = Depends(get_schema_registry),
) -> SchemaDetail:
    """Get schema details with optional inheritance."""
    schema = await registry.get_by_name(domain_name)

    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{domain_name}' not found")

    entity_types = [
        {
            "type_name": et.type_name,
            "parent_type": et.parent_type,
            "description": et.description,
            "attributes": {
                k: {"attribute_type": v.attribute_type, "description": v.description}
                for k, v in et.attributes.items()
            },
        }
        for et in schema.entity_types.values()
    ]

    result = SchemaDetail(
        domain_name=schema.domain_name,
        domain_display_name=schema.domain_display_name,
        level=schema.domain_level.value
        if hasattr(schema.domain_level, "value")
        else str(schema.domain_level),
        parent_domain_name=schema.parent_domain_name,
        inheritance_type=schema.inheritance_type.value
        if hasattr(schema.inheritance_type, "value")
        else str(schema.inheritance_type),
        entity_types=entity_types,
    )

    if include_inherited:
        inherited = []
        for et_name in schema.entity_types:
            inherited_list = await registry.get_inherited_entity_types(
                domain_name, et_name
            )
            for parent_domain, et_def in inherited_list:
                if parent_domain != domain_name:
                    inherited.append(
                        {
                            "type_name": et_def.type_name,
                            "source_domain": parent_domain,
                            "attributes": {
                                k: {"attribute_type": v.attribute_type}
                                for k, v in et_def.attributes.items()
                            },
                        }
                    )
        result.inherited_types = inherited

    return result


@router.post("/", response_model=Dict[str, str])
async def register_schema(
    request: RegisterSchemaRequest,
    registry: SchemaRegistry = Depends(get_schema_registry),
) -> Dict[str, str]:
    """Register a new domain schema."""
    entity_types = {
        et.type_name: EntityTypeDef(
            type_name=et.type_name,
            parent_type=et.parent_type,
            description=et.description,
            attributes={
                k: DomainAttribute(
                    name=k,
                    attribute_type=v.attribute_type,
                    description=v.description,
                    required=v.required,
                )
                for k, v in et.attributes.items()
            },
            required_attributes=et.required_attributes,
        )
        for et in request.entity_types
    }

    from knowledge_base.intelligence.v1.domain_schema_service import (
        DomainSchemaCreate,
        DomainLevel,
    )

    schema_create = DomainSchemaCreate(
        domain_name=request.domain_name,
        domain_display_name=request.domain_display_name,
        parent_domain_name=request.parent_domain_name,
        domain_level=DomainLevel.PRIMARY,
        inheritance_type=InheritanceType(request.inheritance_type)
        if request.inheritance_type
        else InheritanceType.EXTENDS,
        entity_types=entity_types,
    )

    await registry.register(schema_create)

    return {"message": f"Schema '{request.domain_name}' registered successfully"}


@router.delete("/{domain_name}", response_model=Dict[str, str])
async def delete_schema(
    domain_name: str, registry: SchemaRegistry = Depends(get_schema_registry)
) -> Dict[str, str]:
    """Delete a domain schema."""
    if domain_name in ["GENERAL", "ROOT"]:
        raise HTTPException(status_code=400, detail="Cannot delete root schema")

    deleted = await registry.delete(domain_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Schema '{domain_name}' not found")

    return {"message": f"Schema '{domain_name}' deleted"}


@router.get("/{domain_name}/entity-types", response_model=List[Dict[str, Any]])
async def get_entity_types(
    domain_name: str, registry: SchemaRegistry = Depends(get_schema_registry)
) -> List[Dict[str, Any]]:
    """Get entity types for a domain."""
    schema = await registry.get_by_name(domain_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{domain_name}' not found")

    return [
        {
            "type_name": et.type_name,
            "parent_type": et.parent_type,
            "description": et.description,
            "attributes": {
                k: {"attribute_type": v.attribute_type, "description": v.description}
                for k, v in et.attributes.items()
            },
            "required_attributes": et.required_attributes,
        }
        for et in schema.entity_types.values()
    ]


@router.get("/{domain_name}/entity-types/{entity_type}", response_model=Dict[str, Any])
async def get_entity_type(
    domain_name: str,
    entity_type: str,
    include_inherited: bool = False,
    registry: SchemaRegistry = Depends(get_schema_registry),
) -> Dict[str, Any]:
    """Get a specific entity type definition."""
    schema = await registry.get_by_name(domain_name)
    if not schema:
        raise HTTPException(status_code=404, detail=f"Schema '{domain_name}' not found")

    entity_types = await registry.get_inherited_entity_types(domain_name, entity_type)
    if not entity_types:
        raise HTTPException(
            status_code=404,
            detail=f"Entity type '{entity_type}' not found in domain '{domain_name}'",
        )

    result = {
        "type_name": entity_types[0][1].type_name,
        "parent_type": entity_types[0][1].parent_type,
        "description": entity_types[0][1].description,
        "attributes": {
            k: {"attribute_type": v.attribute_type, "description": v.description}
            for k, v in entity_types[0][1].attributes.items()
        },
        "required_attributes": entity_types[0][1].required_attributes,
    }

    if include_inherited:
        result["inheritance_chain"] = [
            {"domain": domain, "type_name": et.type_name} for domain, et in entity_types
        ]

    return result
