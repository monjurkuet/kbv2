"""Domain-aware entity schema service with inheritance support."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import relationship

from knowledge_base.persistence.v1.schema import Base


class DomainLevel(str, Enum):
    """Domain hierarchy levels."""

    ROOT = "root"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


class InheritanceType(str, Enum):
    """Types of schema inheritance."""

    EXTENDS = "extends"
    OVERRIDES = "overrides"
    COMPOSES = "composes"


class SchemaValidationMode(str, Enum):
    """Schema validation modes."""

    STRICT = "strict"
    LAX = "lax"
    ADAPTIVE = "adaptive"


class DomainAttribute(BaseModel):
    """Domain-specific attribute definition."""

    name: str = Field(..., description="Attribute name")
    attribute_type: str = Field(..., description="Python/Pydantic type string")
    description: str = Field(default="", description="Attribute description")
    required: bool = Field(default=False, description="Whether attribute is required")
    default_value: Optional[Any] = Field(
        None, description="Default value if not provided"
    )
    validation_rules: dict[str, Any] = Field(
        default_factory=dict, description="Validation rules for this attribute"
    )
    domain_specific: bool = Field(
        default=True, description="Whether this is a domain-specific attribute"
    )


class EntityTypeDef(BaseModel):
    """Entity type definition with domain-specific attributes."""

    type_name: str = Field(..., description="Name of the entity type")
    parent_type: Optional[str] = Field(
        None, description="Parent entity type for inheritance"
    )
    description: str = Field(default="", description="Entity type description")
    attributes: dict[str, DomainAttribute] = Field(
        default_factory=dict, description="Domain-specific attributes"
    )
    required_attributes: list[str] = Field(
        default_factory=list, description="Required attribute names"
    )
    validation_mode: SchemaValidationMode = Field(
        default=SchemaValidationMode.ADAPTIVE,
        description="Validation mode for this type",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def get_attribute(self, name: str) -> Optional[DomainAttribute]:
        """Get attribute by name."""
        return self.attributes.get(name)

    def get_all_attributes(
        self, include_inherited: bool = True
    ) -> dict[str, DomainAttribute]:
        """Get all attributes, optionally including inherited ones."""
        return self.attributes.copy()


class DomainSchema(Base):
    """Domain schema model with inheritance support."""

    __tablename__ = "domain_schemas"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    domain_name = Column(String(100), nullable=False, unique=True, index=True)
    domain_display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    parent_domain_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("domain_schemas.id"),
        nullable=True,
        index=True,
    )
    domain_level = Column(String(50), nullable=False, default=DomainLevel.PRIMARY)
    inheritance_type = Column(
        String(50), nullable=False, default=InheritanceType.EXTENDS
    )
    entity_types = Column(Text, nullable=False)  # JSON serialized EntityTypeDef dict
    validation_mode = Column(
        String(50), nullable=False, default=SchemaValidationMode.ADAPTIVE
    )
    is_active = Column(String(1), nullable=False, default="Y")
    version = Column(String(20), nullable=False, default="1.0.0")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    parent_domain = relationship(
        "DomainSchema", remote_side=[id], backref="child_domains"
    )

    __table_args__ = (
        Index("idx_domain_schema_parent", "parent_domain_id"),
        Index("idx_domain_schema_level", "domain_level"),
    )


class DomainSchemaModel(BaseModel):
    """Pydantic model for DomainSchema."""

    id: UUID = Field(..., description="Schema ID")
    domain_name: str = Field(..., description="Domain name (unique identifier)")
    domain_display_name: str = Field(..., description="Human-readable domain name")
    description: Optional[str] = Field(None, description="Domain description")
    parent_domain_id: Optional[UUID] = Field(
        None, description="Parent domain for inheritance"
    )
    parent_domain_name: Optional[str] = Field(None, description="Parent domain name")
    domain_level: DomainLevel = Field(..., description="Domain hierarchy level")
    inheritance_type: InheritanceType = Field(..., description="Type of inheritance")
    entity_types: dict[str, EntityTypeDef] = Field(
        default_factory=dict, description="Entity type definitions"
    )
    validation_mode: SchemaValidationMode = Field(..., description="Validation mode")
    is_active: bool = Field(..., description="Whether schema is active")
    version: str = Field(..., description="Schema version")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        from_attributes = True


class DomainSchemaCreate(BaseModel):
    """Model for creating a new domain schema."""

    domain_name: str = Field(..., description="Domain name (unique identifier)")
    domain_display_name: str = Field(..., description="Human-readable domain name")
    description: Optional[str] = Field(None, description="Domain description")
    parent_domain_name: Optional[str] = Field(
        None, description="Parent domain name for inheritance"
    )
    domain_level: DomainLevel = Field(
        default=DomainLevel.PRIMARY, description="Domain hierarchy level"
    )
    inheritance_type: InheritanceType = Field(
        default=InheritanceType.EXTENDS, description="Type of inheritance"
    )
    entity_types: dict[str, EntityTypeDef] = Field(
        default_factory=dict, description="Entity type definitions"
    )
    validation_mode: SchemaValidationMode = Field(
        default=SchemaValidationMode.ADAPTIVE, description="Validation mode"
    )
    version: str = Field(default="1.0.0", description="Schema version")


class DomainSchemaUpdate(BaseModel):
    """Model for updating an existing domain schema."""

    domain_display_name: Optional[str] = Field(
        None, description="Human-readable domain name"
    )
    description: Optional[str] = Field(None, description="Domain description")
    parent_domain_name: Optional[str] = Field(
        None, description="Parent domain name for inheritance"
    )
    domain_level: Optional[DomainLevel] = Field(
        None, description="Domain hierarchy level"
    )
    inheritance_type: Optional[InheritanceType] = Field(
        None, description="Type of inheritance"
    )
    entity_types: Optional[dict[str, EntityTypeDef]] = Field(
        None, description="Entity type definitions"
    )
    validation_mode: Optional[SchemaValidationMode] = Field(
        None, description="Validation mode"
    )
    is_active: Optional[bool] = Field(None, description="Whether schema is active")
    version: Optional[str] = Field(None, description="Schema version")


class SchemaRegistry:
    """Registry for managing domain schemas with inheritance support."""

    def __init__(self, db: AsyncSession) -> None:
        """Initialize schema registry.

        Args:
            db: Database session for persistence operations.
        """
        self.db = db
        self._cache: dict[str, DomainSchemaModel] = {}

    async def register(self, schema: DomainSchemaCreate) -> DomainSchemaModel:
        """Register a new domain schema.

        Args:
            schema: Domain schema to register.

        Returns:
            Registered domain schema model.
        """
        import json

        parent_domain_id = None
        if schema.parent_domain_name:
            parent_schema = await self.get_by_name(schema.parent_domain_name)
            if parent_schema:
                parent_domain_id = parent_schema.id

        db_schema = DomainSchema(
            domain_name=schema.domain_name,
            domain_display_name=schema.domain_display_name,
            description=schema.description,
            parent_domain_id=parent_domain_id,
            domain_level=schema.domain_level.value,
            inheritance_type=schema.inheritance_type.value,
            entity_types=json.dumps(
                {name: et.model_dump() for name, et in schema.entity_types.items()}
            ),
            validation_mode=schema.validation_mode.value,
            version=schema.version,
        )

        self.db.add(db_schema)
        await self.db.commit()
        await self.db.refresh(db_schema)

        model = await self._db_to_model(db_schema)
        self._cache[model.domain_name] = model

        return model

    async def get_by_name(self, domain_name: str) -> Optional[DomainSchemaModel]:
        """Get domain schema by name.

        Args:
            domain_name: Name of the domain schema.

        Returns:
            Domain schema model or None if not found.
        """
        if domain_name in self._cache:
            return self._cache[domain_name]

        from sqlalchemy import select

        query = select(DomainSchema).where(DomainSchema.domain_name == domain_name)
        result = await self.db.execute(query)
        db_schema = result.scalar_one_or_none()

        if not db_schema:
            return None

        model = await self._db_to_model(db_schema)
        self._cache[model.domain_name] = model

        return model

    async def get_by_id(self, schema_id: UUID) -> Optional[DomainSchemaModel]:
        """Get domain schema by ID.

        Args:
            schema_id: UUID of the domain schema.

        Returns:
            Domain schema model or None if not found.
        """
        from sqlalchemy import select

        query = select(DomainSchema).where(DomainSchema.id == schema_id)
        result = await self.db.execute(query)
        db_schema = result.scalar_one_or_none()

        if not db_schema:
            return None

        return await self._db_to_model(db_schema)

    async def list_schemas(
        self,
        active_only: bool = True,
        level: Optional[DomainLevel] = None,
    ) -> list[DomainSchemaModel]:
        """List all domain schemas.

        Args:
            active_only: Whether to include only active schemas.
            level: Optional level filter.

        Returns:
            List of domain schema models.
        """
        from sqlalchemy import select

        query = select(DomainSchema)

        if active_only:
            query = query.where(DomainSchema.is_active == True)

        if level:
            query = query.where(DomainSchema.domain_level == level.value)

        query = query.order_by(DomainSchema.domain_name)
        result = await self.db.execute(query)
        db_schemas = result.scalars().all()

        return [await self._db_to_model(db_schema) for db_schema in db_schemas]

    async def update(
        self, domain_name: str, update: DomainSchemaUpdate
    ) -> Optional[DomainSchemaModel]:
        """Update an existing domain schema.

        Args:
            domain_name: Name of the domain schema to update.
            update: Update data.

        Returns:
            Updated domain schema model or None if not found.
        """
        import json

        schema = await self.get_by_name(domain_name)
        if not schema:
            return None

        from sqlalchemy import select

        query = select(DomainSchema).where(DomainSchema.domain_name == domain_name)
        result = await self.db.execute(query)
        db_schema = result.scalar_one()

        if update.domain_display_name is not None:
            db_schema.domain_display_name = update.domain_display_name
        if update.description is not None:
            db_schema.description = update.description
        if update.parent_domain_name is not None:
            parent_schema = await self.get_by_name(update.parent_domain_name)
            db_schema.parent_domain_id = parent_schema.id if parent_schema else None
        if update.domain_level is not None:
            db_schema.domain_level = update.domain_level.value
        if update.inheritance_type is not None:
            db_schema.inheritance_type = update.inheritance_type.value
        if update.entity_types is not None:
            db_schema.entity_types = json.dumps(
                {name: et.model_dump() for name, et in update.entity_types.items()}
            )
        if update.validation_mode is not None:
            db_schema.validation_mode = update.validation_mode.value
        if update.is_active is not None:
            db_schema.is_active = "Y" if update.is_active else "N"
        if update.version is not None:
            db_schema.version = update.version

        await self.db.commit()
        await self.db.refresh(db_schema)

        self._cache.pop(domain_name, None)
        model = await self._db_to_model(db_schema)
        self._cache[model.domain_name] = model

        return model

    async def delete(self, domain_name: str) -> bool:
        """Delete a domain schema.

        Args:
            domain_name: Name of the domain schema to delete.

        Returns:
            True if deleted, False if not found.
        """
        schema = await self.get_by_name(domain_name)
        if not schema:
            return False

        from sqlalchemy import delete

        query = delete(DomainSchema).where(DomainSchema.domain_name == domain_name)
        await self.db.execute(query)
        await self.db.commit()

        self._cache.pop(domain_name, None)

        return True

    async def get_entity_type(
        self, domain_name: str, entity_type: str
    ) -> Optional[EntityTypeDef]:
        """Get entity type definition from a domain schema.

        Supports inheritance - will check parent domains if not found.

        Args:
            domain_name: Domain schema name.
            entity_type: Entity type name.

        Returns:
            Entity type definition or None if not found.
        """
        schema = await self.get_by_name(domain_name)
        if not schema:
            return None

        if entity_type in schema.entity_types:
            return schema.entity_types[entity_type]

        if schema.parent_domain_name:
            return await self.get_entity_type(schema.parent_domain_name, entity_type)

        return None

    async def get_inherited_entity_types(
        self, domain_name: str, entity_type: str
    ) -> list[tuple[str, EntityTypeDef]]:
        """Get entity type definition from a domain and all its parent domains.

        Args:
            domain_name: Domain schema name.
            entity_type: Entity type name.

        Returns:
            List of (domain_name, entity_type_def) tuples from inheritance chain.
        """
        schema = await self.get_by_name(domain_name)
        if not schema:
            return []

        result = []

        if entity_type in schema.entity_types:
            result.append((domain_name, schema.entity_types[entity_type]))

        if schema.parent_domain_name:
            parent_results = await self.get_inherited_entity_types(
                schema.parent_domain_name, entity_type
            )
            result.extend(parent_results)

        return result

    async def apply_inheritance(
        self, domain_name: str, entity_type: str
    ) -> EntityTypeDef:
        """Get entity type with inheritance applied.

        Combines attributes from parent and child domains according to inheritance type.

        Args:
            domain_name: Domain schema name.
            entity_type: Entity type name.

        Returns:
            Entity type definition with inherited attributes.
        """
        schema = await self.get_by_name(domain_name)
        if not schema:
            raise ValueError(f"Domain schema not found: {domain_name}")

        entity_types_inherited = await self.get_inherited_entity_types(
            domain_name, entity_type
        )

        if not entity_types_inherited:
            raise ValueError(
                f"Entity type {entity_type} not found in domain {domain_name}"
            )

        if len(entity_types_inherited) == 1:
            return entity_types_inherited[0][1]

        child_domain, child_def = entity_types_inherited[0]

        if schema.inheritance_type == InheritanceType.EXTENDS:
            for parent_domain, parent_def in reversed(entity_types_inherited[1:]):
                for attr_name, attr in parent_def.attributes.items():
                    if attr_name not in child_def.attributes:
                        child_def.attributes[attr_name] = attr
                child_def.required_attributes = list(
                    set(child_def.required_attributes + parent_def.required_attributes)
                )

        elif schema.inheritance_type == InheritanceType.OVERRIDES:
            for parent_domain, parent_def in reversed(entity_types_inherited[1:]):
                for attr_name, attr in parent_def.attributes.items():
                    child_def.attributes[attr_name] = attr
                child_def.required_attributes = list(
                    set(child_def.required_attributes + parent_def.required_attributes)
                )

        elif schema.inheritance_type == InheritanceType.COMPOSES:
            for parent_domain, parent_def in entity_types_inherited[1:]:
                for attr_name, attr in parent_def.attributes.items():
                    if attr_name not in child_def.attributes:
                        child_def.attributes[f"{parent_domain}_{attr_name}"] = attr

        return child_def

    async def _db_to_model(self, db_schema: DomainSchema) -> DomainSchemaModel:
        """Convert database model to Pydantic model.

        Args:
            db_schema: Database domain schema model.

        Returns:
            Pydantic domain schema model.
        """
        import json

        parent_domain_name = None
        if db_schema.parent_domain:
            parent_domain_name = db_schema.parent_domain.domain_name

        entity_types = {}
        try:
            entity_types_raw = json.loads(db_schema.entity_types)
            for name, et_data in entity_types_raw.items():
                entity_types[name] = EntityTypeDef(**et_data)
        except (json.JSONDecodeError, TypeError):
            entity_types = {}

        return DomainSchemaModel(
            id=db_schema.id,
            domain_name=db_schema.domain_name,
            domain_display_name=db_schema.domain_display_name,
            description=db_schema.description,
            parent_domain_id=db_schema.parent_domain_id,
            parent_domain_name=parent_domain_name,
            domain_level=DomainLevel(db_schema.domain_level.lower()),
            inheritance_type=InheritanceType(db_schema.inheritance_type.lower()),
            entity_types=entity_types,
            validation_mode=SchemaValidationMode(db_schema.validation_mode),
            is_active=db_schema.is_active == "Y",
            version=db_schema.version,
            created_at=db_schema.created_at,
            updated_at=db_schema.updated_at,
        )

    def clear_cache(self) -> None:
        """Clear the schema cache."""
        self._cache.clear()


def get_schema_registry(db: AsyncSession) -> SchemaRegistry:
    """Dependency factory for SchemaRegistry instances.

    Args:
        db: Database session from dependency injection.

    Returns:
        SchemaRegistry instance ready for use in services.
    """
    return SchemaRegistry(db)
