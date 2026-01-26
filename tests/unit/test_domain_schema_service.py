"""Unit tests for domain schema service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime

from knowledge_base.intelligence.v1.domain_schema_service import (
    DomainLevel,
    InheritanceType,
    SchemaValidationMode,
    DomainAttribute,
    EntityTypeDef,
    DomainSchema,
    DomainSchemaModel,
    DomainSchemaCreate,
    DomainSchemaUpdate,
    SchemaRegistry,
)


class TestDomainAttribute:
    """Tests for DomainAttribute model."""

    def test_create_attribute(self) -> None:
        """Test creating a domain attribute."""
        attr = DomainAttribute(
            name="age",
            attribute_type="int",
            description="Person's age",
            required=True,
            default_value=0,
        )
        assert attr.name == "age"
        assert attr.attribute_type == "int"
        assert attr.required is True
        assert attr.default_value == 0

    def test_attribute_with_validation_rules(self) -> None:
        """Test attribute with validation rules."""
        attr = DomainAttribute(
            name="email",
            attribute_type="str",
            validation_rules={
                "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$",
                "max_length": 255,
            },
        )
        assert "pattern" in attr.validation_rules
        assert attr.validation_rules["max_length"] == 255

    def test_default_domain_specific(self) -> None:
        """Test default domain_specific flag."""
        attr = DomainAttribute(name="test", attribute_type="str")
        assert attr.domain_specific is True


class TestEntityTypeDef:
    """Tests for EntityTypeDef model."""

    def test_create_entity_type_def(self) -> None:
        """Test creating entity type definition."""
        entity_type = EntityTypeDef(
            type_name="Person",
            description="A human being",
            required_attributes=["name", "age"],
        )
        assert entity_type.type_name == "Person"
        assert entity_type.parent_type is None
        assert "name" in entity_type.required_attributes

    def test_entity_type_with_parent(self) -> None:
        """Test entity type with parent type."""
        entity_type = EntityTypeDef(
            type_name="Doctor",
            parent_type="Person",
            description="A medical professional",
        )
        assert entity_type.parent_type == "Person"

    def test_get_attribute(self) -> None:
        """Test getting attribute by name."""
        attr = DomainAttribute(name="specialty", attribute_type="str")
        entity_type = EntityTypeDef(
            type_name="Doctor",
            attributes={"specialty": attr},
        )
        retrieved = entity_type.get_attribute("specialty")
        assert retrieved is not None
        assert retrieved.name == "specialty"

    def test_get_missing_attribute(self) -> None:
        """Test getting non-existent attribute."""
        entity_type = EntityTypeDef(type_name="Person")
        retrieved = entity_type.get_attribute("nonexistent")
        assert retrieved is None

    def test_entity_type_with_attributes(self) -> None:
        """Test entity type with domain attributes."""
        attrs = {
            "name": DomainAttribute(name="name", attribute_type="str", required=True),
            "age": DomainAttribute(name="age", attribute_type="int", required=False),
            "specialty": DomainAttribute(name="specialty", attribute_type="str"),
        }
        entity_type = EntityTypeDef(
            type_name="Doctor",
            attributes=attrs,
            required_attributes=["name"],
        )
        assert len(entity_type.attributes) == 3
        assert "name" in entity_type.required_attributes


class TestDomainSchemaModel:
    """Tests for DomainSchemaModel."""

    def test_create_domain_schema_model(self) -> None:
        """Test creating domain schema model."""
        schema = DomainSchemaModel(
            id=uuid4(),
            domain_name="medical",
            domain_display_name="Medical Domain",
            description="Domain for medical entities",
            domain_level=DomainLevel.PRIMARY,
            inheritance_type=InheritanceType.EXTENDS,
            entity_types={},
            validation_mode=SchemaValidationMode.ADAPTIVE,
            is_active=True,
            version="1.0.0",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert schema.domain_name == "medical"
        assert schema.domain_level == DomainLevel.PRIMARY
        assert schema.is_active is True

    def test_schema_with_entity_types(self) -> None:
        """Test schema with entity type definitions."""
        entity_type = EntityTypeDef(
            type_name="Patient",
            attributes={
                "name": DomainAttribute(
                    name="name", attribute_type="str", required=True
                ),
                "medical_record_number": DomainAttribute(
                    name="medical_record_number",
                    attribute_type="str",
                    required=True,
                ),
            },
        )
        schema = DomainSchemaModel(
            id=uuid4(),
            domain_name="medical",
            domain_display_name="Medical Domain",
            entity_types={"Patient": entity_type},
            domain_level=DomainLevel.PRIMARY,
            inheritance_type=InheritanceType.EXTENDS,
            validation_mode=SchemaValidationMode.STRICT,
            is_active=True,
            version="1.0.0",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        assert "Patient" in schema.entity_types
        assert schema.entity_types["Patient"].type_name == "Patient"


class TestDomainSchemaCreate:
    """Tests for DomainSchemaCreate model."""

    def test_create_schema_creation_model(self) -> None:
        """Test creating schema creation model."""
        schema = DomainSchemaCreate(
            domain_name="legal",
            domain_display_name="Legal Domain",
            description="Domain for legal entities",
            domain_level=DomainLevel.SECONDARY,
            inheritance_type=InheritanceType.COMPOSES,
        )
        assert schema.domain_name == "legal"
        assert schema.domain_level == DomainLevel.SECONDARY

    def test_create_with_entity_types(self) -> None:
        """Test creating schema with entity types."""
        entity_type = EntityTypeDef(type_name="Contract")
        schema = DomainSchemaCreate(
            domain_name="legal",
            domain_display_name="Legal Domain",
            entity_types={"Contract": entity_type},
        )
        assert "Contract" in schema.entity_types


class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    @pytest.fixture
    def mock_db_session(self) -> MagicMock:
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def registry(self, mock_db_session: MagicMock) -> SchemaRegistry:
        """Create schema registry with mock session."""
        return SchemaRegistry(mock_db_session)

    def test_init(self, registry: SchemaRegistry) -> None:
        """Test registry initialization."""
        assert registry._cache == {}

    @pytest.mark.asyncio
    async def test_register_schema(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test registering a new schema."""
        schema_create = DomainSchemaCreate(
            domain_name="technology",
            domain_display_name="Technology Domain",
            description="Domain for technology entities",
        )

        schema_id = uuid4()
        now = datetime.utcnow()

        async def mock_refresh(db_schema):
            db_schema.id = schema_id
            db_schema.created_at = now
            db_schema.updated_at = now
            db_schema.parent_domain_id = None
            db_schema.parent_domain = None
            db_schema.domain_level = "primary"
            db_schema.inheritance_type = "extends"
            db_schema.entity_types = "{}"
            db_schema.validation_mode = "adaptive"
            db_schema.is_active = "Y"

        mock_db_session.refresh = mock_refresh

        result = await registry.register(schema_create)

        assert result.domain_name == "technology"
        assert result.domain_display_name == "Technology Domain"
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_name_found(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test getting schema by name when it exists."""
        schema_id = uuid4()
        db_schema = MagicMock()
        db_schema.id = schema_id
        db_schema.domain_name = "test_domain"
        db_schema.domain_display_name = "Test Domain"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await registry.get_by_name("test_domain")

        assert result is not None
        assert result.domain_name == "test_domain"

    @pytest.mark.asyncio
    async def test_get_by_name_not_found(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test getting schema by name when it doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await registry.get_by_name("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_schemas(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test listing all schemas."""
        schema_id = uuid4()
        db_schema = MagicMock()
        db_schema.id = schema_id
        db_schema.domain_name = "domain1"
        db_schema.domain_display_name = "Domain 1"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.scalars().all = MagicMock(return_value=[db_schema])
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        results = await registry.list_schemas()

        assert len(results) == 1
        assert results[0].domain_name == "domain1"

    @pytest.mark.asyncio
    async def test_update_schema(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test updating a schema."""
        schema_id = uuid4()
        db_schema = MagicMock()
        db_schema.id = schema_id
        db_schema.domain_name = "test_domain"
        db_schema.domain_display_name = "Original Name"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.scalar_one = MagicMock(return_value=db_schema)
        mock_result.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        update = DomainSchemaUpdate(domain_display_name="Updated Name")
        result = await registry.update("test_domain", update)

        assert result is not None
        assert db_schema.domain_display_name == "Updated Name"
        mock_db_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_schema_found(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test deleting an existing schema."""
        db_schema = MagicMock()
        db_schema.domain_name = "test_domain"
        db_schema.domain_display_name = "Test Domain"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()
        db_schema.id = uuid4()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await registry.delete("test_domain")

        assert result is True
        mock_db_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_delete_schema_not_found(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test deleting a non-existent schema."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result = await registry.delete("nonexistent")

        assert result is False

    def test_clear_cache(self, registry: SchemaRegistry) -> None:
        """Test clearing the cache."""
        registry._cache["test"] = MagicMock()
        registry.clear_cache()
        assert registry._cache == {}


class TestDomainLevel:
    """Tests for DomainLevel enum."""

    def test_domain_level_values(self) -> None:
        """Test domain level enum values."""
        assert DomainLevel.ROOT.value == "root"
        assert DomainLevel.PRIMARY.value == "primary"
        assert DomainLevel.SECONDARY.value == "secondary"
        assert DomainLevel.TERTIARY.value == "tertiary"


class TestInheritanceType:
    """Tests for InheritanceType enum."""

    def test_inheritance_type_values(self) -> None:
        """Test inheritance type enum values."""
        assert InheritanceType.EXTENDS.value == "extends"
        assert InheritanceType.OVERRIDES.value == "overrides"
        assert InheritanceType.COMPOSES.value == "composes"


class TestSchemaValidationMode:
    """Tests for SchemaValidationMode enum."""

    def test_validation_mode_values(self) -> None:
        """Test validation mode enum values."""
        assert SchemaValidationMode.STRICT.value == "strict"
        assert SchemaValidationMode.LAX.value == "lax"
        assert SchemaValidationMode.ADAPTIVE.value == "adaptive"


class TestSchemaInheritance:
    """Tests for schema inheritance functionality."""

    @pytest.fixture
    def parent_entity_type(self) -> EntityTypeDef:
        """Create parent entity type."""
        return EntityTypeDef(
            type_name="Person",
            description="A human being",
            attributes={
                "name": DomainAttribute(
                    name="name", attribute_type="str", required=True
                ),
                "age": DomainAttribute(
                    name="age", attribute_type="int", required=False
                ),
            },
            required_attributes=["name"],
        )

    @pytest.fixture
    def child_entity_type(self) -> EntityTypeDef:
        """Create child entity type with additional attributes."""
        return EntityTypeDef(
            type_name="Doctor",
            parent_type="Person",
            description="A medical professional",
            attributes={
                "specialty": DomainAttribute(
                    name="specialty", attribute_type="str", required=True
                ),
                "license_number": DomainAttribute(
                    name="license_number", attribute_type="str", required=True
                ),
            },
            required_attributes=["specialty", "license_number"],
        )

    def test_entity_type_inheritance_chain(
        self, parent_entity_type: EntityTypeDef, child_entity_type: EntityTypeDef
    ) -> None:
        """Test entity type inheritance chain."""
        assert child_entity_type.parent_type == "Person"
        assert parent_entity_type.type_name == "Person"

    def test_combined_attributes(self) -> None:
        """Test combining attributes from parent and child."""
        parent_attrs = {
            "name": DomainAttribute(name="name", attribute_type="str", required=True),
            "age": DomainAttribute(name="age", attribute_type="int", required=False),
        }
        child_attrs = {
            "specialty": DomainAttribute(
                name="specialty", attribute_type="str", required=True
            ),
        }

        all_attrs = parent_attrs.copy()
        all_attrs.update(child_attrs)

        assert "name" in all_attrs
        assert "specialty" in all_attrs
        assert len(all_attrs) == 3


class TestDomainSchemaEdgeCases:
    """Tests for edge cases and error handling."""

    def test_entity_type_def_without_attributes(self) -> None:
        """Test entity type with no attributes."""
        entity_type = EntityTypeDef(type_name="Minimal")
        assert entity_type.attributes == {}
        assert entity_type.required_attributes == []

    def test_schema_create_with_defaults(self) -> None:
        """Test schema creation with default values."""
        schema = DomainSchemaCreate(
            domain_name="test",
            domain_display_name="Test",
        )
        assert schema.domain_level == DomainLevel.PRIMARY
        assert schema.inheritance_type == InheritanceType.EXTENDS
        assert schema.validation_mode == SchemaValidationMode.ADAPTIVE
        assert schema.version == "1.0.0"

    def test_domain_attribute_complex_types(self) -> None:
        """Test domain attribute with complex types."""
        attr = DomainAttribute(
            name="tags",
            attribute_type="list[str]",
            default_value=[],
        )
        assert attr.attribute_type == "list[str]"
        assert attr.default_value == []

    def test_entity_type_with_metadata(self) -> None:
        """Test entity type with custom metadata."""
        entity_type = EntityTypeDef(
            type_name="CustomEntity",
            metadata={"version": "1.0", "author": "test"},
        )
        assert entity_type.metadata["version"] == "1.0"
        assert entity_type.metadata["author"] == "test"


class TestSchemaRegistryCache:
    """Tests for schema registry caching behavior."""

    @pytest.fixture
    def mock_db_session(self) -> MagicMock:
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.fixture
    def registry(self, mock_db_session: MagicMock) -> SchemaRegistry:
        """Create schema registry with mock session."""
        return SchemaRegistry(mock_db_session)

    @pytest.mark.asyncio
    async def test_cache_hit_after_get(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test that cached schema is returned on subsequent calls."""
        schema_id = uuid4()
        db_schema = MagicMock()
        db_schema.id = schema_id
        db_schema.domain_name = "cached_domain"
        db_schema.domain_display_name = "Cached Domain"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        result1 = await registry.get_by_name("cached_domain")
        result2 = await registry.get_by_name("cached_domain")

        assert result1 is not None
        assert result2 is not None
        assert result1.id == result2.id
        assert mock_db_session.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_cleared_on_delete(
        self, mock_db_session: MagicMock, registry: SchemaRegistry
    ) -> None:
        """Test that cache is cleared when schema is deleted."""
        schema_id = uuid4()
        db_schema = MagicMock()
        db_schema.domain_name = "to_delete"
        db_schema.domain_display_name = "To Delete"
        db_schema.description = None
        db_schema.parent_domain_id = None
        db_schema.parent_domain = None
        db_schema.domain_level = "primary"
        db_schema.inheritance_type = "extends"
        db_schema.entity_types = "{}"
        db_schema.validation_mode = "adaptive"
        db_schema.is_active = "Y"
        db_schema.version = "1.0.0"
        db_schema.created_at = datetime.utcnow()
        db_schema.updated_at = datetime.utcnow()
        db_schema.id = schema_id

        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        await registry.get_by_name("to_delete")
        assert "to_delete" in registry._cache

        mock_db_execute = MagicMock()
        mock_db_execute.scalar_one_or_none = MagicMock(return_value=db_schema)
        mock_db_session.execute = AsyncMock(return_value=mock_db_execute)

        await registry.delete("to_delete")
        assert "to_delete" not in registry._cache
