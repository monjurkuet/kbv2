"""Unit tests for schema API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from knowledge_base.schema_api import router, get_schema_registry
from knowledge_base.intelligence import (
    EntityTypeDef,
    DomainAttribute,
    InheritanceType,
    DomainLevel,
)
from knowledge_base.intelligence.v1.domain_schema_service import (
    DomainSchemaModel,
    SchemaValidationMode,
    SchemaRegistry,
)
from uuid import uuid4
from datetime import datetime


@pytest.fixture
def sample_schema_model():
    """Create a sample domain schema model."""
    return DomainSchemaModel(
        id=uuid4(),
        domain_name="TECHNOLOGY",
        domain_display_name="Technology Domain",
        description="Domain for technology-related entities",
        parent_domain_id=None,
        parent_domain_name="GENERAL",
        domain_level=DomainLevel.PRIMARY,
        inheritance_type=InheritanceType.EXTENDS,
        entity_types={
            "Software": EntityTypeDef(
                type_name="Software",
                parent_type="PRODUCT",
                description="Software application",
                attributes={
                    "version": DomainAttribute(name="version", attribute_type="str"),
                    "license": DomainAttribute(name="license", attribute_type="str"),
                },
            ),
            "API": EntityTypeDef(
                type_name="API",
                parent_type="CONCEPT",
                description="Application Programming Interface",
                attributes={
                    "endpoint": DomainAttribute(name="endpoint", attribute_type="str"),
                    "method": DomainAttribute(name="method", attribute_type="str"),
                },
            ),
        },
        validation_mode=SchemaValidationMode.ADAPTIVE,
        is_active=True,
        version="1.0.0",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


class TestSchemaListItem:
    """Tests for SchemaListItem model."""

    def test_schema_list_item_creation(self):
        """Test creating a SchemaListItem."""
        from knowledge_base.schema_api import SchemaListItem

        item = SchemaListItem(
            domain_name="TECHNOLOGY",
            level="primary",
            entity_types=2,
            parent_domain_name="GENERAL",
        )
        assert item.domain_name == "TECHNOLOGY"
        assert item.level == "primary"
        assert item.entity_types == 2
        assert item.parent_domain_name == "GENERAL"


class TestEntityTypeDefRequest:
    """Tests for EntityTypeDefRequest model."""

    def test_entity_type_def_request_creation(self):
        """Test creating an EntityTypeDefRequest."""
        from knowledge_base.schema_api import (
            EntityTypeDefRequest,
            DomainAttributeRequest,
        )

        request = EntityTypeDefRequest(
            type_name="Software",
            parent_type="PRODUCT",
            description="Software application",
            attributes={
                "version": DomainAttributeRequest(name="version", attribute_type="str"),
            },
            required_attributes=["version"],
        )
        assert request.type_name == "Software"
        assert request.parent_type == "PRODUCT"
        assert "version" in request.attributes
        assert "version" in request.required_attributes


class TestRegisterSchemaRequest:
    """Tests for RegisterSchemaRequest model."""

    def test_register_schema_request_creation(self):
        """Test creating a RegisterSchemaRequest."""
        from knowledge_base.schema_api import (
            RegisterSchemaRequest,
            EntityTypeDefRequest,
        )

        request = RegisterSchemaRequest(
            domain_name="CUSTOM",
            domain_display_name="Custom Domain",
            entity_types=[
                EntityTypeDefRequest(
                    type_name="CustomEntity", description="A custom entity"
                )
            ],
            parent_domain_name="GENERAL",
            inheritance_type="extends",
        )
        assert request.domain_name == "CUSTOM"
        assert request.domain_display_name == "Custom Domain"
        assert len(request.entity_types) == 1
        assert request.parent_domain_name == "GENERAL"
        assert request.inheritance_type == "extends"


class TestSchemaEndpoints:
    """Tests for schema API endpoints."""

    def test_list_schemas_empty(self):
        """Test listing schemas when none exist."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.list_schemas = AsyncMock(return_value=[])

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/")
            assert response.status_code == 200
            assert response.json() == []

    def test_list_schemas_with_data(self, sample_schema_model):
        """Test listing schemas with existing data."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.list_schemas = AsyncMock(return_value=[sample_schema_model])

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["domain_name"] == "TECHNOLOGY"
            assert data[0]["level"] == "primary"
            assert data[0]["entity_types"] == 2

    def test_get_schema_not_found(self):
        """Test getting a non-existent schema."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=None)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/NONEXISTENT")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    def test_get_schema_success(self, sample_schema_model):
        """Test getting a schema successfully."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)
        mock_registry.get_inherited_entity_types = AsyncMock(return_value=[])

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/TECHNOLOGY")
            assert response.status_code == 200
            data = response.json()
            assert data["domain_name"] == "TECHNOLOGY"
            assert data["domain_display_name"] == "Technology Domain"
            assert len(data["entity_types"]) == 2

    def test_get_schema_with_inheritance(self, sample_schema_model):
        """Test getting a schema with inheritance."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)
        mock_registry.get_inherited_entity_types = AsyncMock(
            return_value=[
                ("GENERAL", EntityTypeDef(type_name="NamedEntity", attributes={})),
                ("TECHNOLOGY", sample_schema_model.entity_types["Software"]),
            ]
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/TECHNOLOGY?include_inherited=true")
            assert response.status_code == 200
            data = response.json()
            assert data["inherited_types"] is not None

    def test_delete_schema_not_found(self):
        """Test deleting a non-existent schema."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.delete = AsyncMock(return_value=False)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.delete("/api/v1/schemas/NONEXISTENT")
            assert response.status_code == 404

    def test_delete_root_schema(self):
        """Test deleting a root schema should fail."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: MagicMock()

        with TestClient(app) as client:
            response = client.delete("/api/v1/schemas/GENERAL")
            assert response.status_code == 400
            assert "Cannot delete" in response.json()["detail"]

    def test_delete_root_schema_lowercase(self):
        """Test deleting root schema with uppercase should also fail."""
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: MagicMock()

        with TestClient(app) as client:
            response = client.delete("/api/v1/schemas/ROOT")
            assert response.status_code == 400
            assert "Cannot delete" in response.json()["detail"]

    def test_get_entity_types_not_found(self):
        """Test getting entity types for non-existent schema."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=None)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/NONEXISTENT/entity-types")
            assert response.status_code == 404

    def test_get_entity_types_success(self, sample_schema_model):
        """Test getting entity types successfully."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/TECHNOLOGY/entity-types")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["type_name"] == "Software"

    def test_get_specific_entity_type_not_found(self, sample_schema_model):
        """Test getting a non-existent entity type."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)
        mock_registry.get_inherited_entity_types = AsyncMock(return_value=[])

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/TECHNOLOGY/entity-types/NonExistent")
            assert response.status_code == 404

    def test_get_specific_entity_type_success(self, sample_schema_model):
        """Test getting a specific entity type successfully."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)
        mock_registry.get_inherited_entity_types = AsyncMock(
            return_value=[("TECHNOLOGY", sample_schema_model.entity_types["Software"])]
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get("/api/v1/schemas/TECHNOLOGY/entity-types/Software")
            assert response.status_code == 200
            data = response.json()
            assert data["type_name"] == "Software"
            assert "version" in data["attributes"]

    def test_get_entity_type_with_inheritance_chain(self, sample_schema_model):
        """Test getting entity type with inheritance chain."""
        mock_registry = MagicMock(spec=SchemaRegistry)
        mock_registry.get_by_name = AsyncMock(return_value=sample_schema_model)
        mock_registry.get_inherited_entity_types = AsyncMock(
            return_value=[
                ("GENERAL", EntityTypeDef(type_name="NamedEntity", attributes={})),
                ("TECHNOLOGY", sample_schema_model.entity_types["Software"]),
            ]
        )

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_schema_registry] = lambda: mock_registry

        with TestClient(app) as client:
            response = client.get(
                "/api/v1/schemas/TECHNOLOGY/entity-types/Software?include_inherited=true"
            )
            assert response.status_code == 200
            data = response.json()
            assert "inheritance_chain" in data
            assert len(data["inheritance_chain"]) == 2


class TestSchemaAPIRouter:
    """Tests for schema API router configuration."""

    def test_router_prefix(self):
        """Test router has correct prefix."""
        assert router.prefix == "/api/v1/schemas"

    def test_router_tags(self):
        """Test router has correct tags."""
        assert "schemas" in router.tags

    def test_endpoints_registered(self):
        """Test all expected endpoints are registered."""
        route_paths = [route.path for route in router.routes]

        assert "/api/v1/schemas/" in route_paths
        assert "/api/v1/schemas/{domain_name}" in route_paths
        assert "/api/v1/schemas/{domain_name}/entity-types" in route_paths
        assert "/api/v1/schemas/{domain_name}/entity-types/{entity_type}" in route_paths
