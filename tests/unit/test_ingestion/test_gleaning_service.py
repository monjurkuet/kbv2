"""Unit tests for gleaning_service multi-modal extraction capabilities."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError

from knowledge_base.ingestion.v1.gleaning_service import (
    ExtractedTable,
    ExtractedImage,
    ExtractedFigure,
    ExtractionResult,
    ExtractedEntity,
    ExtractedEdge,
    GleaningService,
    GleaningConfig,
)
from knowledge_base.persistence.v1.schema import EdgeType


class TestExtractedTable:
    """Tests for ExtractedTable model."""

    def test_table_creation_with_all_fields(self):
        """Test creating a table with all fields."""
        table = ExtractedTable(
            content="| Header1 | Header2 |\n| --- | --- |\n| Cell1 | Cell2 |",
            page_number=1,
            description="Sales data for Q1 2024",
        )
        assert (
            table.content == "| Header1 | Header2 |\n| --- | --- |\n| Cell1 | Cell2 |"
        )
        assert table.page_number == 1
        assert table.description == "Sales data for Q1 2024"

    def test_table_creation_without_page_number(self):
        """Test creating a table without page number."""
        table = ExtractedTable(
            content="| Col1 | Col2 |\n| --- | --- |\n| Val1 | Val2 |",
            description="Simple table",
        )
        assert table.content == "| Col1 | Col2 |\n| --- | --- |\n| Val1 | Val2 |"
        assert table.page_number is None
        assert table.description == "Simple table"

    def test_table_creation_with_empty_description(self):
        """Test creating a table with empty description."""
        table = ExtractedTable(content="| A | B |\n| --- | --- |\n| 1 | 2 |")
        assert table.content == "| A | B |\n| --- | --- |\n| 1 | 2 |"
        assert table.description == ""

    def test_table_markdown_format_validation(self):
        """Test that markdown table format is properly stored."""
        table = ExtractedTable(
            content="| Name | Age | Role |\n| --- | --- | --- |\n| Alice | 30 | Engineer |\n| Bob | 25 | Designer |",
            page_number=3,
            description="Team members table",
        )
        assert "|" in table.content
        assert "---" in table.content

    def test_table_json_serialization(self):
        """Test table can be serialized to JSON."""
        table = ExtractedTable(
            content="| X | Y |\n| --- | --- |\n| 1 | 2 |",
            page_number=5,
            description="Coordinates",
        )
        json_data = table.model_dump()
        assert json_data["content"] == "| X | Y |\n| --- | --- |\n| 1 | 2 |"
        assert json_data["page_number"] == 5
        assert json_data["description"] == "Coordinates"


class TestExtractedImage:
    """Tests for ExtractedImage model."""

    def test_image_creation_with_all_fields(self):
        """Test creating an image with all fields."""
        image = ExtractedImage(
            description="Dashboard screenshot showing key metrics",
            embedded_text="Total Users: 1,234\nRevenue: $56,789\nGrowth: 15%",
            page_number=2,
        )
        assert image.description == "Dashboard screenshot showing key metrics"
        assert "Total Users: 1,234" in image.embedded_text
        assert image.page_number == 2

    def test_image_creation_without_page_number(self):
        """Test creating an image without page number."""
        image = ExtractedImage(description="Photo of the team", embedded_text="")
        assert image.description == "Photo of the team"
        assert image.page_number is None
        assert image.embedded_text == ""

    def test_image_with_multiline_text(self):
        """Test image with multiple lines of extracted text."""
        image = ExtractedImage(
            description="Document scan",
            embedded_text="Line 1: Header\nLine 2: Content\nLine 3: Footer",
            page_number=10,
        )
        lines = image.embedded_text.split("\n")
        assert len(lines) == 3

    def test_image_json_serialization(self):
        """Test image can be serialized to JSON."""
        image = ExtractedImage(
            description="Screenshot", embedded_text="Text: Hello World", page_number=1
        )
        json_data = image.model_dump()
        assert json_data["description"] == "Screenshot"
        assert json_data["embedded_text"] == "Text: Hello World"
        assert json_data["page_number"] == 1


class TestExtractedFigure:
    """Tests for ExtractedFigure model."""

    def test_bar_chart_creation(self):
        """Test creating a bar chart figure."""
        figure = ExtractedFigure(
            type="bar_chart",
            description="Monthly revenue trend for 2024",
            data_points=[
                {"month": "Jan", "value": 10000},
                {"month": "Feb", "value": 15000},
                {"month": "Mar", "value": 12000},
            ],
        )
        assert figure.type == "bar_chart"
        assert len(figure.data_points) == 3
        assert figure.data_points[0]["month"] == "Jan"

    def test_line_graph_creation(self):
        """Test creating a line graph figure."""
        figure = ExtractedFigure(
            type="line_graph",
            description="Stock price over time",
            data_points=[
                {"date": "2024-01-01", "price": 100.5},
                {"date": "2024-01-02", "price": 102.3},
            ],
        )
        assert figure.type == "line_graph"
        assert len(figure.data_points) == 2

    def test_pie_chart_creation(self):
        """Test creating a pie chart figure."""
        figure = ExtractedFigure(
            type="pie_chart",
            description="Market share distribution",
            data_points=[
                {"category": "Product A", "percentage": 45},
                {"category": "Product B", "percentage": 30},
                {"category": "Product C", "percentage": 25},
            ],
        )
        assert figure.type == "pie_chart"
        assert len(figure.data_points) == 3

    def test_diagram_creation(self):
        """Test creating a diagram figure."""
        figure = ExtractedFigure(
            type="diagram",
            description="System architecture diagram",
            data_points=[
                {"component": "Frontend", "connections": 2},
                {"component": "Backend", "connections": 3},
            ],
        )
        assert figure.type == "diagram"

    def test_flowchart_creation(self):
        """Test creating a flowchart figure."""
        figure = ExtractedFigure(
            type="flowchart",
            description="Decision process flowchart",
            data_points=[
                {"step": "Start", "next": "Step 1"},
                {"step": "Step 1", "next": "Step 2"},
                {"step": "Step 2", "next": "End"},
            ],
        )
        assert figure.type == "flowchart"
        assert len(figure.data_points) == 3

    def test_figure_with_empty_data_points(self):
        """Test creating a figure with empty data points list."""
        figure = ExtractedFigure(
            type="other", description="Generic figure without specific data"
        )
        assert figure.type == "other"
        assert figure.data_points == []

    def test_figure_json_serialization(self):
        """Test figure can be serialized to JSON."""
        figure = ExtractedFigure(
            type="bar_chart", description="Test chart", data_points=[{"x": 1, "y": 2}]
        )
        json_data = figure.model_dump()
        assert json_data["type"] == "bar_chart"
        assert json_data["data_points"] == [{"x": 1, "y": 2}]


class TestExtractionResult:
    """Tests for ExtractionResult model with multi-modal fields."""

    def test_extraction_result_with_all_fields(self):
        """Test creating extraction result with all multi-modal fields."""
        result = ExtractionResult(
            entities=[ExtractedEntity(name="Test Entity", entity_type="TEST")],
            edges=[
                ExtractedEdge(
                    source="Entity1", target="Entity2", edge_type=EdgeType.RELATED_TO
                )
            ],
            tables=[
                ExtractedTable(
                    content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                    page_number=1,
                    description="Test table",
                )
            ],
            images_with_text=[
                ExtractedImage(
                    description="Test image",
                    embedded_text="Extracted text",
                    page_number=2,
                )
            ],
            figures=[
                ExtractedFigure(
                    type="bar_chart",
                    description="Test figure",
                    data_points=[{"x": 1, "y": 2}],
                )
            ],
            information_density=0.8,
        )
        assert len(result.entities) == 1
        assert len(result.tables) == 1
        assert len(result.images_with_text) == 1
        assert len(result.figures) == 1
        assert result.information_density == 0.8

    def test_extraction_result_empty_multi_modal_fields(self):
        """Test extraction result with empty multi-modal fields."""
        result = ExtractionResult(entities=[], edges=[], information_density=0.5)
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []

    def test_extraction_result_default_values(self):
        """Test extraction result has correct default values."""
        result = ExtractionResult()
        assert result.entities == []
        assert result.edges == []
        assert result.temporal_claims == []
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []
        assert result.information_density == 0.0

    def test_extraction_result_json_schema(self):
        """Test extraction result matches expected JSON schema."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Company ABC",
                    entity_type="ORGANIZATION",
                    description="A tech company",
                    confidence=0.95,
                )
            ],
            tables=[
                ExtractedTable(
                    content="| Revenue | Year |\n| --- | --- |\n| 100M | 2023 |",
                    page_number=5,
                    description="Financial summary",
                )
            ],
            information_density=0.7,
        )
        schema = result.model_json_schema()
        assert "tables" in schema["properties"]
        assert "images_with_text" in schema["properties"]
        assert "figures" in schema["properties"]


class TestGleaningService:
    """Tests for GleaningService multi-modal extraction methods."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock gateway client."""
        gateway = AsyncMock()
        return gateway

    @pytest.fixture
    def gleaning_service(self, mock_gateway):
        """Create a gleaning service with mock gateway."""
        config = GleaningConfig(
            max_passes=2,
            min_density_threshold=0.3,
            max_density_threshold=0.8,
            diminishing_returns_threshold=0.05,
            stability_threshold=0.90,
        )
        return GleaningService(gateway=mock_gateway, config=config)

    def test_get_discovery_prompt_contains_table_instruction(self):
        """Test discovery prompt includes table extraction instructions."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert "TABLES:" in prompt
        assert "markdown format" in prompt

    def test_get_discovery_prompt_contains_image_instruction(self):
        """Test discovery prompt includes image extraction instructions."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert "IMAGES:" in prompt
        assert "visible text" in prompt

    def test_get_discovery_prompt_contains_figure_instruction(self):
        """Test discovery prompt includes figure extraction instructions."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert "FIGURES:" in prompt
        assert "bar_chart" in prompt or "line_graph" in prompt

    def test_get_discovery_prompt_json_schema_includes_tables(self):
        """Test discovery prompt JSON schema includes tables field."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert '"tables":' in prompt
        assert '"page_number":' in prompt

    def test_get_discovery_prompt_json_schema_includes_images(self):
        """Test discovery prompt JSON schema includes images_with_text field."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert '"images_with_text":' in prompt
        assert '"embedded_text":' in prompt

    def test_get_discovery_prompt_json_schema_includes_figures(self):
        """Test discovery prompt JSON schema includes figures field."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_discovery_prompt()
        assert '"figures":' in prompt
        assert '"data_points":' in prompt

    def test_get_gleaning_prompt_contains_multi_modal_instruction(self):
        """Test gleaning prompt includes multi-modal content focus."""
        service = GleaningService(gateway=AsyncMock())
        prompt = service._get_gleaning_prompt()
        assert "TABLES:" in prompt
        assert "IMAGES:" in prompt
        assert "FIGURES:" in prompt

    @pytest.mark.asyncio
    async def test_extract_parses_table_from_response(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction parses tables from LLM response."""
        mock_gateway.generate_text.return_value = """{
            "entities": [],
            "edges": [],
            "temporal_claims": [],
            "tables": [
                {
                    "content": "| Header1 | Header2 |\\n| --- | --- |\\n| Cell1 | Cell2 |",
                    "page_number": 1,
                    "description": "Test table"
                }
            ],
            "images_with_text": [],
            "figures": [],
            "information_density": 0.5
        }"""

        result = await gleaning_service.extract("Test text")

        assert len(result.tables) == 1
        assert (
            result.tables[0].content
            == "| Header1 | Header2 |\n| --- | --- |\n| Cell1 | Cell2 |"
        )
        assert result.tables[0].page_number == 1

    @pytest.mark.asyncio
    async def test_extract_parses_image_from_response(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction parses images with text from LLM response."""
        mock_gateway.generate_text.return_value = """{
            "entities": [],
            "edges": [],
            "temporal_claims": [],
            "tables": [],
            "images_with_text": [
                {
                    "description": "Dashboard screenshot",
                    "embedded_text": "Revenue: $100K",
                    "page_number": 3
                }
            ],
            "figures": [],
            "information_density": 0.5
        }"""

        result = await gleaning_service.extract("Test text")

        assert len(result.images_with_text) == 1
        assert result.images_with_text[0].description == "Dashboard screenshot"
        assert "Revenue: $100K" in result.images_with_text[0].embedded_text

    @pytest.mark.asyncio
    async def test_extract_parses_figure_from_response(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction parses figures from LLM response."""
        mock_gateway.generate_text.return_value = """{
            "entities": [],
            "edges": [],
            "temporal_claims": [],
            "tables": [],
            "images_with_text": [],
            "figures": [
                {
                    "type": "pie_chart",
                    "description": "Market share",
                    "data_points": [
                        {"product": "A", "share": 40},
                        {"product": "B", "share": 60}
                    ]
                }
            ],
            "information_density": 0.5
        }"""

        result = await gleaning_service.extract("Test text")

        assert len(result.figures) == 1
        assert result.figures[0].type == "pie_chart"
        assert len(result.figures[0].data_points) == 2

    @pytest.mark.asyncio
    async def test_extract_handles_empty_multi_modal_fields(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction handles empty tables, images, and figures gracefully."""
        mock_gateway.generate_text.return_value = """{
            "entities": [{"name": "Test", "type": "TEST"}],
            "edges": [],
            "temporal_claims": [],
            "tables": [],
            "images_with_text": [],
            "figures": [],
            "information_density": 0.5
        }"""

        result = await gleaning_service.extract("Test text")

        assert len(result.entities) == 1
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []

    @pytest.mark.asyncio
    async def test_extract_merges_multi_modal_results_from_passes(
        self, gleaning_service, mock_gateway
    ):
        """Test multi-modal results are merged across passes."""
        gleaning_service._config.max_passes = 2
        gleaning_service._config.min_density_threshold = 0.1

        mock_gateway.generate_text.side_effect = [
            """{
                "entities": [{"name": "Entity1", "type": "TEST"}],
                "edges": [],
                "temporal_claims": [],
                "tables": [{"content": "| A | B |", "page_number": 1, "description": "Table 1"}],
                "images_with_text": [],
                "figures": [],
                "information_density": 0.6
            }""",
            """{
                "entities": [],
                "edges": [],
                "temporal_claims": [],
                "tables": [{"content": "| C | D |", "page_number": 2, "description": "Table 2"}],
                "images_with_text": [{"description": "Image 1", "embedded_text": "Text", "page_number": 1}],
                "figures": [],
                "information_density": 0.4
            }""",
        ]

        result = await gleaning_service.extract("Test text")

        assert len(result.tables) == 2
        assert len(result.images_with_text) == 1

    @pytest.mark.asyncio
    async def test_extract_handles_markdown_code_blocks(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction handles markdown code blocks around JSON."""
        mock_gateway.generate_text.return_value = """```json
{
    "entities": [],
    "edges": [],
    "temporal_claims": [],
    "tables": [{"content": "| X | Y |", "page_number": 1, "description": "Table"}],
    "images_with_text": [],
    "figures": [],
    "information_density": 0.5
}
```"""

        result = await gleaning_service.extract("Test text")

        assert len(result.tables) == 1
        assert result.tables[0].content == "| X | Y |"

    @pytest.mark.asyncio
    async def test_extract_handles_json_parse_error_gracefully(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction returns empty result on JSON parse error."""
        mock_gateway.generate_text.return_value = "Invalid JSON { broken"

        result = await gleaning_service.extract("Test text")

        assert result.entities == []
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []

    @pytest.mark.asyncio
    async def test_extract_handles_missing_fields_gracefully(
        self, gleaning_service, mock_gateway
    ):
        """Test extraction handles missing fields in response."""
        mock_gateway.generate_text.return_value = """{
            "entities": [{"name": "Test", "type": "TEST"}],
            "information_density": 0.5
        }"""

        result = await gleaning_service.extract("Test text")

        assert len(result.entities) == 1
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []

    def test_merge_results_preserves_multi_modal_content(self):
        """Test _merge_results preserves tables, images, and figures."""
        service = GleaningService(gateway=AsyncMock())

        result1 = ExtractionResult(
            tables=[
                ExtractedTable(content="| A |", page_number=1, description="Table 1")
            ],
            images_with_text=[
                ExtractedImage(description="Image 1", embedded_text="Text 1")
            ],
            figures=[ExtractedFigure(type="bar_chart", description="Figure 1")],
            information_density=0.5,
        )

        result2 = ExtractionResult(
            tables=[
                ExtractedTable(content="| B |", page_number=2, description="Table 2")
            ],
            images_with_text=[
                ExtractedImage(description="Image 2", embedded_text="Text 2")
            ],
            figures=[ExtractedFigure(type="line_graph", description="Figure 2")],
            information_density=0.4,
        )

        merged = service._merge_results([result1, result2])

        assert len(merged.tables) == 2
        assert len(merged.images_with_text) == 2
        assert len(merged.figures) == 2


class TestJSONSchemaValidation:
    """Tests for JSON schema validation with multi-modal fields."""

    def test_table_json_schema(self):
        """Test table JSON schema is valid."""
        schema = ExtractedTable.model_json_schema()
        assert schema["properties"]["content"]["type"] == "string"
        assert schema["properties"]["page_number"]["type"] == ["integer", "null"]
        assert schema["properties"]["description"]["type"] == "string"

    def test_image_json_schema(self):
        """Test image JSON schema is valid."""
        schema = ExtractedImage.model_json_schema()
        assert schema["properties"]["description"]["type"] == "string"
        assert schema["properties"]["embedded_text"]["type"] == "string"
        assert schema["properties"]["page_number"]["type"] == ["integer", "null"]

    def test_figure_json_schema(self):
        """Test figure JSON schema is valid."""
        schema = ExtractedFigure.model_json_schema()
        assert schema["properties"]["type"]["type"] == "string"
        assert schema["properties"]["description"]["type"] == "string"
        assert schema["properties"]["data_points"]["type"] == "array"

    def test_extraction_result_json_schema(self):
        """Test extraction result JSON schema includes all multi-modal fields."""
        schema = ExtractionResult.model_json_schema()
        required = schema.get("required", [])
        properties = schema["properties"]

        assert "tables" in properties
        assert "images_with_text" in properties
        assert "figures" in properties

        assert properties["tables"]["type"] == "array"
        assert properties["images_with_text"]["type"] == "array"
        assert properties["figures"]["type"] == "array"


class TestParsingErrorHandling:
    """Tests for error handling in parsing multi-modal content."""

    def test_parse_table_with_missing_fields(self):
        """Test table parsing handles missing optional fields."""
        table = ExtractedTable(content="| A |")
        assert table.page_number is None
        assert table.description == ""

    def test_parse_image_with_missing_fields(self):
        """Test image parsing handles missing optional fields."""
        image = ExtractedImage(description="Test")
        assert image.embedded_text == ""
        assert image.page_number is None

    def test_parse_figure_with_minimal_fields(self):
        """Test figure parsing handles minimal fields."""
        figure = ExtractedFigure(type="other", description="Test")
        assert figure.data_points == []

    def test_parse_table_with_invalid_content(self):
        """Test table parsing handles non-table content gracefully."""
        table = ExtractedTable(content="Not a table", page_number=1, description="Test")
        assert table.content == "Not a table"

    def test_parse_image_with_empty_embedded_text(self):
        """Test image parsing handles empty embedded text."""
        image = ExtractedImage(
            description="Empty image", embedded_text="", page_number=1
        )
        assert image.embedded_text == ""

    def test_parse_figure_with_complex_data_points(self):
        """Test figure parsing handles complex data point structures."""
        figure = ExtractedFigure(
            type="scatter_plot",
            description="Complex data",
            data_points=[
                {"x": 1.5, "y": 2.5, "label": "Point 1"},
                {"x": 3.0, "y": 4.5, "label": "Point 2"},
            ],
        )
        assert len(figure.data_points) == 2
        assert figure.data_points[0]["label"] == "Point 1"


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing extraction."""

    def test_extraction_result_works_without_multi_modal(self):
        """Test extraction result still works with only traditional fields."""
        entity = ExtractedEntity(name="Test Entity", entity_type="TEST")
        edge = ExtractedEdge(
            source="Entity1", target="Entity2", edge_type=EdgeType.RELATED_TO
        )

        result = ExtractionResult(
            entities=[entity], edges=[edge], information_density=0.7
        )

        assert len(result.entities) == 1
        assert len(result.edges) == 1
        assert result.tables == []
        assert result.images_with_text == []
        assert result.figures == []

    def test_prompt_backward_compatibility(self):
        """Test prompts still include traditional extraction fields."""
        service = GleaningService(gateway=AsyncMock())

        discovery_prompt = service._get_discovery_prompt()
        gleaning_prompt = service._get_gleaning_prompt()

        assert '"entities":' in discovery_prompt
        assert '"edges":' in discovery_prompt
        assert '"temporal_claims":' in discovery_prompt
        assert '"entities":' in gleaning_prompt
        assert '"edges":' in gleaning_prompt
        assert '"temporal_claims":' in gleaning_prompt

    def test_service_config_backward_compatibility(self):
        """Test service config still accepts traditional parameters."""
        config = GleaningConfig(
            max_passes=2,
            min_density_threshold=0.3,
            max_density_threshold=0.8,
            diminishing_returns_threshold=0.05,
            stability_threshold=0.90,
        )
        assert config.max_passes == 2
        assert config.min_density_threshold == 0.3
