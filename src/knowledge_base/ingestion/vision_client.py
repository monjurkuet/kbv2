"""Vision model client for PDF/image to markdown conversion.

This module provides a client for the local vision model API (localhost:8087)
that converts PDFs and images to structured markdown.

Supported models:
- qwen3-vl-plus: Vision model for PDF/image parsing, chart analysis
- qwen3-max: Text model for structured extraction
- qwen3-32b: Fast extraction tasks
- deepseek-v3.2: Reasoning and complex analysis
- claude-sonnet-4-6: High-quality extraction
"""

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VisionModelConfig(BaseModel):
    """Vision model configuration."""

    base_url: str = "http://localhost:8087"
    vision_model: str = "qwen3-vl-plus"
    text_model: str = "qwen3-max"
    timeout: float = 300.0  # 5 minutes for large documents
    max_retries: int = 3
    retry_delay: float = 1.0


class ConversionResult(BaseModel):
    """Result from document conversion."""

    content: str
    source_path: str
    page_count: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    processing_time_ms: float = 0.0


class ExtractionResult(BaseModel):
    """Result from structured extraction."""

    data: dict[str, Any]
    source_text: str
    model_used: str = ""
    confidence: float = 1.0


class VisionModelClient:
    """Client for vision model API (OpenAI-compatible).

    This client connects to a local vision model server for:
    - PDF to Markdown conversion
    - Image to Markdown conversion
    - Chart analysis and description
    - Structured data extraction

    Example:
        >>> client = VisionModelClient()
        >>> result = await client.pdf_to_markdown("document.pdf")
        >>> print(result.content)
    """

    def __init__(self, config: Optional[VisionModelConfig] = None) -> None:
        """Initialize vision model client.

        Args:
            config: Vision model configuration.
        """
        self._config = config or VisionModelConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout,
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.

        Returns:
            HTTP client instance.
        """
        if not self._client:
            await self.initialize()
        return self._client  # type: ignore

    async def _call_chat_api(
        self,
        messages: list[dict[str, Any]],
        model: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Call the chat completions API.

        Args:
            messages: Chat messages.
            model: Model to use.
            response_format: Response format (e.g., {"type": "json_object"}).

        Returns:
            API response.
        """
        client = await self._ensure_client()
        model = model or self._config.vision_model

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }

        if response_format:
            payload["response_format"] = response_format

        for attempt in range(self._config.max_retries):
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = self._config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except httpx.RequestError as e:
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _encode_file_to_base64(self, file_path: Union[str, Path]) -> str:
        """Encode a file to base64.

        Args:
            file_path: Path to the file.

        Returns:
            Base64 encoded string.
        """
        path = Path(file_path)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> tuple[str, str]:
        """Encode an image to base64 with media type.

        Args:
            image_path: Path to the image.

        Returns:
            Tuple of (base64_string, media_type).
        """
        path = Path(image_path)
        suffix = path.suffix.lower()

        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        media_type = media_types.get(suffix, "image/png")
        base64_data = self._encode_file_to_base64(path)

        return base64_data, media_type

    # ==================== PDF Conversion ====================

    async def pdf_to_markdown(
        self,
        pdf_path: Union[str, Path],
        page_range: Optional[tuple[int, int]] = None,
    ) -> ConversionResult:
        """Convert a PDF file to Markdown.

        Args:
            pdf_path: Path to the PDF file.
            page_range: Optional (start, end) page range (1-indexed).

        Returns:
            Conversion result with markdown content.
        """
        import time
        start_time = time.time()

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Encode PDF to base64
        base64_pdf = self._encode_file_to_base64(path)

        # Build prompt
        prompt = """Convert this PDF document to well-structured Markdown format.

Requirements:
1. Preserve all text content accurately
2. Convert tables to Markdown tables
3. Use appropriate headers (##, ###, etc.)
4. Preserve list formatting (numbered and bulleted)
5. Include any captions for images/figures as text descriptions
6. Maintain document structure and hierarchy
7. Do NOT add any commentary - only the converted content

Convert the PDF content to Markdown:"""

        if page_range:
            prompt = f"{prompt}\n\nOnly convert pages {page_range[0]} to {page_range[1]}."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": path.name,
                            "file_data": f"data:application/pdf;base64,{base64_pdf}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = await self._call_chat_api(messages, model=self._config.vision_model)

        content = response["choices"][0]["message"]["content"]
        processing_time = (time.time() - start_time) * 1000

        return ConversionResult(
            content=content,
            source_path=str(path),
            metadata={"original_filename": path.name},
            model_used=response.get("model", self._config.vision_model),
            processing_time_ms=processing_time,
        )

    async def pdf_page_to_markdown(
        self,
        pdf_path: Union[str, Path],
        page_number: int,
    ) -> ConversionResult:
        """Convert a single PDF page to Markdown.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-indexed).

        Returns:
            Conversion result with markdown content.
        """
        return await self.pdf_to_markdown(pdf_path, page_range=(page_number, page_number))

    # ==================== Image Conversion ====================

    async def image_to_markdown(
        self,
        image_path: Union[str, Path],
        context: Optional[str] = None,
    ) -> ConversionResult:
        """Convert an image to Markdown description.

        Args:
            image_path: Path to the image file.
            context: Optional context about the image.

        Returns:
            Conversion result with markdown description.
        """
        import time
        start_time = time.time()

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        base64_image, media_type = self._encode_image_to_base64(path)

        prompt = """Analyze this image and create a detailed Markdown description.

Requirements:
1. Describe the main content clearly
2. If it's a chart/graph, describe the data, axes, trends, and insights
3. If it's a diagram, explain the components and relationships
4. Use Markdown formatting for structure
5. Include any visible text verbatim

Convert the image to Markdown:"""

        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = await self._call_chat_api(messages, model=self._config.vision_model)

        content = response["choices"][0]["message"]["content"]
        processing_time = (time.time() - start_time) * 1000

        return ConversionResult(
            content=content,
            source_path=str(path),
            metadata={"media_type": media_type, "original_filename": path.name},
            model_used=response.get("model", self._config.vision_model),
            processing_time_ms=processing_time,
        )

    async def chart_to_analysis(
        self,
        image_path: Union[str, Path],
        analysis_type: str = "trading",
    ) -> ConversionResult:
        """Analyze a chart image and extract insights.

        Args:
            image_path: Path to the chart image.
            analysis_type: Type of analysis ("trading", "general").

        Returns:
            Conversion result with analysis.
        """
        import time
        start_time = time.time()

        path = Path(image_path)
        base64_image, media_type = self._encode_image_to_base64(path)

        if analysis_type == "trading":
            prompt = """Analyze this trading chart and provide:

1. **Asset Information**: What asset/timeframe is shown
2. **Trend Analysis**: Current trend direction and strength
3. **Key Levels**: Support and resistance levels visible
4. **Patterns**: Any chart patterns (head & shoulders, triangles, etc.)
5. **Indicators**: Any technical indicators visible and their readings
6. **Potential Setup**: Any trading setups visible
7. **Risk Areas**: Key levels to watch for invalidation

Provide a structured Markdown analysis:"""
        else:
            prompt = """Analyze this chart and provide:

1. **Chart Type**: What type of chart is shown
2. **Data Summary**: What data is being visualized
3. **Key Insights**: Main trends and patterns
4. **Notable Points**: Significant data points or anomalies
5. **Conclusions**: What the chart indicates

Provide a structured Markdown analysis:"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{base64_image}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        response = await self._call_chat_api(messages, model=self._config.vision_model)

        content = response["choices"][0]["message"]["content"]
        processing_time = (time.time() - start_time) * 1000

        return ConversionResult(
            content=content,
            source_path=str(path),
            metadata={"analysis_type": analysis_type, "media_type": media_type},
            model_used=response.get("model", self._config.vision_model),
            processing_time_ms=processing_time,
        )

    # ==================== Structured Extraction ====================

    async def extract_structured(
        self,
        text: str,
        schema: dict[str, Any],
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract structured data from text.

        Args:
            text: Text to extract from.
            schema: JSON schema for the expected output.
            model: Model to use (defaults to text_model).

        Returns:
            Extraction result with structured data.
        """
        import json

        prompt = f"""Extract structured information from the following text.

Output Schema:
```json
{json.dumps(schema, indent=2)}
```

Text to analyze:
```
{text}
```

Return only valid JSON matching the schema. No additional text or explanation."""

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response = await self._call_chat_api(
            messages,
            model=model or self._config.text_model,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("Failed to parse JSON response")

        return ExtractionResult(
            data=data,
            source_text=text,
            model_used=response.get("model", self._config.text_model),
        )

    async def extract_price_targets(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract price targets from trading content.

        Args:
            text: Text to extract from.
            model: Model to use.

        Returns:
            Extraction result with price targets.
        """
        schema = {
            "type": "object",
            "properties": {
                "price_targets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Asset symbol (e.g., BTC, ETH)"},
                            "target_price": {"type": "number", "description": "Target price"},
                            "timeframe": {"type": "string", "description": "Expected timeframe"},
                            "confidence": {"type": "string", "description": "Confidence level if mentioned"},
                            "rationale": {"type": "string", "description": "Reasoning for the target"},
                        },
                    },
                },
                "market_bias": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "Overall market sentiment",
                },
            },
        }

        return await self.extract_structured(text, schema, model)

    async def extract_trading_setup(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract trading setup from content.

        Args:
            text: Text to extract from.
            model: Model to use.

        Returns:
            Extraction result with trading setup.
        """
        schema = {
            "type": "object",
            "properties": {
                "setups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "setup_type": {
                                "type": "string",
                                "enum": ["long", "short", "range", "breakout"],
                            },
                            "symbol": {"type": "string"},
                            "entry_conditions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "entry_price": {"type": "number"},
                            "stop_loss": {"type": "number"},
                            "take_profit_levels": {
                                "type": "array",
                                "items": {"type": "number"},
                            },
                            "risk_reward_ratio": {"type": "number"},
                        },
                    },
                },
            },
        }

        return await self.extract_structured(text, schema, model)

    # ==================== Health Check ====================

    async def health_check(self) -> dict[str, Any]:
        """Check if the vision model API is healthy.

        Returns:
            Health status dictionary.
        """
        try:
            client = await self._ensure_client()
            response = await client.get("/v1/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            model_ids = [m.get("id") for m in models]

            return {
                "status": "healthy",
                "base_url": self._config.base_url,
                "available_models": model_ids,
                "vision_model_available": self._config.vision_model in model_ids,
                "text_model_available": self._config.text_model in model_ids,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self._config.base_url,
            }

    # ==================== Context Managers ====================

    async def __aenter__(self) -> "VisionModelClient":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
