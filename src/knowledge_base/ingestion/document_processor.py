"""Document processor for various file types.

This module provides unified document processing that:
1. Detects file type and routes to appropriate processor
2. Extracts text and metadata from documents
3. Converts to standardized markdown format
4. Handles batch processing with progress tracking
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field

from knowledge_base.ingestion.vision_client import VisionModelClient, ConversionResult

logger = logging.getLogger(__name__)


class ProcessedDocument(BaseModel):
    """Processed document result."""

    id: str = Field(default_factory=lambda: "")
    name: str
    source_path: str
    content: str
    mime_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "processed"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.id:
            # Generate ID from content hash
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.id = f"doc_{content_hash}"


@dataclass
class ProcessingProgress:
    """Progress tracking for batch processing."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    current_file: str = ""
    errors: list[str] = field(default_factory=list)


class DocumentProcessor:
    """Unified document processor for various file types.

    Supports:
    - PDF documents (via vision model)
    - Images (via vision model)
    - Markdown files (direct)
    - Text files (direct)
    - HTML files (via unstructured)

    Example:
        >>> processor = DocumentProcessor(vision_client)
        >>> doc = await processor.process("document.pdf")
        >>> print(doc.content)
    """

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Documents
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        # Images
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        # Text
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".rst": "text/x-rst",
        # Web
        ".html": "text/html",
        ".htm": "text/html",
        # Data
        ".json": "application/json",
        ".csv": "text/csv",
    }

    def __init__(
        self,
        vision_client: Optional[VisionModelClient] = None,
        max_file_size_mb: int = 100,
    ) -> None:
        """Initialize document processor.

        Args:
            vision_client: Vision model client for PDF/image processing.
            max_file_size_mb: Maximum file size in MB.
        """
        self._vision_client = vision_client
        self._max_file_size = max_file_size_mb * 1024 * 1024

    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type from file extension.

        Args:
            file_path: Path to the file.

        Returns:
            MIME type string.
        """
        suffix = file_path.suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(suffix, "application/octet-stream")

    def _validate_file(self, file_path: Path) -> None:
        """Validate file exists and is within size limits.

        Args:
            file_path: Path to the file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file is too large or unsupported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > self._max_file_size:
            raise ValueError(
                f"File too large: {file_path} ({file_size / (1024*1024):.1f}MB > {self._max_file_size / (1024*1024):.1f}MB limit)"
            )

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # ==================== Processing Methods ====================

    async def process(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """Process a single document.

        Args:
            file_path: Path to the document.

        Returns:
            Processed document with content and metadata.
        """
        path = Path(file_path)
        self._validate_file(path)

        mime_type = self._detect_mime_type(path)

        # Route to appropriate processor
        if mime_type == "application/pdf":
            result = await self._process_pdf(path)
        elif mime_type.startswith("image/"):
            result = await self._process_image(path)
        elif mime_type in ("text/plain", "text/markdown", "text/x-rst"):
            result = await self._process_text(path)
        elif mime_type in ("text/html", "text/htm"):
            result = await self._process_html(path)
        elif mime_type == "application/json":
            result = await self._process_json(path)
        elif mime_type == "text/csv":
            result = await self._process_csv(path)
        else:
            raise ValueError(f"Unsupported MIME type: {mime_type}")

        return ProcessedDocument(
            name=path.name,
            source_path=str(path),
            content=result.content,
            mime_type=mime_type,
            metadata={
                **result.metadata,
                "original_size_bytes": path.stat().st_size,
                "processed_at": datetime.utcnow().isoformat(),
            },
        )

    async def _process_pdf(self, path: Path) -> ConversionResult:
        """Process a PDF document.

        Args:
            path: Path to the PDF.

        Returns:
            Conversion result.
        """
        if not self._vision_client:
            raise RuntimeError("Vision client required for PDF processing")

        return await self._vision_client.pdf_to_markdown(path)

    async def _process_image(self, path: Path) -> ConversionResult:
        """Process an image.

        Args:
            path: Path to the image.

        Returns:
            Conversion result.
        """
        if not self._vision_client:
            raise RuntimeError("Vision client required for image processing")

        return await self._vision_client.image_to_markdown(path)

    async def _process_text(self, path: Path) -> ConversionResult:
        """Process a text file.

        Args:
            path: Path to the text file.

        Returns:
            Conversion result.
        """
        content = await asyncio.get_event_loop().run_in_executor(
            None, path.read_text, "utf-8"
        )

        # Add frontmatter for markdown files
        if path.suffix.lower() in (".md", ".markdown"):
            # Check if already has frontmatter
            if not content.startswith("---"):
                frontmatter = f"""---
source: {path.name}
processed: {datetime.utcnow().isoformat()}
---

"""
                content = frontmatter + content

        return ConversionResult(
            content=content,
            source_path=str(path),
            metadata={"processor": "text"},
        )

    async def _process_html(self, path: Path) -> ConversionResult:
        """Process an HTML file.

        Args:
            path: Path to the HTML file.

        Returns:
            Conversion result.
        """
        try:
            from unstructured.partition.html import partition_html

            def _partition():
                elements = partition_html(filename=str(path))
                return "\n\n".join(str(el) for el in elements)

            content = await asyncio.get_event_loop().run_in_executor(None, _partition)

            return ConversionResult(
                content=content,
                source_path=str(path),
                metadata={"processor": "html"},
            )
        except ImportError:
            # Fallback to simple text extraction
            logger.warning("unstructured not installed, using basic HTML processing")
            import re
            from bs4 import BeautifulSoup

            html_content = await asyncio.get_event_loop().run_in_executor(
                None, path.read_text, "utf-8"
            )
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()

            text = soup.get_text(separator="\n")
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)

            return ConversionResult(
                content=text,
                source_path=str(path),
                metadata={"processor": "html_basic"},
            )

    async def _process_json(self, path: Path) -> ConversionResult:
        """Process a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Conversion result.
        """
        import json

        content = await asyncio.get_event_loop().run_in_executor(
            None, path.read_text, "utf-8"
        )
        data = json.loads(content)

        # Convert to readable markdown
        if isinstance(data, list):
            markdown = "# Data\n\n"
            for i, item in enumerate(data, 1):
                markdown += f"## Item {i}\n\n```json\n{json.dumps(item, indent=2)}\n```\n\n"
        elif isinstance(data, dict):
            markdown = f"# Data\n\n```json\n{json.dumps(data, indent=2)}\n```\n"
        else:
            markdown = f"# Data\n\n{data}"

        return ConversionResult(
            content=markdown,
            source_path=str(path),
            metadata={"processor": "json"},
        )

    async def _process_csv(self, path: Path) -> ConversionResult:
        """Process a CSV file.

        Args:
            path: Path to the CSV file.

        Returns:
            Conversion result.
        """
        import csv

        def _read_csv():
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            return rows

        rows = await asyncio.get_event_loop().run_in_executor(None, _read_csv)

        if not rows:
            return ConversionResult(
                content="",
                source_path=str(path),
                metadata={"processor": "csv", "rows": 0},
            )

        # Convert to markdown table
        headers = rows[0]
        markdown = f"# {path.stem}\n\n"
        markdown += "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in rows[1:100]:  # Limit to 100 rows
            markdown += "| " + " | ".join(row) + " |\n"

        if len(rows) > 101:
            markdown += f"\n*...and {len(rows) - 101} more rows*\n"

        return ConversionResult(
            content=markdown,
            source_path=str(path),
            metadata={"processor": "csv", "rows": len(rows)},
        )

    # ==================== Batch Processing ====================

    async def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> list[ProcessedDocument]:
        """Process all documents in a directory.

        Args:
            directory: Directory path.
            recursive: Process subdirectories.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of processed documents.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        # Find all supported files
        if recursive:
            files = [
                f for f in dir_path.rglob("*")
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        else:
            files = [
                f for f in dir_path.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

        progress = ProcessingProgress(total=len(files))
        results = []

        for file_path in files:
            progress.current_file = file_path.name

            try:
                doc = await self.process(file_path)
                results.append(doc)
                progress.completed += 1
            except Exception as e:
                progress.failed += 1
                progress.errors.append(f"{file_path.name}: {str(e)}")
                logger.error(f"Failed to process {file_path}: {e}")

            if progress_callback:
                progress_callback(progress)

        return results

    async def process_batch(
        self,
        file_paths: list[Union[str, Path]],
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> list[ProcessedDocument]:
        """Process multiple files.

        Args:
            file_paths: List of file paths.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of processed documents.
        """
        progress = ProcessingProgress(total=len(file_paths))
        results = []

        for file_path in file_paths:
            path = Path(file_path)
            progress.current_file = path.name

            try:
                doc = await self.process(path)
                results.append(doc)
                progress.completed += 1
            except Exception as e:
                progress.failed += 1
                progress.errors.append(f"{path.name}: {str(e)}")
                logger.error(f"Failed to process {path}: {e}")

            if progress_callback:
                progress_callback(progress)

        return results

    # ==================== Utility Methods ====================

    def compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Content to hash.

        Returns:
            Hash string.
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of extensions (e.g., [".pdf", ".txt"]).
        """
        return list(self.SUPPORTED_EXTENSIONS.keys())
