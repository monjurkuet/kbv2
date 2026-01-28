"""
Command-line interface for KBV2 WebSocket client.

This module provides an interactive CLI for ingesting documents and interacting
with the KBV2 Knowledge Base API via WebSocket.
"""

import argparse
import asyncio
import os
import sys
import logging
from pathlib import Path
from typing import Optional

from knowledge_base.clients.websocket_client import KBV2WebSocketClient
from knowledge_base.clients.progress import ProgressVisualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".html"}


def validate_file_type(file_path: str) -> bool:
    """Validate that the file has a supported extension.

    Args:
        file_path: Path to the file to validate.

    Returns:
        True if the file type is supported.

    Raises:
        ValueError: If the file type is not supported.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return True


class IngestionCLI:
    """Command-line interface for document ingestion."""

    def __init__(self, args):
        """Initialize the CLI with parsed arguments.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.visualizer = ProgressVisualizer(verbose=args.verbose)

    async def run(self) -> int:
        """Run the ingestion process.

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        logger.info("=" * 80)
        logger.info("Starting KBV2 document ingestion")
        logger.info("=" * 80)

        file_path = self.args.file
        logger.info(f"File path: {file_path}")

        if not file_path:
            logger.error("File path is required")
            self.visualizer.error("File path is required")
            return 1

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            self.visualizer.error(f"File not found: {file_path}")
            return 1

        if not os.path.isfile(file_path):
            logger.error(f"Not a file: {file_path}")
            self.visualizer.error(f"Not a file: {file_path}")
            return 1

        try:
            validate_file_type(file_path)
        except ValueError as e:
            logger.error(str(e))
            self.visualizer.error(str(e))
            return 1

        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size:,} bytes")
        self.visualizer.info(f"Starting ingestion of {file_path} ({file_size:,} bytes)")

        if self.args.document_name:
            logger.info(f"Document name: {self.args.document_name}")
            self.visualizer.info(f"Document name: {self.args.document_name}")
        if self.args.domain:
            logger.info(f"Domain: {self.args.domain}")
            self.visualizer.info(f"Domain: {self.args.domain}")

        logger.info(f"Server: {self.args.host}:{self.args.port}")
        logger.info(f"Timeout: {self.args.timeout}s")
        logger.info(f"Verbose: {self.args.verbose}")

        try:
            logger.info("Creating WebSocket client...")
            async with KBV2WebSocketClient(
                host=self.args.host,
                port=self.args.port,
                timeout=self.args.timeout,
                progress_callback=self.visualizer.update,
            ) as client:
                logger.info("WebSocket client created successfully")
                self.visualizer.start()
                logger.info("Starting document ingestion request...")

                response = await client.ingest_document(
                    file_path=file_path,
                    document_name=self.args.document_name,
                    domain=self.args.domain,
                )

                logger.info(
                    f"Received response: error={response.error}, result={response.result}"
                )
                if response.error:
                    self.visualizer.complete(f"Failed: {response.error}", success=False)
                    return 1
                else:
                    self.visualizer.complete(
                        f"Successfully ingested document", success=True
                    )
                    if self.args.verbose and response.result:
                        self.visualizer.info(f"Result: {response.result}")
                    return 0

        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            self.visualizer.error(f"Connection error: {e}")
            return 1
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout error after {self.args.timeout}s: {e}")
            self.visualizer.error(
                f"Timeout error after {self.args.timeout}s - The ingestion is taking longer than expected"
            )
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.visualizer.error(f"Unexpected error: {e}")
            if self.args.verbose:
                import traceback

                traceback.print_exc()
            return 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="KBV2 Document Ingestion Client via WebSocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ingest /path/to/document.md
  %(prog)s ingest /path/to/document.md --name "My Document" --domain "technical"
  %(prog)s ingest /path/to/document.md --host localhost --port 8765 --verbose
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest a document into the knowledge base"
    )
    ingest_parser.add_argument("file", help="Path to the document file to ingest")
    ingest_parser.add_argument(
        "--name",
        "--document-name",
        dest="document_name",
        help="Optional document name",
    )
    ingest_parser.add_argument(
        "--domain",
        help="Optional document domain (e.g., 'technical', 'finance', 'medical')",
    )
    ingest_parser.add_argument(
        "--host",
        default="localhost",
        help="KBV2 server host (default: localhost)",
    )
    ingest_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="KBV2 server port (default: 8765)",
    )
    ingest_parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Request timeout in seconds (default: 3600 = 1 hour)",
    )
    ingest_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed progress information",
    )
    ingest_parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    if not args.command:
        print("Error: No command specified")
        print("Use --help for usage information")
        return 1

    if args.command == "ingest":
        if args.quiet:
            args.verbose = False

        cli = IngestionCLI(args)
        try:
            return asyncio.run(cli.run())
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            return 130
    else:
        print(f"Error: Unknown command '{args.command}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
