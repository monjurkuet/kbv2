#!/usr/bin/env python3
"""
Unified CLI for KBV2 document ingestion.

This CLI can work in two modes:
1. Direct mode (default): Uses orchestrator directly (no server needed)
2. Server mode: Connects to running WebSocket server

Usage:
  # Direct ingestion (recommended)
  python ingest_cli.py /path/to/document.md --domain BITCOIN

  # Via WebSocket server (requires server running)
  python ingest_cli.py /path/to/document.md --domain BITCOIN --server

  # With custom settings
  python ingest_cli.py /path/to/document.md --domain DEFI --verbose
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_DOMAINS = [
    "BITCOIN",
    "DEFI",
    "INSTITUTIONAL_CRYPTO",
    "STABLECOINS",
    "CRYPTO_REGULATION",
    "GENERAL",
]


class IngestionCLI:
    """Unified CLI for document ingestion."""

    def __init__(self, args):
        self.args = args
        self.setup_logging()

    def setup_logging(self):
        """Configure logging based on verbosity."""
        if self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
            # Reduce noise from httpx
            logging.getLogger("httpx").setLevel(logging.WARNING)

    def validate_file(self, file_path: str) -> Path:
        """Validate the input file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        # Check extension
        supported = {".md", ".txt", ".pdf", ".docx", ".html", ".json"}
        if path.suffix.lower() not in supported:
            logger.warning(f"File extension {path.suffix} may not be fully supported")

        return path

    def validate_domain(self, domain: str) -> str:
        """Validate the domain."""
        domain = domain.upper()
        if domain not in SUPPORTED_DOMAINS:
            logger.warning(
                f"Domain '{domain}' not in standard list: {SUPPORTED_DOMAINS}"
            )
            logger.warning("Using domain anyway, but results may vary")
        return domain

    async def ingest_direct(self, file_path: Path, domain: str) -> bool:
        """Ingest document directly using orchestrator."""
        print(f"\n{'=' * 60}")
        print(f"KBV2 Document Ingestion (Direct Mode)")
        print(f"{'=' * 60}\n")

        print(f"📄 File: {file_path.name}")
        print(f"📁 Path: {file_path}")
        print(f"🏷️  Domain: {domain}")
        print(f"📊 Size: {file_path.stat().st_size:,} bytes")
        print()

        orchestrator = None
        try:
            print("🔄 Initializing SelfImprovingOrchestrator...")
            orchestrator = SelfImprovingOrchestrator()
            await orchestrator.initialize()
            print("✅ Orchestrator ready\n")

            print("⚙️  Processing document...")
            document = await orchestrator.process_document(
                file_path=str(file_path.absolute()),
                document_name=file_path.stem,
                domain=domain,
            )

            print(f"\n{'=' * 60}")
            print(f"✅ Ingestion Complete!")
            print(f"{'=' * 60}")
            print(f"📋 Document ID: {document.id}")
            print(f"📊 Status: {document.status}")
            print(f"🕐 Created: {document.created_at}")
            print(f"{'=' * 60}\n")

            return True

        except Exception as e:
            print(f"\n❌ Ingestion failed: {e}")
            if self.args.verbose:
                import traceback

                traceback.print_exc()
            return False
        finally:
            if orchestrator:
                await orchestrator.close()

    async def ingest_via_server(self, file_path: Path, domain: str) -> bool:
        """Ingest document via WebSocket server."""
        from knowledge_base.clients.websocket_client import KBV2WebSocketClient

        print(f"\n{'=' * 60}")
        print(f"KBV2 Document Ingestion (Server Mode)")
        print(f"{'=' * 60}\n")

        print(f"📄 File: {file_path.name}")
        print(f"🌐 Server: {self.args.host}:{self.args.port}")
        print(f"🏷️  Domain: {domain}")
        print()

        try:
            client = KBV2WebSocketClient(
                host=self.args.host,
                port=self.args.port,
                timeout=self.args.timeout,
            )

            print("🔄 Connecting to server...")
            await client.connect()
            print("✅ Connected\n")

            print("⚙️  Sending ingestion request...")
            response = await client.ingest_document(
                file_path=str(file_path.absolute()),
                document_name=file_path.stem,
                domain=domain,
            )

            if response.error:
                print(f"\n❌ Server error: {response.error}")
                return False

            result = response.result or {}
            print(f"\n{'=' * 60}")
            print(f"✅ Ingestion Complete!")
            print(f"{'=' * 60}")
            print(f"📋 Document ID: {result.get('document_id', 'N/A')}")
            print(f"📊 Status: {result.get('status', 'N/A')}")
            print(f"{'=' * 60}\n")

            return True

        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            print("💡 Tip: Use direct mode (without --server) or start the server:")
            print("   uv run python -m knowledge_base.production")
            return False

    async def run(self) -> int:
        """Run the CLI."""
        try:
            # Validate inputs
            file_path = self.validate_file(self.args.file)
            domain = self.validate_domain(self.args.domain)

            # Choose mode
            if self.args.server:
                success = await self.ingest_via_server(file_path, domain)
            else:
                success = await self.ingest_direct(file_path, domain)

            return 0 if success else 1

        except FileNotFoundError as e:
            print(f"\n❌ {e}")
            return 1
        except ValueError as e:
            print(f"\n❌ {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            if self.args.verbose:
                import traceback

                traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="KBV2 Document Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Direct ingestion (recommended - no server needed)
  %(prog)s document.md --domain BITCOIN
  
  # With verbose output
  %(prog)s document.md --domain DEFI --verbose
  
  # Via WebSocket server (requires server running)
  %(prog)s document.md --domain BITCOIN --server

Supported Domains:
  {", ".join(SUPPORTED_DOMAINS)}
        """,
    )

    parser.add_argument("file", help="Path to the document file to ingest")
    parser.add_argument(
        "--domain",
        "-d",
        default="GENERAL",
        help=f"Domain for extraction (default: GENERAL)",
    )
    parser.add_argument(
        "--server",
        "-s",
        action="store_true",
        help="Use WebSocket server mode (requires server running)",
    )
    parser.add_argument(
        "--host", "-H", default="localhost", help="Server host (only with --server)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8765, help="Server port (only with --server)"
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=3600,
        help="Timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    cli = IngestionCLI(args)
    exit_code = asyncio.run(cli.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
