#!/usr/bin/env python3
"""Batch ingest Bitcoin trading documents into the knowledge base."""

import asyncio
import logging
from pathlib import Path
from knowledge_base.orchestrator import IngestionOrchestrator
from knowledge_base.domain.domain import Domain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_trading_documents(
    input_dir: str,
    output_dir: str | None = None,
):
    """Batch ingest trading documents.

    Args:
        input_dir: Directory containing markdown trading documents
        output_dir: Optional output directory for reports
    """
    orchestrator = IngestionOrchestrator()
    await orchestrator.initialize()

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Find all markdown files
    md_files = list(input_path.glob("**/*.md"))
    logger.info(f"Found {len(md_files)} markdown files to process")

    results = []
    for i, md_file in enumerate(md_files, 1):
        logger.info(f"[{i}/{len(md_files)}] Processing: {md_file.name}")

        try:
            document = await orchestrator.process_document(
                file_path=str(md_file),
                document_name=md_file.stem,
                domain="CRYPTO_TRADING",
            )
            results.append(
                {
                    "file": md_file.name,
                    "status": "success",
                    "document_id": document.id,
                }
            )
            logger.info(f"  ✓ Processed successfully")

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            results.append(
                {
                    "file": md_file.name,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Print summary
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(
        f"\nSummary: {success_count}/{len(results)} documents processed successfully"
    )

    await orchestrator.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Bitcoin trading documents")
    parser.add_argument("input_dir", help="Directory containing markdown files")
    parser.add_argument("--output", "-o", help="Output directory for reports")

    args = parser.parse_args()

    asyncio.run(ingest_trading_documents(args.input_dir, args.output))
