#!/usr/bin/env python3
"""Simple script to ingest markdown files directly without server."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base.orchestrator_self_improving import SelfImprovingOrchestrator


async def ingest_file(file_path: str, domain: str = "GENERAL"):
    """Ingest a single file directly using the orchestrator."""
    print(f"🔄 Initializing orchestrator...")
    orchestrator = SelfImprovingOrchestrator()
    await orchestrator.initialize()
    print(f"✅ Orchestrator initialized\n")

    try:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return

        print(f"📄 Ingesting: {path.name}")
        print(f"   Domain: {domain}")
        print(f"   Size: {path.stat().st_size:,} bytes")
        print()

        document = await orchestrator.process_document(
            file_path=str(path.absolute()),
            document_name=path.stem,
            domain=domain,
        )

        print(f"\n✅ Successfully ingested!")
        print(f"   Document ID: {document.id}")
        print(f"   Status: {document.status}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await orchestrator.close()

    print(f"\n✅ Done!")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <file_path> [domain]")
        print()
        print(
            "Domains: BITCOIN, DEFI, INSTITUTIONAL_CRYPTO, STABLECOINS, CRYPTO_REGULATION, GENERAL"
        )
        print()
        print("Example:")
        print("  python ingest.py ~/documents/bitcoin_report.md BITCOIN")
        sys.exit(1)

    file_path = sys.argv[1]
    domain = sys.argv[2] if len(sys.argv) > 2 else "GENERAL"

    exit_code = asyncio.run(ingest_file(file_path, domain))
    sys.exit(exit_code)
