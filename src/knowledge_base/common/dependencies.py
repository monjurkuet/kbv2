"""
Common dependency injection utilities.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

# Global session factory
_session_factory = None


def set_session_factory(factory):
    """Set the global session factory."""
    global _session_factory
    _session_factory = factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    if _session_factory is None:
        raise RuntimeError(
            "Session factory not initialized. Call set_session_factory first."
        )

    async with _session_factory() as session:
        yield session
