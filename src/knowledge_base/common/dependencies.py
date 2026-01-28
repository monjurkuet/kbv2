"""
Common dependency injection utilities.
"""

from typing import AsyncGenerator, Callable, Optional
from sqlalchemy.ext.asyncio import AsyncSession

# Global session factory
_session_factory: Optional[Callable[[], AsyncGenerator[AsyncSession, None]]] = None


def set_session_factory(factory):
    """Set the global session factory."""
    global _session_factory
    _session_factory = factory


def get_session_factory() -> Callable[[], AsyncGenerator[AsyncSession, None]]:
    """Get the global session factory.

    Returns:
        Callable that yields an async database session.

    Raises:
        RuntimeError: If session factory not initialized.
    """
    global _session_factory
    if _session_factory is None:
        raise RuntimeError(
            "Session factory not initialized. Call set_session_factory first."
        )
    return _session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
