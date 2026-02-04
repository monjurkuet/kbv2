"""Base service class for orchestration services."""

import logging
from abc import ABC, abstractmethod


class BaseService(ABC):
    """Base class for all orchestration services."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service and cleanup resources."""
        pass
