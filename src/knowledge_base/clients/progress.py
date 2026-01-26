"""
Progress visualization utilities for KBV2 WebSocket client.

This module provides beautiful progress displays using the rich library.
"""

from typing import Optional
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from rich.text import Text
from knowledge_base.clients.websocket_client import ProgressUpdate

STAGE_DESCRIPTIONS = {
    0: "Initializing ingestion pipeline",
    1: "Reading and parsing document",
    2: "Partitioning document into chunks",
    3: "Extracting entities from text",
    4: "Resolving and deduplicating entities",
    5: "Generating embeddings for chunks",
    6: "Building knowledge graph connections",
    7: "Creating document metadata",
    8: "Finalizing and storing results",
    9: "Completing ingestion process",
}


class ProgressVisualizer:
    """Visualizes progress updates from the ingestion pipeline."""

    def __init__(self, verbose: bool = True):
        """Initialize the progress visualizer.

        Args:
            verbose: Whether to show detailed progress information
        """
        self.console = Console()
        self.verbose = verbose
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None
        self.start_time: Optional[float] = None
        self.last_stage = -1

    def start(self, total_stages: int = 10) -> None:
        """Start the progress visualization.

        Args:
            total_stages: Total number of stages in the pipeline
        """
        if not self.verbose:
            return

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            "[cyan]Document Ingestion", total=total_stages
        )
        self.start_time = None

    def update(self, progress_update: ProgressUpdate) -> None:
        """Update the progress visualization.

        Args:
            progress_update: Progress update from the server
        """
        if not self.verbose or not self.progress:
            return

        stage = progress_update.stage
        status = progress_update.status
        message = progress_update.message
        duration = progress_update.duration

        if self.start_time is None:
            self.start_time = progress_update.timestamp

        if stage != self.last_stage:
            self.last_stage = stage
            stage_desc = STAGE_DESCRIPTIONS.get(stage, f"Stage {stage}")
            self.progress.update(
                self.task_id,
                description=f"[cyan]Stage {stage}: {stage_desc}",
                advance=1,
            )

        status_color = self._get_status_color(status)
        self.console.print(
            f"[{status_color}]{status}[/{status_color}]: {message}",
            highlight=False,
        )

        if duration and duration > 0:
            self.console.print(
                f"[dim]Duration: {duration:.2f}s[/dim]",
                highlight=False,
            )

    def complete(self, result: str, success: bool = True) -> None:
        """Complete the progress visualization.

        Args:
            result: Result message to display
            success: Whether the operation was successful
        """
        if not self.verbose:
            return

        if self.progress:
            self.progress.stop()

        color = "green" if success else "red"
        emoji = "✓" if success else "✗"

        panel = Panel(
            Text(f"{emoji} {result}", style=f"bold {color}"),
            title="[bold]Ingestion Complete[/bold]"
            if success
            else "[bold]Ingestion Failed[/bold]",
            border_style=color,
            padding=(1, 2),
        )
        self.console.print(panel)

    def error(self, error_message: str) -> None:
        """Display an error message.

        Args:
            error_message: Error message to display
        """
        if not self.verbose:
            return

        if self.progress:
            self.progress.stop()

        panel = Panel(
            Text(f"✗ {error_message}", style="bold red"),
            title="[bold]Error[/bold]",
            border_style="red",
            padding=(1, 2),
        )
        self.console.print(panel)

    def info(self, message: str) -> None:
        """Display an informational message.

        Args:
            message: Informational message to display
        """
        if not self.verbose:
            return

        self.console.print(f"[dim]ℹ {message}[/dim]", highlight=False)

    def warning(self, message: str) -> None:
        """Display a warning message.

        Args:
            message: Warning message to display
        """
        if not self.verbose:
            return

        self.console.print(f"[yellow]⚠ {message}[/yellow]", highlight=False)

    def success(self, message: str) -> None:
        """Display a success message.

        Args:
            message: Success message to display
        """
        if not self.verbose:
            return

        self.console.print(f"[green]✓ {message}[/green]", highlight=False)

    def _get_status_color(self, status: str) -> str:
        """Get color for a status string.

        Args:
            status: Status string

        Returns:
            Color string for rich formatting
        """
        status_lower = status.lower()
        if status_lower in ("started", "processing", "progress"):
            return "blue"
        elif status_lower in ("completed", "success", "finished"):
            return "green"
        elif status_lower in ("error", "failed", "exception"):
            return "red"
        elif status_lower in ("warning", "pending"):
            return "yellow"
        else:
            return "white"
