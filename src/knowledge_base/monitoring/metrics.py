"""Monitoring and metrics for KBv2 Crypto Knowledgebase.

Provides Prometheus-compatible metrics and health checks for production deployment.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class MetricType(str):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class MetricValue:
    """A single metric value."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = MetricType.GAUGE


class MetricsCollector:
    """Collects and exposes metrics for the knowledgebase."""

    def __init__(self):
        self.metrics: Dict[str, List[MetricValue]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Document processing metrics
        self.documents_processed = 0
        self.documents_failed = 0
        self.total_entities_extracted = 0
        self.total_relationships_extracted = 0

        # Experience Bank metrics
        self.experiences_stored = 0
        self.experiences_retrieved = 0
        self.experience_bank_hits = 0
        self.experience_bank_misses = 0

        # Quality metrics
        self.extraction_quality_scores: List[float] = []
        self.validation_scores: List[float] = []

        # Domain distribution
        self.domain_counts: Dict[str, int] = defaultdict(int)

    def record_document_processed(
        self,
        domain: str,
        entity_count: int,
        relationship_count: int,
        processing_time_ms: float,
        success: bool = True,
    ):
        """Record document processing metrics."""
        if success:
            self.documents_processed += 1
            self.total_entities_extracted += entity_count
            self.total_relationships_extracted += relationship_count
        else:
            self.documents_failed += 1

        self.domain_counts[domain] += 1

        # Store metric
        self._record("documents_processed_total", self.documents_processed)
        self._record("entities_extracted_total", self.total_entities_extracted)
        self._record("processing_time_ms", processing_time_ms)
        self._record("document_entities", entity_count, {"domain": domain})

    def record_experience_bank_usage(
        self, stored: bool = False, retrieved: int = 0, hit: bool = False
    ):
        """Record Experience Bank usage metrics."""
        if stored:
            self.experiences_stored += 1

        self.experiences_retrieved += retrieved

        if hit:
            self.experience_bank_hits += 1
        else:
            self.experience_bank_misses += 1

        self._record("experiences_stored_total", self.experiences_stored)
        self._record("experiences_retrieved_total", self.experiences_retrieved)

    def record_extraction_quality(self, quality_score: float):
        """Record extraction quality score."""
        self.extraction_quality_scores.append(quality_score)
        self._record("extraction_quality", quality_score)

    def record_validation(self, validation_score: float, errors: int, warnings: int):
        """Record ontology validation results."""
        self.validation_scores.append(validation_score)
        self._record("validation_score", validation_score)
        self._record("validation_errors", errors)
        self._record("validation_warnings", warnings)

    def _record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric = MetricValue(name=name, value=value, labels=labels or {})
        self.metrics[name].append(metric)

        # Keep only last 1000 values per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Document processing metrics
        lines.append("# HELP kb_documents_processed_total Total documents processed")
        lines.append("# TYPE kb_documents_processed_total counter")
        lines.append(f"kb_documents_processed_total {self.documents_processed}")

        lines.append("# HELP kb_documents_failed_total Total documents failed")
        lines.append("# TYPE kb_documents_failed_total counter")
        lines.append(f"kb_documents_failed_total {self.documents_failed}")

        lines.append("# HELP kb_entities_extracted_total Total entities extracted")
        lines.append("# TYPE kb_entities_extracted_total counter")
        lines.append(f"kb_entities_extracted_total {self.total_entities_extracted}")

        lines.append(
            "# HELP kb_relationships_extracted_total Total relationships extracted"
        )
        lines.append("# TYPE kb_relationships_extracted_total counter")
        lines.append(
            f"kb_relationships_extracted_total {self.total_relationships_extracted}"
        )

        # Experience Bank metrics
        lines.append("# HELP kb_experience_bank_stored_total Experiences stored")
        lines.append("# TYPE kb_experience_bank_stored_total counter")
        lines.append(f"kb_experience_bank_stored_total {self.experiences_stored}")

        lines.append("# HELP kb_experience_bank_hits_total Cache hits")
        lines.append("# TYPE kb_experience_bank_hits_total counter")
        lines.append(f"kb_experience_bank_hits_total {self.experience_bank_hits}")

        lines.append("# HELP kb_experience_bank_misses_total Cache misses")
        lines.append("# TYPE kb_experience_bank_misses_total counter")
        lines.append(f"kb_experience_bank_misses_total {self.experience_bank_misses}")

        if self.experience_bank_hits + self.experience_bank_misses > 0:
            hit_rate = self.experience_bank_hits / (
                self.experience_bank_hits + self.experience_bank_misses
            )
            lines.append("# HELP kb_experience_bank_hit_rate Experience bank hit rate")
            lines.append("# TYPE kb_experience_bank_hit_rate gauge")
            lines.append(f"kb_experience_bank_hit_rate {hit_rate:.4f}")

        # Domain distribution
        lines.append("# HELP kb_documents_by_domain Documents processed by domain")
        lines.append("# TYPE kb_documents_by_domain gauge")
        for domain, count in self.domain_counts.items():
            lines.append(f'kb_documents_by_domain{{domain="{domain}"}} {count}')

        # Quality metrics
        if self.extraction_quality_scores:
            avg_quality = sum(self.extraction_quality_scores) / len(
                self.extraction_quality_scores
            )
            lines.append("# HELP kb_extraction_quality_avg Average extraction quality")
            lines.append("# TYPE kb_extraction_quality_avg gauge")
            lines.append(f"kb_extraction_quality_avg {avg_quality:.4f}")

        if self.validation_scores:
            avg_validation = sum(self.validation_scores) / len(self.validation_scores)
            lines.append("# HELP kb_validation_score_avg Average validation score")
            lines.append("# TYPE kb_validation_score_avg gauge")
            lines.append(f"kb_validation_score_avg {avg_validation:.4f}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        total_requests = self.experience_bank_hits + self.experience_bank_misses
        hit_rate = (
            self.experience_bank_hits / total_requests if total_requests > 0 else 0.0
        )

        return {
            "documents": {
                "processed": self.documents_processed,
                "failed": self.documents_failed,
                "success_rate": 1.0
                - (self.documents_failed / max(self.documents_processed, 1)),
            },
            "extraction": {
                "total_entities": self.total_entities_extracted,
                "total_relationships": self.total_relationships_extracted,
                "avg_quality": sum(self.extraction_quality_scores)
                / len(self.extraction_quality_scores)
                if self.extraction_quality_scores
                else 0.0,
            },
            "experience_bank": {
                "stored": self.experiences_stored,
                "retrieved": self.experiences_retrieved,
                "hit_rate": hit_rate,
            },
            "domains": dict(self.domain_counts),
        }


class HealthStatus(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    version: str = "2.0.0"
    checks: Dict[str, Any]


class HealthChecker:
    """Health check for KBv2 components."""

    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.last_check: Optional[datetime] = None
        self.healthy = True

    def register_check(self, name: str, check_fn: Callable[[], bool]):
        """Register a health check function."""
        self.checks[name] = check_fn

    async def check_health(self) -> HealthStatus:
        """Run all health checks."""
        checks = {}
        all_healthy = True

        for name, check_fn in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await check_fn()
                else:
                    result = check_fn()
                checks[name] = {"healthy": result}
                if not result:
                    all_healthy = False
            except Exception as e:
                checks[name] = {"healthy": False, "error": str(e)}
                all_healthy = False

        self.last_check = datetime.utcnow()
        self.healthy = all_healthy

        return HealthStatus(
            status="healthy" if all_healthy else "unhealthy",
            timestamp=self.last_check,
            checks=checks,
        )


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()


def create_monitoring_app() -> FastAPI:
    """Create FastAPI app for monitoring endpoints."""
    app = FastAPI(title="KBv2 Monitoring")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        status = await health_checker.check_health()
        if status.status == "healthy":
            return status
        else:
            raise HTTPException(status_code=503, detail=status.dict())

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return metrics_collector.get_prometheus_format()

    @app.get("/stats")
    async def stats():
        """Statistics endpoint."""
        return metrics_collector.get_stats()

    return app


# Decorator for tracking function metrics
def track_metrics(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track function execution metrics."""

    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                metrics_collector._record(
                    f"{metric_name}_duration_ms",
                    (time.time() - start_time) * 1000,
                    labels,
                )
                metrics_collector._record(f"{metric_name}_success", 1, labels)
                return result
            except Exception as e:
                metrics_collector._record(
                    f"{metric_name}_duration_ms",
                    (time.time() - start_time) * 1000,
                    labels,
                )
                metrics_collector._record(f"{metric_name}_success", 0, labels)
                raise

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                metrics_collector._record(
                    f"{metric_name}_duration_ms",
                    (time.time() - start_time) * 1000,
                    labels,
                )
                metrics_collector._record(f"{metric_name}_success", 1, labels)
                return result
            except Exception as e:
                metrics_collector._record(
                    f"{metric_name}_duration_ms",
                    (time.time() - start_time) * 1000,
                    labels,
                )
                metrics_collector._record(f"{metric_name}_success", 0, labels)
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator
