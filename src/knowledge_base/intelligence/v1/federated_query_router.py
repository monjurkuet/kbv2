"""Federated query routing for multi-domain knowledge base queries."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Optional
from pydantic import BaseModel
from knowledge_base.intelligence.v1.hybrid_retriever import (
    HybridEntityRetriever,
    HybridRetrievalResult,
)
from knowledge_base.ingestion.v1.embedding_client import EmbeddingClient


class QueryDomain(str, Enum):
    """Known query domains."""

    GENERAL = "general"
    TECHNICAL = "technical"
    BUSINESS = "business"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    ANALYTICS = "analytics"


class ExecutionStrategy(str, Enum):
    """Strategies for executing federated queries."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY = "priority"


@dataclass
class SubQuery:
    """A sub-query targeting a specific domain."""

    domain: QueryDomain
    query_text: str
    priority: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainMatch:
    """Match result for domain detection."""

    domain: QueryDomain
    confidence: float
    keywords: list[str]


@dataclass
class QueryPlanStep:
    """A step in the federated query execution plan."""

    sub_query: SubQuery
    depends_on: list[int] = field(default_factory=list)
    estimated_results: int = 0


@dataclass
class FederatedQueryPlan:
    """Execution plan for a federated query."""

    original_query: str
    steps: list[QueryPlanStep]
    strategy: ExecutionStrategy
    aggregation_mode: str = "merge"
    timeout_seconds: int = 30


class FederatedQueryResult:
    """Result from a federated query execution."""

    def __init__(
        self,
        query: str,
        plan: FederatedQueryPlan,
    ) -> None:
        self.query = query
        self.plan = plan
        self.results: dict[QueryDomain, list[dict[str, Any]]] = {}
        self.errors: dict[QueryDomain, str] = {}
        self.execution_time_ms: float = 0.0
        self.total_results: int = 0

    def add_domain_results(
        self,
        domain: QueryDomain,
        results: list[dict[str, Any]],
    ) -> None:
        """Add results from a domain."""
        self.results[domain] = results
        self.total_results += len(results)

    def add_domain_error(
        self,
        domain: QueryDomain,
        error: str,
    ) -> None:
        """Add an error from a domain."""
        self.errors[domain] = error

    def get_combined_results(self) -> list[dict[str, Any]]:
        """Get all results combined with domain attribution."""
        combined: list[dict[str, Any]] = []
        for domain, results in self.results.items():
            for result in results:
                combined.append({**result, "_domain": domain.value})
        return combined


class DomainDetector:
    """Detects relevant domains from a query."""

    DOMAIN_DESCRIPTIONS: dict[QueryDomain, str] = {
        QueryDomain.TECHNICAL: "APIs, software engineering, infrastructure, debugging, and system design.",
        QueryDomain.BUSINESS: "Business metrics, revenue, strategy, customers, and market insights.",
        QueryDomain.DOCUMENTATION: "Documentation, guides, manuals, references, and how-to instructions.",
        QueryDomain.RESEARCH: "Research papers, studies, experiments, analysis, and academic findings.",
        QueryDomain.ANALYTICS: "Analytics, trends, dashboards, statistics, and reporting.",
    }

    KEYWORDS: dict[QueryDomain, list[str]] = {
        QueryDomain.TECHNICAL: [
            "api",
            "code",
            "function",
            "class",
            "method",
            "database",
            "server",
            "endpoint",
            "authentication",
            "algorithm",
            "bug",
            "error",
            "debug",
            "test",
            "deployment",
            "infrastructure",
        ],
        QueryDomain.BUSINESS: [
            "revenue",
            "customer",
            "sales",
            "profit",
            "strategy",
            "market",
            "competitor",
            "quarter",
            "budget",
            "roi",
            "kpi",
            "metric",
            "forecast",
            "growth",
            "stakeholder",
        ],
        QueryDomain.DOCUMENTATION: [
            "documentation",
            "guide",
            "tutorial",
            "reference",
            "manual",
            "specification",
            "readme",
            "how-to",
            "example",
            "help",
        ],
        QueryDomain.RESEARCH: [
            "study",
            "analysis",
            "hypothesis",
            "findings",
            "paper",
            "academic",
            "theory",
            "experiment",
            "methodology",
            "review",
            "literature",
            "citation",
            "data",
            "correlation",
        ],
        QueryDomain.ANALYTICS: [
            "trend",
            "pattern",
            "statistic",
            "distribution",
            "correlation",
            "average",
            "sum",
            "count",
            "percentage",
            "dashboard",
            "report",
            "visualization",
            "chart",
            "graph",
            "metric",
        ],
    }

    def __init__(
        self,
        default_domain: QueryDomain = QueryDomain.GENERAL,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        """Initialize domain detector.

        Args:
            default_domain: Default domain if no match is found.
        """
        self._default_domain = default_domain
        self._embedding_client = embedding_client
        self._domain_embedding_cache: dict[QueryDomain, list[float]] = {}

    def detect(
        self,
        query: str,
        max_domains: int = 3,
        min_confidence: float = 0.3,
    ) -> list[DomainMatch]:
        """Detect relevant domains from a query.

        Args:
            query: The natural language query.
            max_domains: Maximum number of domains to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of domain matches sorted by confidence.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        matches: list[DomainMatch] = []

        for domain, keywords in self.KEYWORDS.items():
            matched_keywords = [kw for kw in keywords if kw in query_lower]
            if matched_keywords:
                confidence = len(matched_keywords) / len(keywords)
                confidence = min(confidence * 1.5, 1.0)
                if confidence >= min_confidence:
                    matches.append(
                        DomainMatch(
                            domain=domain,
                            confidence=confidence,
                            keywords=matched_keywords,
                        )
                    )

        if not matches:
            matches.append(
                DomainMatch(
                    domain=self._default_domain,
                    confidence=1.0,
                    keywords=[],
                )
            )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:max_domains]

    async def detect_async(
        self,
        query: str,
        max_domains: int = 3,
        min_confidence: float = 0.3,
        embedding_weight: float = 0.6,
    ) -> list[DomainMatch]:
        """Detect relevant domains using keywords + embeddings."""
        keyword_matches = self.detect(
            query=query,
            max_domains=max_domains,
            min_confidence=0.0,
        )
        keyword_scores = {match.domain: match for match in keyword_matches}

        if self._embedding_client is None:
            return [
                match
                for match in keyword_matches
                if match.confidence >= min_confidence
            ][:max_domains]

        query_embedding = await self._embedding_client.embed_text(query)
        domain_embeddings = await self._get_domain_embeddings()

        combined_matches: list[DomainMatch] = []
        for domain, domain_embedding in domain_embeddings.items():
            similarity = self._cosine_similarity(query_embedding, domain_embedding)
            keyword_match = keyword_scores.get(domain)
            keyword_confidence = keyword_match.confidence if keyword_match else 0.0
            combined_confidence = (
                embedding_weight * similarity
                + (1 - embedding_weight) * keyword_confidence
            )
            if combined_confidence >= min_confidence:
                combined_matches.append(
                    DomainMatch(
                        domain=domain,
                        confidence=combined_confidence,
                        keywords=keyword_match.keywords if keyword_match else [],
                    )
                )

        if not combined_matches:
            return [DomainMatch(self._default_domain, 1.0, [])]

        combined_matches.sort(key=lambda m: m.confidence, reverse=True)
        return combined_matches[:max_domains]

    async def _get_domain_embeddings(self) -> dict[QueryDomain, list[float]]:
        """Get cached embeddings for domain descriptions."""
        if self._domain_embedding_cache:
            return self._domain_embedding_cache

        if self._embedding_client is None:
            return {}

        for domain, description in self.DOMAIN_DESCRIPTIONS.items():
            self._domain_embedding_cache[domain] = await self._embedding_client.embed_text(
                description
            )

        return self._domain_embedding_cache

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class SubQueryBuilder:
    """Builds sub-queries for each detected domain."""

    def __init__(self) -> None:
        """Initialize sub-query builder."""
        self._templates: dict[QueryDomain, list[str]] = {
            QueryDomain.GENERAL: [
                "Find information about {query}",
            ],
            QueryDomain.TECHNICAL: [
                "Search technical documentation for {query}",
                "Find code examples related to {query}",
                "Look up API references for {query}",
            ],
            QueryDomain.BUSINESS: [
                "Find business intelligence on {query}",
                "Search for strategic information about {query}",
                "Look up business metrics related to {query}",
            ],
            QueryDomain.DOCUMENTATION: [
                "Find documentation about {query}",
                "Search user guides for {query}",
                "Look up reference materials for {query}",
            ],
            QueryDomain.RESEARCH: [
                "Find research findings on {query}",
                "Search academic sources for {query}",
                "Look up analysis of {query}",
            ],
            QueryDomain.ANALYTICS: [
                "Find analytical data on {query}",
                "Search for metrics and statistics on {query}",
                "Look up trend analysis for {query}",
            ],
        }

    def build(
        self,
        original_query: str,
        domain: QueryDomain,
        context: Optional[dict[str, Any]] = None,
    ) -> list[SubQuery]:
        """Build sub-queries for a domain.

        Args:
            original_query: The original natural language query.
            domain: The target domain.
            context: Optional context for query refinement.

        Returns:
            List of sub-queries for the domain.
        """
        templates = self._templates.get(domain, self._templates[QueryDomain.GENERAL])
        sub_queries: list[SubQuery] = []

        for priority, template in enumerate(templates):
            query_text = template.format(query=original_query)
            sub_query = SubQuery(
                domain=domain,
                query_text=query_text,
                priority=priority,
                parameters=context or {},
            )
            sub_queries.append(sub_query)

        return sub_queries

    def add_template(
        self,
        domain: QueryDomain,
        template: str,
        priority: Optional[int] = None,
    ) -> None:
        """Add a custom template for a domain.

        Args:
            domain: The target domain.
            template: The query template with {query} placeholder.
            priority: Optional priority (higher = executed first).
        """
        if domain not in self._templates:
            self._templates[domain] = []

        if priority is not None:
            templates = [(p, t) for p, t in enumerate(self._templates[domain])]
            templates.insert(priority, (priority, template))
            self._templates[domain] = [t for _, t in sorted(templates)]
        else:
            self._templates[domain].append(template)


class ResultAggregator:
    """Aggregates results from multiple domain queries."""

    def __init__(
        self,
        mode: str = "merge",
        deduplicate: bool = True,
        ranking_strategy: str = "confidence",
    ) -> None:
        """Initialize result aggregator.

        Args:
            mode: Aggregation mode (merge, intersect, union).
            deduplicate: Whether to remove duplicate results.
            ranking_strategy: Strategy for ranking combined results.
        """
        self._mode = mode
        self._deduplicate = deduplicate
        self._ranking_strategy = ranking_strategy

    def aggregate(
        self,
        results: dict[QueryDomain, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Aggregate results from multiple domains.

        Args:
            results: Dictionary of domain to results list.

        Returns:
            Aggregated list of results.
        """
        if self._mode == "merge":
            return self._merge(results)
        elif self._mode == "intersect":
            return self._intersect(results)
        elif self._mode == "union":
            return self._union(results)
        return self._merge(results)

    def _merge(
        self,
        results: dict[QueryDomain, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merge results from all domains."""
        all_results: list[dict[str, Any]] = []
        for domain, domain_results in results.items():
            for result in domain_results:
                result["_domain"] = domain.value
                all_results.append(result)

        if self._deduplicate:
            all_results = self._deduplicate_results(all_results)

        if self._ranking_strategy == "confidence":
            all_results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        elif self._ranking_strategy == "domain_priority":
            domain_order = list(QueryDomain)
            all_results.sort(
                key=lambda r: domain_order.index(
                    QueryDomain(r.get("_domain", "general"))
                )
            )

        return all_results

    def _intersect(
        self,
        results: dict[QueryDomain, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Find common results across domains."""
        if not results:
            return []

        domain_lists = list(results.values())
        if not domain_lists:
            return []

        smallest_list = min(domain_lists, key=len)
        other_lists = [
            set(r.get("id", r.get("name", str(r))) for r in lst) for lst in domain_lists
        ]

        common_ids = set(r.get("id", r.get("name", str(r))) for r in smallest_list)
        for other_set in other_lists:
            common_ids &= other_set

        common_results = [
            r for r in smallest_list if r.get("id", r.get("name", str(r))) in common_ids
        ]
        return (
            self._deduplicate_results(common_results)
            if self._deduplicate
            else common_results
        )

    def _union(
        self,
        results: dict[QueryDomain, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Union results from all domains."""
        return self._merge(results)

    def _deduplicate_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove duplicate results based on id or name."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []

        for result in results:
            key = result.get("id", result.get("name", str(result)))
            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique


class FederatedQueryRouter:
    """Router for federated queries across multiple knowledge domains."""

    def __init__(
        self,
        retriever: Optional[HybridEntityRetriever] = None,
        default_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
        timeout_seconds: int = 30,
    ) -> None:
        """Initialize federated query router.

        Args:
            retriever: Optional hybrid entity retriever for execution.
            default_strategy: Default execution strategy.
            timeout_seconds: Default timeout for queries.
        """
        self._retriever = retriever
        self._default_strategy = default_strategy
        self._timeout_seconds = timeout_seconds
        self._domain_detector = DomainDetector(embedding_client=EmbeddingClient())
        self._sub_query_builder = SubQueryBuilder()
        self._aggregator = ResultAggregator()

    def create_plan(
        self,
        query: str,
        detected_domains: Optional[list[QueryDomain]] = None,
        strategy: Optional[ExecutionStrategy] = None,
        aggregation_mode: str = "merge",
    ) -> FederatedQueryPlan:
        """Create an execution plan for a federated query.

        Args:
            query: The original natural language query.
            detected_domains: Optional pre-detected domains.
            strategy: Optional execution strategy override.
            aggregation_mode: Mode for result aggregation.

        Returns:
            FederatedQueryPlan with execution steps.
        """
        if detected_domains is None:
            domain_matches = self._domain_detector.detect(query)
            detected_domains = [m.domain for m in domain_matches]

        steps: list[QueryPlanStep] = []
        for domain in detected_domains:
            sub_queries = self._sub_query_builder.build(query, domain)
            for sub_query in sub_queries:
                steps.append(QueryPlanStep(sub_query=sub_query))

        return FederatedQueryPlan(
            original_query=query,
            steps=steps,
            strategy=strategy or self._default_strategy,
            aggregation_mode=aggregation_mode,
            timeout_seconds=self._timeout_seconds,
        )

    async def execute_plan(
        self,
        plan: FederatedQueryPlan,
    ) -> FederatedQueryResult:
        """Execute a federated query plan.

        Args:
            plan: The execution plan to run.

        Returns:
            FederatedQueryResult with aggregated results.
        """
        import time

        start_time = time.time()

        result = FederatedQueryResult(query=plan.original_query, plan=plan)

        if plan.strategy == ExecutionStrategy.PARALLEL:
            await self._execute_parallel(plan, result)
        else:
            await self._execute_sequential(plan, result)

        result.execution_time_ms = (time.time() - start_time) * 1000
        return result

    async def _execute_sequential(
        self,
        plan: FederatedQueryPlan,
        result: FederatedQueryResult,
    ) -> None:
        """Execute plan steps sequentially."""
        for step in plan.steps:
            try:
                domain_results = await self._execute_sub_query(step.sub_query)
                result.add_domain_results(step.sub_query.domain, domain_results)
            except Exception as e:
                result.add_domain_error(step.sub_query.domain, str(e))

    async def _execute_parallel(
        self,
        plan: FederatedQueryPlan,
        result: FederatedQueryResult,
    ) -> None:
        """Execute plan steps in parallel."""
        import asyncio

        async def execute_step(
            step: QueryPlanStep,
        ) -> tuple[QueryDomain, list[dict[str, Any]], Optional[str]]:
            try:
                results = await self._execute_sub_query(step.sub_query)
                return step.sub_query.domain, results, None
            except Exception as e:
                return step.sub_query.domain, [], str(e)

        tasks = [execute_step(step) for step in plan.steps]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                continue
            domain, results, error = outcome
            if error:
                result.add_domain_error(domain, error)
            else:
                result.add_domain_results(domain, results)

    async def _execute_sub_query(
        self,
        sub_query: SubQuery,
    ) -> list[dict[str, Any]]:
        """Execute a single sub-query.

        Args:
            sub_query: The sub-query to execute.

        Returns:
            List of result dictionaries.
        """
        if self._retriever is None:
            return []

        try:
            retrieval_result: HybridRetrievalResult = await self._retriever.retrieve(
                query=sub_query.query_text,
                query_embedding=[],
                domain=sub_query.domain.value,
            )

            return [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "type": e.entity_type,
                    "description": e.description,
                    "confidence": e.final_score,
                    "source": e.source,
                }
                for e in retrieval_result.entities
            ]
        except Exception:
            return []

    async def route_and_execute(
        self,
        query: str,
        strategy: Optional[ExecutionStrategy] = None,
        aggregation_mode: Optional[str] = None,
    ) -> FederatedQueryResult:
        """Detect domains, create plan, and execute federated query.

        Args:
            query: The natural language query.
            strategy: Optional execution strategy override.
            aggregation_mode: Optional aggregation mode override.

        Returns:
            FederatedQueryResult with aggregated results.
        """
        domain_matches = await self._domain_detector.detect_async(query)
        domains = [m.domain for m in domain_matches]

        plan = self.create_plan(
            query=query,
            detected_domains=domains,
            strategy=strategy,
            aggregation_mode=aggregation_mode or "merge",
        )

        result = await self.execute_plan(plan)
        result.results = {domain: result.results.get(domain, []) for domain in domains}

        return result

    def aggregate_results(
        self,
        result: FederatedQueryResult,
        mode: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Aggregate results from a federated query.

        Args:
            result: The federated query result.
            mode: Optional aggregation mode override.

        Returns:
            Aggregated list of results.
        """
        aggregator = ResultAggregator(
            mode=mode or result.plan.aggregation_mode,
        )
        return aggregator.aggregate(result.results)
