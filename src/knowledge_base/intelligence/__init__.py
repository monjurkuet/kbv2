"""Intelligence package."""

from knowledge_base.intelligence.v1.clustering_service import ClusteringService
from knowledge_base.intelligence.v1.entity_typing_service import (
    EntityTyper,
    EntityType,
    DomainType,
    EntityTypingResult,
)
from knowledge_base.intelligence.v1.resolution_agent import ResolutionAgent
from knowledge_base.intelligence.v1.synthesis_agent import SynthesisAgent
from knowledge_base.intelligence.v1.multi_agent_extractor import (
    EntityExtractionManager,
    ExtractionQualityScore,
    EntityExtractionQuality,
)
from knowledge_base.intelligence.v1.hallucination_detector import (
    HallucinationDetector,
    HallucinationDetectorConfig,
    EntityVerification,
    AttributeVerification,
    RiskLevel,
    VerificationStatus,
)
from knowledge_base.intelligence.v1.hybrid_retriever import (
    HybridEntityRetriever,
    HybridRetrievalResult,
    RetrievedEntity,
)
from knowledge_base.intelligence.v1.domain_schema_service import (
    SchemaRegistry,
    DomainSchema,
    EntityTypeDef,
    DomainAttribute,
    InheritanceType,
    DomainLevel,
)
from knowledge_base.intelligence.v1.cross_domain_detector import (
    CrossDomainDetector,
    CrossDomainRelationship,
    RelationshipType,
    DomainType as CrossDomainDomainType,
)
from knowledge_base.intelligence.v1.federated_query_router import (
    FederatedQueryRouter,
    FederatedQueryPlan,
    FederatedQueryResult,
    QueryDomain,
    ExecutionStrategy,
)

__all__ = [
    # Existing exports
    "ClusteringService",
    "EntityTyper",
    "EntityType",
    "DomainType",
    "EntityTypingResult",
    "ResolutionAgent",
    "SynthesisAgent",
    # New exports
    "EntityExtractionManager",
    "ExtractionQualityScore",
    "EntityExtractionQuality",
    "HallucinationDetector",
    "HallucinationDetectorConfig",
    "EntityVerification",
    "AttributeVerification",
    "RiskLevel",
    "VerificationStatus",
    "HybridEntityRetriever",
    "HybridRetrievalResult",
    "RetrievedEntity",
    "SchemaRegistry",
    "DomainSchema",
    "EntityTypeDef",
    "DomainAttribute",
    "InheritanceType",
    "DomainLevel",
    "CrossDomainDetector",
    "CrossDomainRelationship",
    "RelationshipType",
    "DomainType as CrossDomainDomainType",
    "FederatedQueryRouter",
    "FederatedQueryPlan",
    "FederatedQueryResult",
    "QueryDomain",
    "ExecutionStrategy",
]
