"""Ontology Validator for KBv2.

Validates extractions against crypto ontology rules to detect:
- Missing required properties
- Relationship cardinality violations
- Semantic contradictions
- Type consistency issues
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel, Field


class ViolationSeverity(str, Enum):
    """Severity levels for ontology violations."""

    ERROR = "error"  # Must fix - breaks ontology
    WARNING = "warning"  # Should fix - incomplete extraction
    INFO = "info"  # Minor issue - optional improvement


class ViolationType(str, Enum):
    """Types of ontology violations."""

    MISSING_REQUIRED_PROPERTY = "missing_required_property"
    INVALID_PROPERTY_VALUE = "invalid_property_value"
    CARDINALITY_VIOLATION = "cardinality_violation"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    TYPE_MISMATCH = "type_mismatch"
    RELATIONSHIP_INCONSISTENCY = "relationship_inconsistency"
    DUPLICATE_ENTITY = "duplicate_entity"
    INVALID_DATE_RANGE = "invalid_date_range"


class OntologyViolation(BaseModel):
    """A single ontology violation."""

    violation_id: UUID = Field(default_factory=UUID)
    violation_type: ViolationType
    severity: ViolationSeverity
    entity_id: Optional[UUID] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None
    property_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    message: str
    suggestion: Optional[str] = None
    auto_fixable: bool = False


class ValidationReport(BaseModel):
    """Complete validation report for an extraction."""

    document_id: Optional[UUID] = None
    chunk_id: Optional[UUID] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    violations: List[OntologyViolation] = Field(default_factory=list)

    # Statistics
    total_entities: int = 0
    total_relationships: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    auto_fixable_count: int = 0

    # Quality metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)

    def get_errors(self) -> List[OntologyViolation]:
        """Get all error-level violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.ERROR]

    def get_warnings(self) -> List[OntologyViolation]:
        """Get all warning-level violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.WARNING]

    def is_valid(self) -> bool:
        """Check if extraction passes validation (no errors)."""
        return self.error_count == 0


class OntologyRule(BaseModel):
    """A single ontology validation rule."""

    rule_id: str
    entity_type: str
    description: str
    severity: ViolationSeverity

    # Rule conditions
    required_properties: List[str] = Field(default_factory=list)
    optional_properties: List[str] = Field(default_factory=list)
    property_types: Dict[str, str] = Field(default_factory=dict)
    property_patterns: Dict[str, str] = Field(default_factory=dict)  # Regex patterns

    # Cardinality constraints
    min_properties: Optional[int] = None
    max_properties: Optional[int] = None

    # Relationship constraints
    required_relationships: List[str] = Field(default_factory=list)
    valid_relationship_targets: Dict[str, List[str]] = Field(default_factory=dict)

    # Value constraints
    min_values: Dict[str, float] = Field(default_factory=dict)
    max_values: Dict[str, float] = Field(default_factory=dict)
    allowed_values: Dict[str, List[str]] = Field(default_factory=dict)


class CryptoOntologyRules:
    """Ontology rules for cryptocurrency domain."""

    @staticmethod
    def get_all_rules() -> List[OntologyRule]:
        """Get all crypto ontology rules."""
        return [
            # Bitcoin Domain Rules
            CryptoOntologyRules._bitcoin_etf_rule(),
            CryptoOntologyRules._mining_pool_rule(),
            CryptoOntologyRules._digital_asset_treasury_rule(),
            # DeFi Domain Rules
            CryptoOntologyRules._defi_protocol_rule(),
            CryptoOntologyRules._liquidity_pool_rule(),
            CryptoOntologyRules._governance_token_rule(),
            # Institutional Domain Rules
            CryptoOntologyRules._etf_issuer_rule(),
            CryptoOntologyRules._crypto_custodian_rule(),
            # Stablecoin Domain Rules
            CryptoOntologyRules._stablecoin_rule(),
            CryptoOntologyRules._stablecoin_issuer_rule(),
            # Regulatory Domain Rules
            CryptoOntologyRules._regulatory_body_rule(),
            CryptoOntologyRules._regulation_rule(),
            # General Rules
            CryptoOntologyRules._cryptocurrency_rule(),
        ]

    @staticmethod
    def _bitcoin_etf_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="bitcoin_etf_001",
            entity_type="BitcoinETF",
            description="Bitcoin ETF must have issuer, ticker, and type",
            severity=ViolationSeverity.ERROR,
            required_properties=["issuer", "ticker", "etf_type"],
            optional_properties=[
                "aum",
                "expense_ratio",
                "bitcoin_holdings",
                "launch_date",
            ],
            property_types={
                "aum": "number",
                "expense_ratio": "number",
                "bitcoin_holdings": "number",
                "launch_date": "date",
            },
            allowed_values={
                "etf_type": ["spot", "futures"],
            },
            min_values={
                "expense_ratio": 0.0,
                "aum": 0.0,
            },
        )

    @staticmethod
    def _mining_pool_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="mining_pool_001",
            entity_type="MiningPool",
            description="Mining pool should have hash rate information",
            severity=ViolationSeverity.WARNING,
            required_properties=["name"],
            optional_properties=[
                "hash_rate",
                "hash_rate_percentage",
                "blocks_mined_24h",
                "location",
            ],
            property_types={
                "hash_rate": "number",
                "hash_rate_percentage": "number",
                "blocks_mined_24h": "number",
            },
            min_values={
                "hash_rate": 0.0,
                "hash_rate_percentage": 0.0,
                "blocks_mined_24h": 0,
            },
            max_values={
                "hash_rate_percentage": 100.0,
            },
        )

    @staticmethod
    def _digital_asset_treasury_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="dat_001",
            entity_type="DigitalAssetTreasury",
            description="Digital Asset Treasury must have company and BTC holdings",
            severity=ViolationSeverity.ERROR,
            required_properties=["company_name", "bitcoin_holdings"],
            optional_properties=[
                "cost_basis",
                "average_entry",
                "total_investment",
                "last_updated",
            ],
            property_types={
                "bitcoin_holdings": "number",
                "cost_basis": "number",
                "average_entry": "number",
                "total_investment": "number",
            },
            min_values={
                "bitcoin_holdings": 0.0,
                "cost_basis": 0.0,
            },
        )

    @staticmethod
    def _defi_protocol_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="defi_protocol_001",
            entity_type="DeFiProtocol",
            description="DeFi protocol should have TVL or market metrics",
            severity=ViolationSeverity.WARNING,
            required_properties=["name", "protocol_type"],
            optional_properties=["tvl", "market_cap", "fdv", "revenue_24h", "chain"],
            property_types={
                "tvl": "number",
                "market_cap": "number",
                "fdv": "number",
                "revenue_24h": "number",
            },
            allowed_values={
                "protocol_type": [
                    "lending",
                    "dex",
                    "yield",
                    "derivative",
                    "bridge",
                    "oracle",
                ],
            },
            min_values={
                "tvl": 0.0,
                "market_cap": 0.0,
            },
        )

    @staticmethod
    def _liquidity_pool_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="liquidity_pool_001",
            entity_type="LiquidityPool",
            description="Liquidity pool must have protocol and token pairs",
            severity=ViolationSeverity.ERROR,
            required_properties=["protocol", "token_pairs"],
            optional_properties=["tvl", "apy", "volume_24h", "fee_tier"],
            property_types={
                "tvl": "number",
                "apy": "number",
                "volume_24h": "number",
                "fee_tier": "number",
            },
            min_values={
                "tvl": 0.0,
                "apy": 0.0,
            },
        )

    @staticmethod
    def _governance_token_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="gov_token_001",
            entity_type="GovernanceToken",
            description="Governance token should have protocol association",
            severity=ViolationSeverity.WARNING,
            required_properties=["symbol", "protocol"],
            optional_properties=["market_cap", "voting_power", "utility"],
            property_types={
                "market_cap": "number",
            },
            min_values={
                "market_cap": 0.0,
            },
        )

    @staticmethod
    def _etf_issuer_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="etf_issuer_001",
            entity_type="ETFIssuer",
            description="ETF issuer must have company name",
            severity=ViolationSeverity.ERROR,
            required_properties=["company_name"],
            optional_properties=["total_aum", "etfs_managed", "custody_partner"],
            property_types={
                "total_aum": "number",
                "etfs_managed": "number",
            },
            min_values={
                "total_aum": 0.0,
                "etfs_managed": 0,
            },
        )

    @staticmethod
    def _crypto_custodian_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="custodian_001",
            entity_type="CryptoCustodian",
            description="Crypto custodian should have assets under custody",
            severity=ViolationSeverity.WARNING,
            required_properties=["company_name"],
            optional_properties=[
                "assets_under_custody",
                "insurance_coverage",
                "clients",
                "audit_status",
            ],
            property_types={
                "assets_under_custody": "number",
                "insurance_coverage": "number",
            },
            min_values={
                "assets_under_custody": 0.0,
                "insurance_coverage": 0.0,
            },
        )

    @staticmethod
    def _stablecoin_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="stablecoin_001",
            entity_type="Stablecoin",
            description="Stablecoin must have issuer and backing type",
            severity=ViolationSeverity.ERROR,
            required_properties=["symbol", "issuer", "backing_type"],
            optional_properties=[
                "market_cap",
                "circulating_supply",
                "collateral_ratio",
                "genius_act_compliant",
            ],
            property_types={
                "market_cap": "number",
                "circulating_supply": "number",
                "collateral_ratio": "number",
            },
            allowed_values={
                "backing_type": [
                    "fiat",
                    "crypto",
                    "commodity",
                    "algorithmic",
                    "hybrid",
                ],
            },
            min_values={
                "market_cap": 0.0,
                "circulating_supply": 0.0,
                "collateral_ratio": 0.0,
            },
        )

    @staticmethod
    def _stablecoin_issuer_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="stablecoin_issuer_001",
            entity_type="StablecoinIssuer",
            description="Stablecoin issuer should have reserve information",
            severity=ViolationSeverity.WARNING,
            required_properties=["company_name"],
            optional_properties=[
                "total_reserves",
                "attestation_url",
                "auditor",
                "monthly_disclosure",
            ],
            property_types={
                "total_reserves": "number",
            },
            min_values={
                "total_reserves": 0.0,
            },
        )

    @staticmethod
    def _regulatory_body_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="reg_body_001",
            entity_type="RegulatoryBody",
            description="Regulatory body must have name and jurisdiction",
            severity=ViolationSeverity.ERROR,
            required_properties=["name", "jurisdiction"],
            optional_properties=["authority_level", "website", "head_official"],
        )

    @staticmethod
    def _regulation_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="regulation_001",
            entity_type="Regulation",
            description="Regulation should have status and effective date",
            severity=ViolationSeverity.WARNING,
            required_properties=["name", "jurisdiction"],
            optional_properties=[
                "status",
                "effective_date",
                "compliance_deadline",
                "regulatory_body",
            ],
            property_types={
                "effective_date": "date",
                "compliance_deadline": "date",
            },
            allowed_values={
                "status": ["proposed", "approved", "effective", "amended", "repealed"],
            },
        )

    @staticmethod
    def _cryptocurrency_rule() -> OntologyRule:
        return OntologyRule(
            rule_id="crypto_001",
            entity_type="Cryptocurrency",
            description="Cryptocurrency must have symbol and name",
            severity=ViolationSeverity.ERROR,
            required_properties=["symbol", "name"],
            optional_properties=[
                "market_cap",
                "circulating_supply",
                "max_supply",
                "consensus_mechanism",
            ],
            property_types={
                "market_cap": "number",
                "circulating_supply": "number",
                "max_supply": "number",
            },
            allowed_values={
                "consensus_mechanism": [
                    "proof_of_work",
                    "proof_of_stake",
                    "delegated_pos",
                    "other",
                ],
            },
            min_values={
                "market_cap": 0.0,
                "circulating_supply": 0.0,
            },
        )


class SemanticContradictionChecker:
    """Checks for semantic contradictions in extractions."""

    # Known contradictory statements
    CONTRADICTION_PAIRS = [
        # Bitcoin contradictions
        (["deflationary", "fixed supply"], ["inflationary", "infinite supply"]),
        (["proof of work"], ["proof of stake"]),
        (["spot etf"], ["futures etf"]),
        # Stablecoin contradictions
        (["fiat backed", "fiat collateralized"], ["algorithmic", "uncollateralized"]),
        (["1:1 reserve"], ["fractional reserve"]),
        # DeFi contradictions
        (["overcollateralized"], ["undercollateralized", "uncollateralized"]),
        (["permissionless"], ["permissioned", "kyc required"]),
        # Regulation contradictions
        (["security"], ["commodity", "currency"]),
        (["approved"], ["rejected", "denied"]),
    ]

    @classmethod
    def check_contradictions(
        cls, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> List[OntologyViolation]:
        """Check for semantic contradictions."""
        violations = []

        # Extract all descriptions and properties
        all_text = []
        for entity in entities:
            all_text.append(entity.get("description", "").lower())
            all_text.append(entity.get("name", "").lower())
            props = entity.get("properties", {})
            if isinstance(props, dict):
                for value in props.values():
                    if isinstance(value, str):
                        all_text.append(value.lower())

        combined_text = " ".join(all_text)

        # Check for contradictions
        for positive_terms, negative_terms in cls.CONTRADICTION_PAIRS:
            has_positive = any(term in combined_text for term in positive_terms)
            has_negative = any(term in combined_text for term in negative_terms)

            if has_positive and has_negative:
                violation = OntologyViolation(
                    violation_type=ViolationType.SEMANTIC_CONTRADICTION,
                    severity=ViolationSeverity.ERROR,
                    message=f"Contradiction detected: '{positive_terms[0]}' vs '{negative_terms[0]}'",
                    suggestion="Verify entity properties and ensure consistency",
                )
                violations.append(violation)

        return violations


class OntologyValidator:
    """Main ontology validation engine."""

    def __init__(self, rules: Optional[List[OntologyRule]] = None):
        self.rules = rules or CryptoOntologyRules.get_all_rules()
        self.rules_by_type: Dict[str, OntologyRule] = {
            rule.entity_type: rule for rule in self.rules
        }
        self.contradiction_checker = SemanticContradictionChecker()

    async def validate_extraction(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        document_id: Optional[UUID] = None,
        chunk_id: Optional[UUID] = None,
    ) -> ValidationReport:
        """Validate extraction against ontology rules.

        Args:
            entities: Extracted entities
            relationships: Extracted relationships
            document_id: Source document ID
            chunk_id: Source chunk ID

        Returns:
            Validation report with violations and scores
        """
        violations = []

        # Validate each entity against rules
        for entity in entities:
            entity_violations = self._validate_entity(entity)
            violations.extend(entity_violations)

        # Validate relationships
        relationship_violations = self._validate_relationships(entities, relationships)
        violations.extend(relationship_violations)

        # Check for semantic contradictions
        contradiction_violations = self.contradiction_checker.check_contradictions(
            entities, relationships
        )
        violations.extend(contradiction_violations)

        # Calculate scores
        report = self._build_report(
            violations=violations,
            entities=entities,
            relationships=relationships,
            document_id=document_id,
            chunk_id=chunk_id,
        )

        return report

    def _validate_entity(self, entity: Dict[str, Any]) -> List[OntologyViolation]:
        """Validate a single entity against its type rule."""
        violations = []

        entity_type = entity.get("entity_type", "Unknown")
        entity_name = entity.get("name", "Unknown")
        entity_id = entity.get("id")
        properties = entity.get("properties", {})

        # Get rule for entity type
        rule = self.rules_by_type.get(entity_type)
        if not rule:
            return violations

        # Check required properties
        for required_prop in rule.required_properties:
            if required_prop not in properties or properties[required_prop] is None:
                violations.append(
                    OntologyViolation(
                        violation_type=ViolationType.MISSING_REQUIRED_PROPERTY,
                        severity=rule.severity,
                        entity_id=entity_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        property_name=required_prop,
                        message=f"Missing required property '{required_prop}' for {entity_type}",
                        suggestion=f"Add '{required_prop}' to entity properties",
                        auto_fixable=False,
                    )
                )

        # Check property types and values
        for prop_name, prop_value in properties.items():
            if prop_name in rule.property_types:
                expected_type = rule.property_types[prop_name]

                # Type checking
                if expected_type == "number" and not isinstance(
                    prop_value, (int, float)
                ):
                    violations.append(
                        OntologyViolation(
                            violation_type=ViolationType.TYPE_MISMATCH,
                            severity=ViolationSeverity.WARNING,
                            entity_id=entity_id,
                            entity_name=entity_name,
                            entity_type=entity_type,
                            property_name=prop_name,
                            expected_value=expected_type,
                            actual_value=type(prop_value).__name__,
                            message=f"Property '{prop_name}' should be a number",
                            suggestion="Convert value to numeric type",
                            auto_fixable=True,
                        )
                    )

                # Min/max value checking
                if expected_type == "number" and isinstance(prop_value, (int, float)):
                    if (
                        prop_name in rule.min_values
                        and prop_value < rule.min_values[prop_name]
                    ):
                        violations.append(
                            OntologyViolation(
                                violation_type=ViolationType.INVALID_PROPERTY_VALUE,
                                severity=ViolationSeverity.ERROR,
                                entity_id=entity_id,
                                entity_name=entity_name,
                                entity_type=entity_type,
                                property_name=prop_name,
                                expected_value=f">= {rule.min_values[prop_name]}",
                                actual_value=prop_value,
                                message=f"Value {prop_value} below minimum {rule.min_values[prop_name]}",
                            )
                        )

                    if (
                        prop_name in rule.max_values
                        and prop_value > rule.max_values[prop_name]
                    ):
                        violations.append(
                            OntologyViolation(
                                violation_type=ViolationType.INVALID_PROPERTY_VALUE,
                                severity=ViolationSeverity.ERROR,
                                entity_id=entity_id,
                                entity_name=entity_name,
                                entity_type=entity_type,
                                property_name=prop_name,
                                expected_value=f"<= {rule.max_values[prop_name]}",
                                actual_value=prop_value,
                                message=f"Value {prop_value} above maximum {rule.max_values[prop_name]}",
                            )
                        )

                # Allowed values checking
                if prop_name in rule.allowed_values:
                    if isinstance(prop_value, str) and prop_value.lower() not in [
                        v.lower() for v in rule.allowed_values[prop_name]
                    ]:
                        violations.append(
                            OntologyViolation(
                                violation_type=ViolationType.INVALID_PROPERTY_VALUE,
                                severity=ViolationSeverity.WARNING,
                                entity_id=entity_id,
                                entity_name=entity_name,
                                entity_type=entity_type,
                                property_name=prop_name,
                                expected_value=rule.allowed_values[prop_name],
                                actual_value=prop_value,
                                message=f"Value '{prop_value}' not in allowed values",
                                suggestion=f"Use one of: {', '.join(rule.allowed_values[prop_name])}",
                            )
                        )

        # Check property count
        if rule.min_properties and len(properties) < rule.min_properties:
            violations.append(
                OntologyViolation(
                    violation_type=ViolationType.CARDINALITY_VIOLATION,
                    severity=ViolationSeverity.WARNING,
                    entity_id=entity_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    expected_value=f">= {rule.min_properties} properties",
                    actual_value=f"{len(properties)} properties",
                    message=f"Entity has too few properties ({len(properties)} < {rule.min_properties})",
                )
            )

        return violations

    def _validate_relationships(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> List[OntologyViolation]:
        """Validate relationships between entities."""
        violations = []

        # Build entity lookup
        entity_ids = {str(e.get("id", "")): e for e in entities}
        entity_names = {e.get("name", "").lower(): e for e in entities}

        for rel in relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            rel_type = rel.get("relationship_type", "RELATED_TO")

            # Check if source and target exist
            source_exists = source in entity_ids or source.lower() in entity_names
            target_exists = target in entity_ids or target.lower() in entity_names

            if not source_exists:
                violations.append(
                    OntologyViolation(
                        violation_type=ViolationType.RELATIONSHIP_INCONSISTENCY,
                        severity=ViolationSeverity.ERROR,
                        message=f"Relationship source entity '{source}' not found",
                        suggestion="Create entity or fix relationship reference",
                    )
                )

            if not target_exists:
                violations.append(
                    OntologyViolation(
                        violation_type=ViolationType.RELATIONSHIP_INCONSISTENCY,
                        severity=ViolationSeverity.ERROR,
                        message=f"Relationship target entity '{target}' not found",
                        suggestion="Create entity or fix relationship reference",
                    )
                )

        return violations

    def _build_report(
        self,
        violations: List[OntologyViolation],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        document_id: Optional[UUID],
        chunk_id: Optional[UUID],
    ) -> ValidationReport:
        """Build validation report from violations."""

        error_count = sum(
            1 for v in violations if v.severity == ViolationSeverity.ERROR
        )
        warning_count = sum(
            1 for v in violations if v.severity == ViolationSeverity.WARNING
        )
        info_count = sum(1 for v in violations if v.severity == ViolationSeverity.INFO)
        auto_fixable_count = sum(1 for v in violations if v.auto_fixable)

        # Calculate completeness score
        total_properties = sum(len(e.get("properties", {})) for e in entities)
        required_properties = sum(
            len(
                self.rules_by_type.get(
                    e.get("entity_type"),
                    OntologyRule(
                        rule_id="",
                        entity_type="",
                        description="",
                        severity=ViolationSeverity.INFO,
                    ),
                ).required_properties
            )
            for e in entities
        )
        completeness_score = 1.0 - (error_count / max(required_properties, 1))

        # Calculate consistency score
        consistency_score = 1.0 - (
            len(
                [
                    v
                    for v in violations
                    if v.violation_type == ViolationType.SEMANTIC_CONTRADICTION
                ]
            )
            / max(len(entities), 1)
        )

        # Overall score
        overall_score = (completeness_score + consistency_score) / 2

        return ValidationReport(
            document_id=document_id,
            chunk_id=chunk_id,
            violations=violations,
            total_entities=len(entities),
            total_relationships=len(relationships),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            auto_fixable_count=auto_fixable_count,
            completeness_score=max(0.0, min(1.0, completeness_score)),
            consistency_score=max(0.0, min(1.0, consistency_score)),
            overall_score=max(0.0, min(1.0, overall_score)),
        )

    def add_rule(self, rule: OntologyRule) -> None:
        """Add a custom ontology rule."""
        self.rules.append(rule)
        self.rules_by_type[rule.entity_type] = rule

    def get_rules_for_type(self, entity_type: str) -> Optional[OntologyRule]:
        """Get rules for a specific entity type."""
        return self.rules_by_type.get(entity_type)
