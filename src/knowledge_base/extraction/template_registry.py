"""Domain-specific extraction goal templates for guided extraction."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ExtractionGoal(BaseModel):
    """Single extraction goal for a specific domain.

    Attributes:
        name: Unique identifier for this goal.
        description: Human-readable description of what to extract.
        target_entities: List of entity types to extract.
        target_relationships: List of relationship types to extract.
        priority: Priority level (1-5, lower is higher priority).
        examples: Example entities or values for few-shot prompting.
    """

    name: str = Field(..., description="Unique goal identifier")
    description: str = Field(..., description="Goal description")
    target_entities: List[str] = Field(..., description="Entity types to extract")
    target_relationships: List[str] = Field(
        ..., description="Relationship types to extract"
    )
    priority: int = Field(
        default=1, ge=1, le=5, description="Priority (1=highest, 5=lowest)"
    )
    examples: List[str] = Field(
        default_factory=list, description="Example entities for guidance"
    )


class TemplateRegistry(BaseModel):
    """Registry of domain-specific extraction goals."""

    default_goals: Dict[str, List[ExtractionGoal]] = Field(default_factory=dict)

    def __init__(self, **data) -> None:
        """Initialize registry with default goals for each domain."""
        if not data.get("default_goals"):
            data["default_goals"] = DEFAULT_GOALS
        super().__init__(**data)

    def get_goals(self, domain: str) -> List[ExtractionGoal]:
        """Get extraction goals for a specific domain.

        Args:
            domain: Domain name (e.g., 'TECHNOLOGY', 'FINANCIAL').

        Returns:
            List of extraction goals for the domain.
        """
        return self.default_goals.get(domain.upper(), DEFAULT_GOALS["GENERAL"])

    def get_goal_by_name(self, domain: str, goal_name: str) -> Optional[ExtractionGoal]:
        """Get a specific goal by name within a domain.

        Args:
            domain: Domain name.
            goal_name: Goal identifier.

        Returns:
            ExtractionGoal if found, None otherwise.
        """
        for goal in self.get_goals(domain):
            if goal.name == goal_name:
                return goal
        return None

    def get_prioritized_goals(self, domain: str) -> List[ExtractionGoal]:
        """Get goals sorted by priority for a domain.

        Args:
            domain: Domain name.

        Returns:
            Goals sorted from highest to lowest priority.
        """
        return sorted(self.get_goals(domain), key=lambda g: g.priority)


DEFAULT_GOALS: Dict[str, List[ExtractionGoal]] = {
    "TECHNOLOGY": [
        ExtractionGoal(
            name="software_systems",
            description="Extract software systems, libraries, frameworks, and tools",
            target_entities=[
                "Software",
                "Library",
                "Framework",
                "API",
                "Service",
                "Platform",
                "Database",
                "Tool",
            ],
            target_relationships=[
                "uses",
                "depends_on",
                "implements",
                "provides",
                "integrates_with",
                "hosted_on",
            ],
            priority=1,
            examples=["React", "PostgreSQL", "Kubernetes", "TensorFlow"],
        ),
        ExtractionGoal(
            name="architecture",
            description="Extract system architectures, designs, and patterns",
            target_entities=[
                "Architecture",
                "Pattern",
                "Component",
                "Module",
                "Service",
                "Microservice",
            ],
            target_relationships=[
                "composed_of",
                "connects_to",
                "deploys",
                "scales_with",
                "communicates_with",
            ],
            priority=2,
            examples=["microservices", "event-driven", "CQRS", "serverless"],
        ),
        ExtractionGoal(
            name="code_artifacts",
            description="Extract code repositories, files, and version information",
            target_entities=[
                "Repository",
                "File",
                "Function",
                "Class",
                "Version",
                "Commit",
            ],
            target_relationships=[
                "contains",
                "imports",
                "inherits_from",
                "defined_in",
                "committed_at",
            ],
            priority=3,
            examples=["main.py", "User class", "v2.1.0", "commit abc123"],
        ),
        ExtractionGoal(
            name="infrastructure",
            description="Extract infrastructure components and configurations",
            target_entities=[
                "Server",
                "Cluster",
                "Container",
                "Network",
                "LoadBalancer",
                "Firewall",
            ],
            target_relationships=[
                "deployed_on",
                "connected_to",
                "routes_to",
                "scales_with",
            ],
            priority=4,
            examples=["AWS EC2", "Kubernetes cluster", "Nginx", "VPC"],
        ),
    ],
    "FINANCIAL": [
        ExtractionGoal(
            name="companies",
            description="Extract company information and financials",
            target_entities=[
                "Company",
                "Organization",
                "Market",
                "Exchange",
                "Ticker",
                "Subsidiary",
            ],
            target_relationships=[
                "acquires",
                "invests_in",
                "competes_with",
                "trades_on",
                "owns",
                "partners_with",
            ],
            priority=1,
            examples=["Apple", "Microsoft", "Google", "NASDAQ", "AAPL"],
        ),
        ExtractionGoal(
            name="financial_metrics",
            description="Extract financial metrics and performance data",
            target_entities=[
                "Revenue",
                "Profit",
                "Investment",
                "MarketCap",
                "Valuation",
                "EBITDA",
                "Margin",
            ],
            target_relationships=[
                "reported_by",
                "grew_by",
                "decreased_by",
                "valued_at",
                "invested_by",
            ],
            priority=1,
            examples=["$100M revenue", "25% growth", "$50B valuation"],
        ),
        ExtractionGoal(
            name="transactions",
            description="Extract financial transactions and deals",
            target_entities=[
                "Transaction",
                "Deal",
                "Merger",
                "Acquisition",
                "Funding",
                "IPO",
            ],
            target_relationships=[
                "involves",
                "completed_on",
                "valued_at",
                "funded_by",
                "announced_on",
            ],
            priority=2,
            examples=["$5B acquisition", "Series B funding", "IPO"],
        ),
        ExtractionGoal(
            name="market_data",
            description="Extract market data and trends",
            target_entities=[
                "Market",
                "Sector",
                "Index",
                "Trend",
                "Indicator",
            ],
            target_relationships=[
                "tracked_by",
                "influenced_by",
                "belongs_to",
                "forecasts",
            ],
            priority=3,
            examples=["S&P 500", "tech sector", "bull market"],
        ),
    ],
    "MEDICAL": [
        ExtractionGoal(
            name="diseases",
            description="Extract diseases, conditions, and health conditions",
            target_entities=[
                "Disease",
                "Condition",
                "Syndrome",
                "Infection",
                "Disorder",
            ],
            target_relationships=[
                "causes",
                "treats",
                "diagnosed_with",
                "prevents",
                "symptom_of",
            ],
            priority=1,
            examples=["Diabetes", "Hypertension", "Cancer", "COVID-19"],
        ),
        ExtractionGoal(
            name="treatments",
            description="Extract treatments, drugs, and therapies",
            target_entities=[
                "Drug",
                "Treatment",
                "Therapy",
                "Procedure",
                "Medication",
                "Vaccine",
            ],
            target_relationships=[
                "treats",
                "prevents",
                "cures",
                "side_effects",
                "interacts_with",
                "administered_as",
            ],
            priority=1,
            examples=["Metformin", "Chemotherapy", "Surgery", "mRNA vaccine"],
        ),
        ExtractionGoal(
            name="anatomy",
            description="Extract anatomical structures and body systems",
            target_entities=[
                "Organ",
                "Tissue",
                "Cell",
                "System",
                "Structure",
            ],
            target_relationships=[
                "located_in",
                "connected_to",
                "part_of",
                "supplies",
                "drains",
            ],
            priority=2,
            examples=["heart", "liver", "neuron", "circulatory system"],
        ),
        ExtractionGoal(
            name="medical_procedures",
            description="Extract medical procedures and diagnostics",
            target_entities=[
                "Procedure",
                "Diagnostic",
                "Surgery",
                "Test",
                "Screening",
            ],
            target_relationships=[
                "performed_on",
                "diagnoses",
                "treats",
                "requires",
                "results_in",
            ],
            priority=3,
            examples=["MRI", "biopsy", "angioplasty", "blood test"],
        ),
        ExtractionGoal(
            name="clinical_trials",
            description="Extract clinical trial information",
            target_entities=[
                "ClinicalTrial",
                "Phase",
                "Cohort",
                "Eligibility",
                "Outcome",
            ],
            target_relationships=[
                "tests",
                "enrolls",
                "conducted_by",
                "results_in",
                "compares",
            ],
            priority=4,
            examples=["Phase III trial", "NCT01234567", "placebo group"],
        ),
    ],
    "LEGAL": [
        ExtractionGoal(
            name="entities",
            description="Extract legal entities (persons, organizations, courts)",
            target_entities=[
                "Person",
                "Organization",
                "Court",
                "Attorney",
                "Judge",
                "Party",
            ],
            target_relationships=[
                "represented_by",
                "presides_over",
                "filed_by",
                "ruled_on",
            ],
            priority=1,
            examples=["John Smith", "Supreme Court", "Judge Davis"],
        ),
        ExtractionGoal(
            name="cases",
            description="Extract legal case information",
            target_entities=[
                "Case",
                "Opinion",
                "Ruling",
                "Motion",
                "Judgment",
            ],
            target_relationships=[
                "cited_in",
                "overruled",
                "affirmed",
                "appealed",
                "filed",
            ],
            priority=1,
            examples=["Brown v. Board", "landmark case", "dismissed"],
        ),
        ExtractionGoal(
            name="statutes",
            description="Extract statutory references and regulations",
            target_entities=[
                "Statute",
                "Regulation",
                "Code",
                "Section",
                "Act",
            ],
            target_relationships=[
                "codified_in",
                "amends",
                "supersedes",
                "enforced_under",
            ],
            priority=2,
            examples=["42 U.S.C. § 1983", "GDPR", "HIPAA"],
        ),
        ExtractionGoal(
            name="contracts",
            description="Extract contract terms and agreements",
            target_entities=[
                "Contract",
                "Agreement",
                "Clause",
                "Term",
                "Obligation",
            ],
            target_relationships=[
                "parties_to",
                "governed_by",
                "terminates",
                "obligates",
            ],
            priority=3,
            examples=["NDA", "service agreement", "non-compete clause"],
        ),
    ],
    "SCIENTIFIC": [
        ExtractionGoal(
            name="research_papers",
            description="Extract research paper metadata and findings",
            target_entities=[
                "Paper",
                "Study",
                "Experiment",
                "Hypothesis",
                "Finding",
            ],
            target_relationships=[
                "authors",
                "published_in",
                "cites",
                "confirms",
                "refutes",
            ],
            priority=1,
            examples=["Nature journal", "peer-reviewed", "control group"],
        ),
        ExtractionGoal(
            name="chemicals",
            description="Extract chemical compounds and materials",
            target_entities=[
                "Compound",
                "Element",
                "Molecule",
                "Material",
                "Isotope",
            ],
            target_relationships=[
                "composed_of",
                "reacts_with",
                "synthesized_from",
                "contains",
            ],
            priority=1,
            examples=["water", "CO2", "graphene", "H2O"],
        ),
        ExtractionGoal(
            name="biological_entities",
            description="Extract biological entities and organisms",
            target_entities=[
                "Organism",
                "Species",
                "Gene",
                "Protein",
                "Cell",
            ],
            target_relationships=[
                "belongs_to",
                "encodes",
                "interacts_with",
                "expressed_in",
            ],
            priority=2,
            examples=["E. coli", "BRCA1", "hemoglobin", "Homo sapiens"],
        ),
        ExtractionGoal(
            name="physical_properties",
            description="Extract physical properties and measurements",
            target_entities=[
                "Property",
                "Measurement",
                "Constant",
                "Unit",
                "Value",
            ],
            target_relationships=[
                "measured_as",
                "equals",
                "varies_with",
                "approximates",
            ],
            priority=3,
            examples=["boiling point", "Avogadro's number", "10 meters"],
        ),
    ],
    "GENERAL": [
        ExtractionGoal(
            name="entities",
            description="Extract named entities (people, organizations, locations)",
            target_entities=[
                "Person",
                "Organization",
                "Location",
                "Event",
                "Product",
                "Concept",
            ],
            target_relationships=[
                "related_to",
                "located_in",
                "participated_in",
                "produces",
                "known_for",
            ],
            priority=1,
            examples=["John", "Google", "New York", "WWII", "iPhone"],
        ),
        ExtractionGoal(
            name="events",
            description="Extract events and temporal information",
            target_entities=[
                "Event",
                "Date",
                "Time",
                "Occurrence",
                "Milestone",
            ],
            target_relationships=[
                "occurred_on",
                "precedes",
                "follows",
                "during",
                "lasted_for",
            ],
            priority=2,
            examples=["January 1, 2024", "3 hours", "World Cup"],
        ),
        ExtractionGoal(
            name="concepts",
            description="Extract abstract concepts and ideas",
            target_entities=[
                "Concept",
                "Theory",
                "Method",
                "Principle",
                "Framework",
            ],
            target_relationships=[
                "based_on",
                "applies_to",
                "enables",
                "relies_on",
            ],
            priority=3,
            examples=["democracy", "evolution", "Agile methodology"],
        ),
    ],
}


def get_default_goals(domain: str) -> List[ExtractionGoal]:
    """Get default extraction goals for a domain.

    Args:
        domain: Domain name (e.g., 'TECHNOLOGY', 'FINANCIAL').

    Returns:
        List of extraction goals for the domain.
    """
    return DEFAULT_GOALS.get(domain.upper(), DEFAULT_GOALS["GENERAL"])
