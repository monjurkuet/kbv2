"""Automated Prompt Evolution System for KBv2.

Evolves extraction prompts through mutation, A/B testing, and selection
to optimize performance for specific domains.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PromptVariant(BaseModel):
    """A prompt variant for testing."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., description="Variant name")
    prompt_text: str = Field(..., description="The prompt text")
    domain: str = Field(..., description="Target domain")
    mutation_strategy: str = Field(..., description="How this variant was created")
    parent_id: Optional[UUID] = None
    generation: int = Field(default=1, description="Evolution generation")

    # Performance metrics
    total_evaluations: int = 0
    successful_extractions: int = 0
    avg_quality_score: float = 0.0
    avg_entity_count: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0

    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


class PromptEvaluationResult(BaseModel):
    """Result of evaluating a prompt on a document."""

    variant_id: UUID
    document_id: UUID
    quality_score: float
    entity_count: int
    entity_types: List[str]
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PromptEvolutionConfig(BaseModel):
    """Configuration for prompt evolution."""

    # Mutation settings
    num_variants_per_generation: int = 5
    max_generations: int = 10
    mutation_temperature: float = 0.7

    # Evaluation settings
    min_evaluation_samples: int = 10
    evaluation_sample_rate: float = 0.2

    # Selection criteria
    selection_threshold: float = 0.75
    min_improvement: float = 0.05

    # Crypto-specific focus
    crypto_domains: List[str] = [
        "BITCOIN",
        "DEFI",
        "INSTITUTIONAL_CRYPTO",
        "STABLECOINS",
        "CRYPTO_REGULATION",
    ]


class CryptoPromptTemplates:
    """Base templates for crypto domain prompts."""

    BASE_ENTITY_EXTRACTION = """You are an expert cryptocurrency entity extraction system.

Your task is to extract entities from cryptocurrency and blockchain-related text with high precision.

Extract the following entity types:
{entity_types}

Guidelines:
1. Extract entities with exact text spans
2. Include entity type, description, and confidence score
3. Capture relationships between entities
4. Include relevant properties (e.g., amounts, dates, metrics)
5. Focus on factual, verifiable information

Output format (JSON):
{{
  "entities": [
    {{
      "name": "entity name",
      "entity_type": "TYPE",
      "description": "brief description",
      "confidence": 0.95,
      "properties": {{}}
    }}
  ],
  "relationships": [
    {{
      "source": "entity name",
      "target": "entity name",
      "relationship_type": "TYPE",
      "confidence": 0.90
    }}
  ]
}}

Text to analyze:
{text}
"""

    BITCOIN_FOCUS = """You are a Bitcoin specialist extraction system.

Focus on extracting:
- Bitcoin-specific entities: BTC amounts, addresses, transactions
- Mining entities: pools, difficulty, hash rate, halving events
- Institutional products: ETFs (IBIT, GBTC, FBTC), treasuries
- Network metrics: Lightning Network, nodes, fees
- Market data: MVRV, NUPL, HODL waves, realized price

Key entities to extract:
- BitcoinETF: ETF name, issuer, AUM, flows, holdings
- MiningPool: pool name, hash rate, blocks mined
- DigitalAssetTreasury: company name, BTC holdings, cost basis
- BitcoinUpgrade: BIP number, activation date, description

Extract specific metrics:
- "Strategy holds 471,107 BTC" → quantity: 471107
- "Hash rate reached 500 EH/s" → hash_rate: 500
- "IBIT inflow of $500M" → inflow: 500000000

{text}
"""

    DEFI_FOCUS = """You are a DeFi protocol specialist extraction system.

Focus on extracting:
- DeFi protocols: Aave, Compound, Uniswap, Morpho, etc.
- Financial metrics: TVL, APY, APR, market cap, FDV
- Liquidity pools: token pairs, TVL, volume
- Governance: tokens, proposals, voting
- Yield strategies: vaults, farming, staking

Key entities to extract:
- DeFiProtocol: name, TVL, category, chain
- LiquidityPool: protocol, token pairs, TVL, APY
- GovernanceToken: symbol, market cap, utility
- YieldStrategy: protocol, APY, risk level

Extract specific metrics:
- "Aave V3 has $12.5B TVL" → tvl: 12500000000
- "Supply APY of 3.2%" → supply_apy: 3.2
- "Uniswap ETH-USDC pool" → tokens: ["ETH", "USDC"]

{text}
"""

    INSTITUTIONAL_FOCUS = """You are an institutional crypto adoption specialist.

Focus on extracting:
- ETF issuers: BlackRock, Grayscale, Fidelity, products
- Custody providers: Coinbase, BitGo, BNY Mellon
- Corporate treasuries: Strategy, Tesla, Block, Metaplanet
- Regulatory entities: SEC, CFTC, legislation
- Financial metrics: AUM, expense ratios, flows, cost basis

Key entities to extract:
- ETFIssuer: company name, ETFs managed, AUM
- BitcoinETF: ticker, issuer, type (spot/futures), expense ratio
- CryptoCustodian: company, assets under custody, clients
- DigitalAssetTreasury: company, BTC holdings, average cost

Extract specific metrics:
- "BlackRock's IBIT has $45B AUM" → aum: 45000000000
- "Expense ratio of 0.19%" → expense_ratio: 0.19
- "Daily inflow of $500M" → daily_inflow: 500000000

{text}
"""

    STABLECOIN_FOCUS = """You are a stablecoin specialist extraction system.

Focus on extracting:
- Stablecoins: USDC, USDT, DAI, PYUSD, etc.
- Issuers: Circle, Tether, MakerDAO
- Backing mechanisms: fiat, crypto, commodity, algorithmic
- Regulatory compliance: GENIUS Act, attestations
- Market data: supply, market cap, velocity, peg stability

Key entities to extract:
- Stablecoin: symbol, issuer, backing_type, market_cap
- StablecoinIssuer: company, reserves, attestations
- ReserveAsset: type, amount, composition

Extract specific metrics:
- "USDC market cap of $42B" → market_cap: 42000000000
- "Backed 1:1 with cash and treasuries" → backing: "fiat"
- "Monthly attestation from Grant Thornton" → auditor: "Grant Thornton"

{text}
"""

    REGULATION_FOCUS = """You are a crypto regulation specialist extraction system.

Focus on extracting:
- Regulatory bodies: SEC, CFTC, FINMA, FCA, etc.
- Legislation: GENIUS Act, MiCA, market structure bills
- Enforcement actions: cases, settlements, penalties
- Compliance frameworks: KYC, AML, travel rule
- Legal entities: exchanges, funds under investigation

Key entities to extract:
- RegulatoryBody: name, jurisdiction, authority
- Regulation: name, jurisdiction, effective date, status
- LegalCase: parties, charges, outcome, penalties
- ComplianceFramework: name, requirements, applicability

Extract specific details:
- "SEC v. Coinbase" → parties: ["SEC", "Coinbase"]
- "GENIUS Act passed June 2025" → date: "2025-06"
- "$4.3B penalty against Binance" → penalty: 4300000000

{text}
"""

    @classmethod
    def get_base_prompt(cls, domain: str) -> str:
        """Get the base prompt for a domain."""
        prompts = {
            "BITCOIN": cls.BITCOIN_FOCUS,
            "DEFI": cls.DEFI_FOCUS,
            "INSTITUTIONAL_CRYPTO": cls.INSTITUTIONAL_FOCUS,
            "STABLECOINS": cls.STABLECOIN_FOCUS,
            "CRYPTO_REGULATION": cls.REGULATION_FOCUS,
        }
        return prompts.get(domain, cls.BASE_ENTITY_EXTRACTION)


class PromptEvolutionEngine:
    """Engine for evolving extraction prompts."""

    def __init__(self, gateway, config: Optional[PromptEvolutionConfig] = None):
        self.gateway = gateway
        self.config = config or PromptEvolutionConfig()
        self.templates = CryptoPromptTemplates()
        self.variants: Dict[str, List[PromptVariant]] = {}  # domain -> variants
        self.evaluation_results: List[PromptEvaluationResult] = []

    async def initialize_domain(self, domain: str) -> List[PromptVariant]:
        """Initialize prompt variants for a domain.

        Args:
            domain: Domain to initialize

        Returns:
            List of initial variants
        """
        base_prompt = self.templates.get_base_prompt(domain)

        # Create initial variant
        variants = [
            PromptVariant(
                name=f"{domain}_base",
                prompt_text=base_prompt,
                domain=domain,
                mutation_strategy="base_template",
                generation=1,
            )
        ]

        # Generate mutated variants
        mutations = await self._generate_mutations(
            base_prompt=base_prompt,
            domain=domain,
            n_variants=self.config.num_variants_per_generation - 1,
        )

        variants.extend(mutations)
        self.variants[domain] = variants

        return variants

    async def _generate_mutations(
        self, base_prompt: str, domain: str, n_variants: int
    ) -> List[PromptVariant]:
        """Generate mutated prompt variants.

        Args:
            base_prompt: Base prompt to mutate
            domain: Target domain
            n_variants: Number of variants to generate

        Returns:
            List of mutated variants
        """
        mutation_prompt = f"""You are an expert prompt engineer for cryptocurrency entity extraction.

Generate {n_variants} variations of the following extraction prompt for the {domain} domain.

Each variation should emphasize different aspects:
1. Focus on PRECISION - reduce false positives, strict entity boundaries
2. Focus on RECALL - catch more entities, comprehensive coverage
3. Focus on RELATIONSHIPS - extract connections between entities
4. Focus on METRICS - extract numerical data, amounts, percentages
5. Focus on CONTEXT - understand entity significance in context

Base prompt:
{base_prompt}

For each variation:
1. Keep the core extraction task
2. Adjust the emphasis as specified
3. Add domain-specific examples
4. Modify instructions to match the focus

Output as JSON array:
[
  {{
    "name": "variant_name",
    "focus": "precision|recall|relationships|metrics|context",
    "prompt_text": "full prompt text"
  }}
]
"""

        try:
            response = await self.gateway.complete(
                prompt=mutation_prompt, temperature=self.config.mutation_temperature
            )

            # Access content from dictionary response
            prompt_content = response["content"]

            # Parse mutations
            mutations_data = json.loads(prompt_content)

            variants = []
            for i, data in enumerate(mutations_data[:n_variants]):
                variant = PromptVariant(
                    name=f"{domain}_gen1_{data.get('focus', f'var{i}')}",
                    prompt_text=data.get("prompt_text", base_prompt),
                    domain=domain,
                    mutation_strategy=data.get("focus", "unknown"),
                    generation=1,
                )
                variants.append(variant)

            return variants

        except (json.JSONDecodeError, KeyError, Exception) as e:
            import logging

            logging.getLogger(__name__).error(
                f"Error generating mutations for domain {domain}: {e}"
            )
            # Return empty list on failure
            return []

    async def evaluate_variant(
        self, variant: PromptVariant, test_documents: List[Dict[str, Any]]
    ) -> List[PromptEvaluationResult]:
        """Evaluate a prompt variant on test documents.

        Args:
            variant: Prompt variant to evaluate
            test_documents: Documents to test on

        Returns:
            List of evaluation results
        """
        results = []

        for doc in test_documents:
            try:
                # Run extraction with variant prompt
                start_time = datetime.utcnow()

                extraction_result = await self._extract_with_prompt(
                    prompt=variant.prompt_text, text=doc.get("text", "")
                )

                processing_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Calculate metrics
                entity_count = len(extraction_result.get("entities", []))
                entity_types = list(
                    set(
                        e.get("entity_type", "Unknown")
                        for e in extraction_result.get("entities", [])
                    )
                )

                # Estimate quality (in production, use ground truth comparison)
                quality_score = self._estimate_quality(
                    extraction_result, doc.get("expected_entities", [])
                )

                result = PromptEvaluationResult(
                    variant_id=variant.id,
                    document_id=doc.get("id", uuid4()),
                    quality_score=quality_score,
                    entity_count=entity_count,
                    entity_types=entity_types,
                    processing_time_ms=processing_time,
                )

                results.append(result)

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Validation error evaluating document {doc.get('id')}: {e}"
                )
                continue
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(
                    f"Unexpected error evaluating document {doc.get('id')}: {e}",
                    exc_info=True,
                )
                continue

        # Update variant metrics
        if results:
            variant.total_evaluations += len(results)
            variant.avg_quality_score = sum(r.quality_score for r in results) / len(
                results
            )
            variant.avg_entity_count = sum(r.entity_count for r in results) / len(
                results
            )

        return results

    async def evolve_generation(
        self, domain: str, test_documents: List[Dict[str, Any]]
    ) -> PromptVariant:
        """Evolve prompts to next generation.

        Args:
            domain: Domain to evolve
            test_documents: Test documents for evaluation

        Returns:
            Best performing variant
        """
        if domain not in self.variants:
            await self.initialize_domain(domain)

        variants = self.variants[domain]

        # Evaluate all variants
        for variant in variants:
            if not variant.is_active:
                continue

            results = await self.evaluate_variant(variant, test_documents)
            self.evaluation_results.extend(results)

        # Select best variant
        best_variant = max(variants, key=lambda v: v.avg_quality_score)

        # Generate next generation if needed
        if best_variant.generation < self.config.max_generations:
            # Create mutations from best performer
            new_mutations = await self._generate_mutations(
                base_prompt=best_variant.prompt_text,
                domain=domain,
                n_variants=self.config.num_variants_per_generation,
            )

            # Mark new generation
            for mutation in new_mutations:
                mutation.generation = best_variant.generation + 1
                mutation.parent_id = best_variant.id

            # Add to variants
            variants.extend(new_mutations)

            # Deactivate poor performers
            for variant in variants:
                if (
                    variant.total_evaluations >= self.config.min_evaluation_samples
                    and variant.avg_quality_score < self.config.selection_threshold
                ):
                    variant.is_active = False

        return best_variant

    async def get_best_prompt(self, domain: str) -> str:
        """Get the best performing prompt for a domain.

        Args:
            domain: Domain to get prompt for

        Returns:
            Best prompt text
        """
        if domain not in self.variants:
            await self.initialize_domain(domain)

        variants = self.variants[domain]

        # Filter active variants with evaluations
        evaluated = [v for v in variants if v.total_evaluations > 0 and v.is_active]

        if evaluated:
            best = max(evaluated, key=lambda v: v.avg_quality_score)
            return best.prompt_text
        else:
            # Return base prompt if no evaluations yet
            return self.templates.get_base_prompt(domain)

    async def _extract_with_prompt(self, prompt: str, text: str) -> Dict[str, Any]:
        """Extract entities using a specific prompt.

        Args:
            prompt: Prompt template
            text: Text to extract from

        Returns:
            Extraction result
        """
        formatted_prompt = prompt.format(text=text)

        response = await self.gateway.complete(prompt=formatted_prompt, temperature=0.1)

        try:
            return json.loads(response["content"])
        except (json.JSONDecodeError, KeyError):
            return {"entities": [], "relationships": []}

    def _estimate_quality(
        self, extraction: Dict[str, Any], expected: List[Dict[str, Any]]
    ) -> float:
        """Estimate extraction quality.

        In production, compare against ground truth.
        For now, use heuristics based on extraction structure.
        """
        entities = extraction.get("entities", [])

        if not entities:
            return 0.3

        # Check for entity types
        has_types = all(e.get("entity_type") for e in entities)

        # Check for confidence scores
        has_confidence = all(e.get("confidence") is not None for e in entities)

        # Check for properties
        has_properties = any(e.get("properties") for e in entities)

        # Calculate score
        score = 0.5  # Base score for having entities

        if has_types:
            score += 0.2
        if has_confidence:
            score += 0.1
        if has_properties:
            score += 0.1
        if len(entities) >= 3:
            score += 0.1

        return min(score, 1.0)

    def get_domain_statistics(self, domain: str) -> Dict[str, Any]:
        """Get statistics for a domain's prompt evolution.

        Args:
            domain: Domain to get stats for

        Returns:
            Statistics dictionary
        """
        if domain not in self.variants:
            return {"error": "Domain not initialized"}

        variants = self.variants[domain]

        active_variants = [v for v in variants if v.is_active]
        evaluated_variants = [v for v in variants if v.total_evaluations > 0]

        if evaluated_variants:
            best_variant = max(evaluated_variants, key=lambda v: v.avg_quality_score)
            avg_quality = sum(v.avg_quality_score for v in evaluated_variants) / len(
                evaluated_variants
            )
        else:
            best_variant = None
            avg_quality = 0.0

        return {
            "domain": domain,
            "total_variants": len(variants),
            "active_variants": len(active_variants),
            "evaluated_variants": len(evaluated_variants),
            "generations": max((v.generation for v in variants), default=1),
            "best_variant": {
                "id": str(best_variant.id) if best_variant else None,
                "name": best_variant.name if best_variant else None,
                "quality_score": best_variant.avg_quality_score
                if best_variant
                else 0.0,
                "evaluations": best_variant.total_evaluations if best_variant else 0,
            },
            "average_quality": avg_quality,
        }
