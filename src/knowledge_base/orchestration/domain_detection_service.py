"""Domain detection service with intelligent classification.

This module provides domain detection that:
1. First determines if content is crypto-related
2. If crypto: uses specific crypto domains (BITCOIN, DEFI, TRADING, etc.)
3. If not crypto: uses general domains (TECHNOLOGY, HEALTHCARE, FINANCE, LEGAL, SCIENCE)
4. Falls back to GENERAL only when uncertain
"""

import json
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from knowledge_base.orchestration.base_service import BaseService
from knowledge_base.persistence.v1.schema import Document
from knowledge_base.clients.llm import AsyncLLMClient


# Comprehensive domain definitions with rich metadata
DOMAIN_DEFINITIONS = {
    # ============ CRYPTO DOMAINS ============
    "BITCOIN": {
        "category": "crypto",
        "description": "Bitcoin-specific content: BTC price analysis, halving events, mining operations, network difficulty, hash rate, ETFs (IBIT, FBTC), institutional treasuries (Strategy, Tesla), Satoshi Nakamoto, Lightning Network",
        "keywords": [
            "bitcoin",
            "btc",
            "satoshi",
            "nakamoto",
            "mining",
            "miner",
            "halving",
            "block reward",
            "hash rate",
            "difficulty adjustment",
            "network hash",
            "etf",
            "spot etf",
            "ibit",
            "fbtc",
            "arkb",
            "treasury",
            "microstrategy",
            "strategy",
            "michael saylor",
            "lightning network",
            "taproot",
            "ordinal",
            "inscription",
        ],
        "indicators": ["btc", "bitcoin", "satoshi", "halving", "mining", "treasury"],
    },
    "DEFI": {
        "category": "crypto",
        "description": "Decentralized Finance: AMMs, liquidity pools, yield farming, lending protocols (Aave, Compound), DEXs (Uniswap, Curve), vaults, governance tokens, TVL, impermanent loss, flash loans",
        "keywords": [
            "defi",
            "decentralized finance",
            "amm",
            "automated market maker",
            "liquidity pool",
            "yield farming",
            "yield",
            "farming",
            "harvest",
            "lending",
            "borrow",
            "collateral",
            "aave",
            "compound",
            "makerdao",
            "dex",
            "decentralized exchange",
            "uniswap",
            "pancakeswap",
            "curve",
            "balancer",
            "tvl",
            "total value locked",
            "apy",
            "apr",
            "governance token",
            "dao",
            "protocol",
            "impermanent loss",
            "slippage",
            "flash loan",
            "vault",
            "strategy",
        ],
        "indicators": ["defi", "uniswap", "aave", "liquidity", "yield", "tvl", "dex"],
    },
    "TRADING": {
        "category": "crypto",
        "description": "Trading and Technical Analysis: Price action, chart patterns, technical indicators (RSI, MACD, Ichimoku, Bollinger), order book analysis, market microstructure, candlestick patterns, support/resistance",
        "keywords": [
            "trading",
            "trader",
            "trade",
            "technical analysis",
            "chart",
            "price action",
            "indicator",
            "rsi",
            "macd",
            "ichimoku",
            "bollinger",
            "fibonacci",
            "support",
            "resistance",
            "trend",
            "momentum",
            "volume",
            "candlestick",
            "doji",
            "hammer",
            "engulfing",
            "order book",
            "bid",
            "ask",
            "spread",
            "depth",
            "liquidity",
            "long",
            "short",
            "leverage",
            "margin",
            "liquidation",
            "breakout",
            "pullback",
            "consolidation",
            "accumulation",
            "swing trading",
            "day trading",
            "scalping",
        ],
        "indicators": [
            "trading",
            "indicator",
            "chart",
            "fibonacci",
            "candlestick",
            "order book",
        ],
    },
    "INSTITUTIONAL_CRYPTO": {
        "category": "crypto",
        "description": "Institutional Adoption: ETF issuers (BlackRock, Fidelity), custody solutions, corporate treasuries, institutional trading desks, prime brokerage, compliance frameworks",
        "keywords": [
            "institutional",
            "institution",
            "corporate",
            "blackrock",
            "fidelity",
            "grayscale",
            "invesco",
            "vaneck",
            "custody",
            "custodian",
            "prime brokerage",
            "treasury",
            "balance sheet",
            "allocation",
            "accredited investor",
            "qualified custodian",
            "etf approval",
            "sec approval",
            "spot bitcoin etf",
        ],
        "indicators": ["blackrock", "fidelity", "custody", "treasury", "institutional"],
    },
    "STABLECOINS": {
        "category": "crypto",
        "description": "Stablecoins: USDC, USDT, DAI, reserve audits, backing mechanisms, depegging events, regulatory frameworks (GENIUS Act), cross-border payments, remittances",
        "keywords": [
            "stablecoin",
            "stable coin",
            "usdc",
            "usdt",
            "dai",
            "tusd",
            "busd",
            "pegged",
            "peg",
            "depeg",
            "de-pegging",
            "reserve",
            "backing",
            "attestation",
            "audit",
            "circle",
            "tether",
            "makerdao",
            "sky",
            "genius act",
            "payment stablecoin",
            "remittance",
            "cross-border payment",
        ],
        "indicators": ["stablecoin", "usdc", "usdt", "dai", "pegged", "reserve"],
    },
    "CRYPTO_REGULATION": {
        "category": "crypto",
        "description": "Cryptocurrency Regulation: SEC enforcement, CFTC jurisdiction, legislation (MiCA, GENIUS Act), compliance, KYC/AML, securities law, exchange registration",
        "keywords": [
            "regulation",
            "regulatory",
            "compliance",
            "sec",
            "securities and exchange commission",
            "gary gensler",
            "cftc",
            "commodity futures",
            "mica",
            "markets in crypto-assets",
            "genius act",
            "legislation",
            "bill",
            "congress",
            "senate",
            "enforcement",
            "lawsuit",
            "litigation",
            "settlement",
            "fine",
            "kyc",
            "aml",
            "anti-money laundering",
            "know your customer",
            "registered",
            "license",
            "permit",
        ],
        "indicators": [
            "sec",
            "regulation",
            "compliance",
            "mica",
            "cftc",
            "enforcement",
        ],
    },
    # ============ GENERAL DOMAINS ============
    "TECHNOLOGY": {
        "category": "general",
        "description": "Technology and Software: Programming, APIs, databases, cloud computing, AI/ML, web development, system architecture, DevOps, cybersecurity",
        "keywords": [
            "software",
            "programming",
            "code",
            "developer",
            "engineer",
            "api",
            "interface",
            "endpoint",
            "rest",
            "graphql",
            "database",
            "sql",
            "nosql",
            "storage",
            "cloud",
            "aws",
            "azure",
            "gcp",
            "serverless",
            "ai",
            "machine learning",
            "ml",
            "neural network",
            "deep learning",
            "web",
            "frontend",
            "backend",
            "fullstack",
            "javascript",
            "python",
            "devops",
            "ci/cd",
            "docker",
            "kubernetes",
            "microservices",
            "security",
            "cybersecurity",
            "encryption",
            "authentication",
        ],
        "indicators": [
            "software",
            "programming",
            "api",
            "database",
            "cloud",
            "ai",
            "web",
        ],
    },
    "HEALTHCARE": {
        "category": "general",
        "description": "Healthcare and Medicine: Medical research, clinical trials, patient care, pharmaceuticals, diagnosis, treatment, public health",
        "keywords": [
            "healthcare",
            "medical",
            "medicine",
            "clinical",
            "patient",
            "doctor",
            "physician",
            "nurse",
            "hospital",
            "diagnosis",
            "treatment",
            "therapy",
            "surgery",
            "procedure",
            "pharmaceutical",
            "drug",
            "medication",
            "vaccine",
            "research",
            "trial",
            "study",
            "biotech",
            "disease",
            "condition",
            "symptom",
            "pathology",
        ],
        "indicators": [
            "medical",
            "patient",
            "doctor",
            "hospital",
            "diagnosis",
            "treatment",
        ],
    },
    "FINANCE": {
        "category": "general",
        "description": "Traditional Finance: Banking, investment, stock markets, bonds, forex, portfolio management, financial analysis (non-crypto)",
        "keywords": [
            "bank",
            "banking",
            "credit union",
            "stock",
            "equity",
            "share",
            "dividend",
            "bond",
            "fixed income",
            "treasury bill",
            "forex",
            "currency",
            "exchange rate",
            "fiat",
            "portfolio",
            "asset allocation",
            "diversification",
            "mutual fund",
            "etfindex fund",
            "hedge fund",
            "financial statement",
            "balance sheet",
            "income statement",
            "audit",
            "accounting",
            "tax",
        ],
        "indicators": ["bank", "stock", "bond", "forex", "portfolio"],
        "exclude_if": [
            "bitcoin",
            "btc",
            "crypto",
            "blockchain",
            "defi",
            "nft",
        ],  # Exclude if crypto terms present
    },
    "LEGAL": {
        "category": "general",
        "description": "Law and Legal: Contracts, litigation, compliance, intellectual property, corporate law, regulatory (non-crypto specific)",
        "keywords": [
            "law",
            "legal",
            "attorney",
            "lawyer",
            "counsel",
            "contract",
            "agreement",
            "terms",
            "clause",
            "litigation",
            "lawsuit",
            "dispute",
            "arbitration",
            "intellectual property",
            "patent",
            "trademark",
            "copyright",
            "corporate law",
            "mergers",
            "acquisitions",
            "m&a",
            "compliance",
            "regulatory",
            "governance",
        ],
        "indicators": ["legal", "lawyer", "contract", "litigation", "lawsuit"],
        "exclude_if": [
            "sec enforcement",
            "crypto regulation",
            "mica",
        ],  # Exclude crypto-regulation
    },
    "SCIENCE": {
        "category": "general",
        "description": "Science and Research: Academic research, experiments, physics, chemistry, biology, data analysis, publications",
        "keywords": [
            "research",
            "study",
            "experiment",
            "hypothesis",
            "theory",
            "scientific",
            "academic",
            "publication",
            "journal",
            "paper",
            "physics",
            "chemistry",
            "biology",
            "mathematics",
            "statistics",
            "laboratory",
            "lab",
            "observation",
            "measurement",
            "data analysis",
            "methodology",
            "peer review",
        ],
        "indicators": ["research", "scientific", "academic", "study", "experiment"],
    },
    # ============ FALLBACK ============
    "GENERAL": {
        "category": "fallback",
        "description": "General content: Mixed topics, market overviews, news, educational content that doesn't fit specific domains",
        "keywords": [],
        "indicators": [],
    },
}


class DomainDetectionService(BaseService):
    """Intelligent domain detection service.

    Detects both crypto and non-crypto domains without bias.
    Uses two-phase detection: crypto check → domain classification.
    """

    # Keywords that indicate crypto content
    CRYPTO_INDICATORS = [
        # Core crypto terms
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "cryptocurrency",
        "blockchain",
        "token",
        "altcoin",
        "defi",
        "nft",
        "web3",
        "smart contract",
        # Exchanges
        "binance",
        "coinbase",
        "kraken",
        "uniswap",
        "pancakeswap",
        # Specific tokens
        "solana",
        "sol",
        "cardano",
        "ada",
        "polkadot",
        "dot",
        "avalanche",
        "avax",
        # Concepts
        "mining",
        "staking",
        "yield",
        "liquidity",
        "wallet",
        "private key",
        # Stablecoins
        "stablecoin",
        "usdc",
        "usdt",
        "dai",
        # Trading analysis (when combined with price/volume context)
        "bullish",
        "bearish",
        "support",
        "resistance",
        "fibonacci",
        "bollinger",
        "ichimoku",
        "candlestick",
        "chart",
        "trading",
    ]

    def __init__(self):
        super().__init__()
        self._llm_client: Optional[AsyncLLMClient] = None
        self._use_llm = True

    async def initialize(self, llm_client: Optional[AsyncLLMClient] = None) -> None:
        """Initialize the service."""
        self._llm_client = llm_client or AsyncLLMClient()
        self._logger.info(
            "DomainDetectionService initialized (intelligent classification)"
        )

    async def shutdown(self) -> None:
        """Shutdown the service."""
        if self._llm_client:
            await self._llm_client.close()
        self._logger.info("DomainDetectionService shutdown")

    def _is_crypto_content(self, text: str) -> bool:
        """Quick check if content is crypto-related.

        Args:
            text: Content to analyze

        Returns:
            True if crypto-related
        """
        if not text:
            return False

        text_lower = text.lower()
        matches = sum(
            1 for indicator in self.CRYPTO_INDICATORS if indicator in text_lower
        )

        # If 2+ crypto indicators found, consider it crypto content
        return matches >= 2

    def _calculate_keyword_scores(self, text: str, is_crypto: bool) -> Dict[str, float]:
        """Calculate domain scores based on keyword frequency.

        Args:
            text: Content to analyze
            is_crypto: Whether content is crypto-related

        Returns:
            Dictionary of domain scores
        """
        if not text or not text.strip():
            return {}

        text_lower = text.lower()
        scores = {}

        for domain, info in DOMAIN_DEFINITIONS.items():
            # Skip crypto domains for non-crypto content
            if info.get("category") == "crypto" and not is_crypto:
                continue

            # Skip general domains for crypto content (except GENERAL fallback)
            if info.get("category") == "general" and is_crypto:
                continue

            score = 0.0
            keywords = info.get("keywords", [])

            for keyword in keywords:
                count = text_lower.count(keyword.lower())
                # Weight by keyword specificity (longer = more specific)
                weight = min(len(keyword) / 10, 2.0)
                score += count * weight

            # Check for exclusion terms
            exclude_if = info.get("exclude_if", [])
            if exclude_if:
                for exclude_term in exclude_if:
                    if exclude_term.lower() in text_lower:
                        # Reduce score if exclusion terms present
                        score *= 0.3

            if score > 0:
                scores[domain] = score

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    async def detect_domain(
        self, document: Document, content_text: str | None = None
    ) -> str:
        """Determine domain for a document.

        Args:
            document: The document to classify
            content_text: Optional content text

        Returns:
            Domain string
        """
        # Check for explicitly set domain in metadata
        if document.doc_metadata and "domain" in document.doc_metadata:
            return str(document.doc_metadata["domain"]).upper()

        # Determine if crypto-related
        is_crypto = self._is_crypto_content(content_text or "")

        self._logger.info(
            f"Content appears to be {'crypto-related' if is_crypto else 'non-crypto'}"
        )

        # Try LLM-based detection first
        if self._use_llm and self._llm_client and content_text:
            try:
                llm_domain = await self._detect_domain_with_llm(content_text, is_crypto)
                if llm_domain and llm_domain != "ERROR":
                    self._logger.info(f"LLM detected domain: {llm_domain}")
                    return llm_domain
            except Exception as e:
                self._logger.warning(
                    f"LLM domain detection failed, using keyword fallback: {e}"
                )

        # Keyword-based fallback
        return self._detect_domain_keyword_based(content_text or "", is_crypto)

    async def _detect_domain_with_llm(
        self, content_text: str, is_crypto: bool
    ) -> str | None:
        """Use LLM to detect domain.

        Args:
            content_text: Content to analyze
            is_crypto: Whether content is crypto-related

        Returns:
            Detected domain or None
        """
        # Truncate if too long
        max_chars = 3000
        truncated = (
            content_text[:max_chars] + "..."
            if len(content_text) > max_chars
            else content_text
        )

        # Build domain list based on content type
        if is_crypto:
            relevant_domains = {
                k: v
                for k, v in DOMAIN_DEFINITIONS.items()
                if v.get("category") == "crypto"
            }
        else:
            relevant_domains = {
                k: v
                for k, v in DOMAIN_DEFINITIONS.items()
                if v.get("category") in ["general", "fallback"]
            }

        domain_descriptions = "\n".join(
            [
                f"- {domain}: {info['description']}"
                for domain, info in relevant_domains.items()
            ]
        )

        category = "cryptocurrency" if is_crypto else "general"

        prompt = f"""Analyze the following {category} document and classify it into the most appropriate domain.

Available domains:
{domain_descriptions}

Document content (first part):
{truncated}

Respond with ONLY a JSON object:
{{"domain": "DOMAIN_NAME", "confidence": 0.0-1.0, "reasoning": "Brief explanation"}}

Rules:
- domain must be one of: {", ".join(relevant_domains.keys())}
- confidence should reflect clarity of domain match
- Be objective - don't force crypto classification on non-crypto content
- Respond with valid JSON only"""

        try:
            response = await self._llm_client.complete(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )

            content = response.get("content", "")

            # Clean up response
            content = self._clean_llm_response(content)

            if not content:
                return None

            result = json.loads(content)
            detected_domain = result.get("domain", "GENERAL").upper()
            confidence = result.get("confidence", 0.0)

            # Validate domain
            if detected_domain not in DOMAIN_DEFINITIONS:
                self._logger.warning(f"LLM returned invalid domain: {detected_domain}")
                return None

            # Use result if confidence is high enough
            if confidence >= 0.5:
                return detected_domain
            else:
                self._logger.info(f"LLM confidence too low ({confidence})")
                return None

        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse LLM response: {e}")
            return None
        except Exception as e:
            self._logger.error(f"LLM error: {e}")
            return None

    def _clean_llm_response(self, content: str) -> str:
        """Clean up LLM response."""
        content = content.strip()

        # Remove thinking tags
        if "<think>" in content and "</think>" in content:
            think_end = content.find("</think>") + len("</think>")
            content = content[think_end:].strip()

        # Remove markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return content.strip()

    def _detect_domain_keyword_based(self, content_text: str, is_crypto: bool) -> str:
        """Keyword-based domain detection.

        Args:
            content_text: Content to analyze
            is_crypto: Whether content is crypto-related

        Returns:
            Detected domain
        """
        scores = self._calculate_keyword_scores(content_text, is_crypto)

        if not scores:
            return "GENERAL"

        # Get highest scoring domain
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        # Only use keyword result if score is significant
        if best_score >= 0.15:
            self._logger.info(
                f"Keyword detection: {best_domain} (score: {best_score:.2f})"
            )
            return best_domain
        else:
            return "GENERAL"

    def get_domain_info(self, domain: str) -> Dict:
        """Get information about a domain.

        Args:
            domain: Domain name

        Returns:
            Domain information dictionary
        """
        return DOMAIN_DEFINITIONS.get(domain, DOMAIN_DEFINITIONS["GENERAL"])
