# KBv2 Cryptocurrency & Bitcoin Focused Refactoring Plan

## Executive Summary

This document provides a detailed analysis and refactoring plan to transform the KBv2 knowledgebase system into a **cryptocurrency and Bitcoin-focused knowledge management platform**. The plan leverages the existing sophisticated RAG architecture while customizing domain ontologies, entity extraction, and data models for crypto-specific content.

**Key 2026 Context**: The cryptocurrency market has fundamentally shifted from a four-year halving cycle to an institutional flow-driven regime. ETFs now move 12x daily mining supply, regulatory clarity (GENIUS Act, market structure legislation) is accelerating institutional adoption, and the industry is entering what Grayscale calls "The Dawn of the Institutional Era."

---

## Part 1: Current System Architecture Analysis

### 1.1 KBv2 Overview

KBv2 is a sophisticated **Agentic Knowledge Ingestion & Management System** with the following capabilities:

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector Database | PostgreSQL + pgvector (1024-dim) | Semantic search with bge-m3 embeddings |
| Knowledge Graph | SQLAlchemy + igraph/leidenalg | Entity-relationship modeling with community detection |
| Search | Hybrid (BM25 + Vector + Cross-encoder) | Multi-modal retrieval with reranking |
| LLM Integration | OpenAI-compatible API (local:8087) | Multi-agent extraction and synthesis |
| Ingestion Pipeline | 15-stage processing | Domain-aware extraction with auto-detection |
| API | FastAPI + WebSocket MCP | RESTful + real-time Model Context Protocol |

### 1.2 Current Domain Configurations

The system currently supports these domains:

```python
- GENERAL (default)
- TECHNOLOGY (software, APIs, frameworks)
- FINANCIAL (stocks, markets, traditional finance)
- MEDICAL (diseases, drugs, treatments)
- LEGAL (contracts, regulations, courts)
- SCIENTIFIC (research, studies, experiments)
- CRYPTO_TRADING (basic trading terms, indicators)
```

### 1.3 Existing Cryptocurrency Support

The `CRYPTO_TRADING` domain in `ontology_snippets.py` already includes:
- 200+ trading keywords (technical indicators, chart patterns)
- Basic entity types: TechnicalIndicator, ChartPattern, TradingStrategy
- Relationship types for technical analysis

**Gap Analysis**: This is trading-focused only and lacks:
- Bitcoin-specific knowledge (halving, mining, nodes, etc.)
- Layer-1/Layer-2 blockchain concepts
- DeFi protocols and primitives
- Institutional finance integration
- Regulatory and legal frameworks
- On-chain analytics entities

---

## Part 2: 2026 Cryptocurrency Market Research Summary

### 2.1 Key Market Trends for 2026

Based on research from Pantera Capital, Grayscale, Amberdata, and Coinbase:

#### A. Institutional Adoption Era
- **ETF Dominance**: ETF flows now exceed mining supply by 12x; institutional flows are the new marginal price driver
- **401(k) Integration**: DOL guidance expected Q1-Q2 2026 enabling retirement plan crypto allocation
- **Bank Custody**: Major banks (BNY Mellon, State Street, JPMorgan) launching crypto custody services
- **Wirehouse Adoption**: Morgan Stanley, Merrill Lynch, UBS managing $15T+ assets considering 1-3% crypto allocation

#### B. The End of the Four-Year Cycle
- April 2024 halving reduced supply by $40M/day, but ETFs move $500M-$1B daily
- Bitcoin driven by Fed policy, institutional flows, and regulatory catalysts rather than halving cycles
- Expected value for 2026: ~$109K (base case $90K-$120K, bull case $120K-$180K)

#### C. Regulatory Clarity
- **GENIUS Act**: Stablecoin legislation passed, enabling $300B+ stablecoin market growth
- **Market Structure Legislation**: Expected 2026 passage defining token classification (security vs commodity)
- **SAB 121 Rescinded**: Banks can now custody crypto on balance sheets
- **Staking ETFs**: Expected approval allowing yield-bearing ETF products

#### D. Key Growth Sectors

| Sector | Key Assets | Description |
|--------|------------|-------------|
| Store of Value | BTC, ETH, ZEC | Alternative monetary assets amid fiat debasement concerns |
| Stablecoins | USDC, USDT, PYUSD | $300B market, doubling in 2025, expected $500B+ in 2026 |
| DeFi | AAVE, MORPHO, UNI, LINK | Lending, DEXs, derivatives with TVL growth |
| Infrastructure | SUI, MON, NEAR, TAO | Next-gen L1s, AI-blockchain convergence |
| Asset Tokenization | CC, Ondo | Real-world assets on-chain ($16.6B TVL) |
| Privacy | ZEC, AZTEC | Confidential transactions as blockchain goes mainstream |

### 2.2 Knowledgebase Content Implications

A crypto-focused KB must capture:

1. **Technical Knowledge**: Bitcoin protocol, mining, halving mechanics, cryptography
2. **Market Intelligence**: Price analysis, ETF flows, institutional movements, on-chain metrics
3. **Regulatory Landscape**: SEC, CFTC, international regulations, compliance frameworks
4. **DeFi Ecosystem**: Protocols, yields, risks, smart contract interactions
5. **Institutional Products**: ETFs, trusts, custody solutions, treasury strategies
6. **Layer-2 Solutions**: Lightning Network, rollups, scaling solutions
7. **Economic Models**: Tokenomics, supply dynamics, monetary policy

---

## Part 3: Detailed Refactoring Plan

### 3.1 Phase 1: Domain Model Refactoring

#### 3.1.1 Update Domain Enum

**File**: `src/knowledge_base/domain/domain_models.py`

Replace the current domain structure with crypto-focused domains:

```python
class Domain(str, Enum):
    """Cryptocurrency and blockchain-focused domains."""
    
    # Core Bitcoin domain
    BITCOIN = "BITCOIN"  # Bitcoin-specific knowledge (protocol, mining, history)
    
    # Asset classes
    DIGITAL_ASSETS = "DIGITAL_ASSETS"  # General crypto assets, tokens
    STABLECOINS = "STABLECOINS"  # Stablecoin-specific knowledge
    
    # Infrastructure
    BLOCKCHAIN_INFRA = "BLOCKCHAIN_INFRA"  # L1/L2 chains, consensus mechanisms
    DEFI = "DEFI"  # Decentralized finance protocols
    
    # Market & Finance
    CRYPTO_MARKETS = "CRYPTO_MARKETS"  # Trading, markets, price analysis
    INSTITUTIONAL_CRYPTO = "INSTITUTIONAL_CRYPTO"  # ETFs, custody, treasuries
    
    # Regulatory & Legal
    CRYPTO_REGULATION = "CRYPTO_REGULATION"  # SEC, CFTC, compliance
    
    # Emerging
    CRYPTO_AI = "CRYPTO_AI"  # AI-blockchain convergence
    TOKENIZATION = "TOKENIZATION"  # RWA, asset tokenization
    
    # Fallback
    GENERAL = "GENERAL"
```

#### 3.1.2 Enhanced Entity Types

**File**: Update extraction configurations to include crypto-specific entities:

```python
# Cryptocurrency Entity Taxonomy
CRYPTO_ENTITY_TYPES = {
    # Assets
    "Cryptocurrency": "Digital currency or token",
    "Stablecoin": "Fiat-pegged digital currency",
    "GovernanceToken": "Protocol governance token",
    "UtilityToken": "Network utility token",
    "SecurityToken": "Regulated security on blockchain",
    
    # Bitcoin-Specific
    "BitcoinUpgrade": "BIP, soft fork, hard fork",
    "MiningPool": "Bitcoin mining pool",
    "LightningNode": "Lightning Network node/operator",
    "BitcoinETF": "Spot or futures Bitcoin ETF",
    
    # Infrastructure
    "Blockchain": "Layer-1 blockchain network",
    "Layer2": "Scaling solution (L2)",
    "SmartContract": "Programmable contract",
    "ConsensusMechanism": "PoW, PoS, etc.",
    "Validator": "Network validator/staker",
    
    # DeFi
    "DeFiProtocol": "Decentralized finance application",
    "DEX": "Decentralized exchange",
    "LendingProtocol": "Lending/borrowing platform",
    "YieldStrategy": "Yield farming strategy",
    "LiquidityPool": "DEX liquidity pool",
    
    # Market
    "CryptoExchange": "Centralized exchange",
    "MarketIndicator": "On-chain or market metric",
    "TradingPair": "Crypto trading pair",
    "ETFIssuer": "ETF provider (BlackRock, Grayscale, etc.)",
    
    # Institutional
    "CryptoCustodian": "Institutional custody provider",
    "DigitalAssetTreasury": "Corporate crypto treasury (DAT)",
    "CryptoFund": "Investment fund focused on crypto",
    
    # Regulatory
    "RegulatoryBody": "SEC, CFTC, etc.",
    "Regulation": "Specific law or regulation",
    "ComplianceFramework": "Compliance standard",
    
    # People/Organizations
    "CryptoFounder": "Protocol founder/developer",
    "CryptoAnalyst": "Industry analyst/researcher",
    "CryptoInstitution": "Company in crypto space",
}
```

### 3.2 Phase 2: Ontology Expansion

#### 3.2.1 Comprehensive Domain Ontologies

**File**: `src/knowledge_base/domain/ontology_snippets.py`

Expand with comprehensive crypto ontologies covering 2026 trends:

```python
DOMAIN_ONTOLOGIES = {
    "BITCOIN": {
        "keywords": [
            # Core Bitcoin terms
            "bitcoin", "btc", "satoshi", "sats", "satoshi nakamoto",
            "bitcoin protocol", "bitcoin network", "mainnet",
            
            # Monetary aspects
            "digital gold", "store of value", "sound money",
            "21 million", "scarcity", "fixed supply", "deflationary",
            "monetary policy", "disinflationary",
            
            # Mining & consensus
            "mining", "miner", "mining pool", "hash rate", "hashrate",
            "proof of work", "pow", "difficulty adjustment",
            "block reward", "coinbase reward", "halving",
            "2024 halving", "2028 halving", "block subsidy",
            "asic", "mining rig", "antminer", "energy consumption",
            "sustainable mining", "renewable mining",
            
            # Technical
            "blockchain", "block", "transaction", "confirmation",
            "mempool", "fee rate", "sats per byte", "transaction fee",
            "segwit", "taproot", "schnorr signatures", "bech32",
            "utxo", "unspent transaction output", "coin control",
            "full node", "light client", "spv",
            "bitcoin core", "bip", "bitcoin improvement proposal",
            "soft fork", "hard fork", "activation",
            
            # Scaling
            "lightning network", "ln", "payment channel",
            "rgb protocol", "omnibolt", "state chains",
            "sidechain", "liquid network", "rootstock", "rsk",
            "fedimint", "ecash", "cashu",
            
            # Wallets & custody
            "wallet", "hardware wallet", "cold storage",
            "multisig", "multi-signature", "seed phrase",
            "private key", "public key", "address",
            "hd wallet", "bip32", "bip39", "bip44",
            "paper wallet", "brain wallet",
            
            # Institutional
            "bitcoin etf", "spot etf", "futures etf",
            "strategy", "michael saylor", "microstrategy",
            "digital asset treasury", "dat",
            "nation state adoption", "sovereign wealth fund",
            "bitcoin reserve", "strategic bitcoin reserve",
            "corporate treasury", "bitcoin standard",
            
            # Markets
            "bitcoin dominance", "btc dominance", "btcd",
            "realized price", "mvrv", "nupl", "hodl waves",
            "long term holder", "short term holder", "lth", "sth",
            "accumulation", "distribution", "capitulation",
            "fear and greed index", "bitcoin volatility",
            
            # History & culture
            "genesis block", "pizza day", "mt gox", "silk road",
            "blocksize war", "big block", "small block",
            "bitcoin cash", "bch", "bitcoin sv", "bsv",
            "cypherpunk", "cryptography", "libertarian",
            "hodl", "not your keys", "stack sats",
        ],
        "entity_types": [
            "Bitcoin",
            "Satoshi",
            "MiningPool",
            "BitcoinUpgrade",
            "LightningNode",
            "BitcoinETF",
            "DigitalAssetTreasury",
            "BitcoinAnalyst",
            "BitcoinDeveloper",
            "NationState",
        ],
        "relationship_types": [
            "MINES",  # Miner mines Bitcoin
            "HOLDS",  # Entity holds Bitcoin
            "OPERATES",  # Entity operates mining pool/node
            "ISSUES",  # ETF issuer issues ETF
            "BACKS",  # ETF backs shares with Bitcoin
            "UPGRADED_TO",  # Protocol upgrade
            "BUILDS_ON",  # L2 builds on Bitcoin
            "ADOPTED_BY",  # Nation state adoption
        ],
    },
    
    "INSTITUTIONAL_CRYPTO": {
        "keywords": [
            # ETFs & ETPs
            "exchange traded fund", "etf", "etp",
            "spot bitcoin etf", "spot ethereum etf",
            "ibit", "gbtc", "fbtc", "arkb", "bito",
            "blackrock", "ishares", "grayscale", "fidelity",
            "vaneck", "ark invest", "bitwise", "wisdomtree",
            "invesco", "vanguard", "charles schwab",
            "in-kind redemption", "cash redemption",
            "creation basket", "redemption basket",
            "net asset value", "nav", "premium", "discount",
            "expense ratio", "management fee",
            "authorized participant", "ap",
            "market maker", "liquidity provider",
            
            # Custody
            "custody", "custodian", "qualified custodian",
            "cold storage", "multi-sig", "multi-signature",
            "bitgo", "coinbase custody", "fidelity digital assets",
            "bny mellon", "state street", "jpmorgan",
            "copper", "anchorage", "fireblocks",
            "soc 2", "soc 1", "type ii audit",
            "insurance coverage", "lloyd's of london",
            
            # Treasury & Corporate
            "digital asset treasury", "dat",
            "bitcoin treasury", "crypto treasury",
            "strategy", "microstrategy", "michael saylor",
            "tesla bitcoin", "square bitcoin", "block bitcoin",
            "metaplanet", "semler scientific",
            "treasury reserve asset", "tra",
            "bitcoin accumulation", "dollar cost averaging",
            "convertible bond", "at the market offering",
            
            # Investment Vehicles
            "crypto hedge fund", "crypto venture fund",
            "pantera capital", "a16z crypto", "paradigm",
            "multicoin capital", "polychain capital",
            "digital currency group", "dcg",
            "grayscale trust", "otc trust",
            "closed end fund", "cef",
            "interval fund", "40 act fund",
            "qualified purchaser", "accredited investor",
            
            # 401(k) & Retirement
            "401(k)", "retirement account", "ira",
            "self-directed ira", "solo 401(k)",
            "department of labor", "dol",
            "plan sponsor", "fiduciary",
            "forusall", "bitcoin 401k",
            
            # Institutional Metrics
            "assets under management", "aum",
            "inflow", "outflow", "flow",
            "cost basis", "average entry",
            "institutional adoption", "allocation",
            "endowment", "pension fund", "sovereign wealth",
        ],
        "entity_types": [
            "ETFIssuer",
            "CryptoETF",
            "CryptoCustodian",
            "DigitalAssetTreasury",
            "CryptoFund",
            "InstitutionalInvestor",
            "PlanSponsor",
            "AuthorizedParticipant",
        ],
        "relationship_types": [
            "ISSUES",
            "MANAGES",
            "CUSTODIES",
            "HOLDS",
            "INVESTS_IN",
            "PARTICIPATES_IN",
            "PROVIDES_LIQUIDITY",
        ],
    },
    
    "DEFI": {
        "keywords": [
            # General
            "decentralized finance", "defi", "open finance",
            "permissionless", "trustless", "non-custodial",
            "smart contract", "protocol", "dao",
            "tvl", "total value locked", "mcap", "market cap",
            "fdv", "fully diluted valuation",
            "apy", "apr", "yield", "return",
            "impermanent loss", "il",
            "gas fee", "transaction cost",
            
            # Lending
            "lending protocol", "borrowing", "lending",
            "aave", "compound", "morpho", "maple finance",
            "radiant", "venus", "justlend",
            "collateral", "collateral ratio", "ltv",
            "liquidation", "health factor",
            "flash loan", "uncollateralized loan",
            "overcollateralized", "undercollateralized",
            "isolated lending", "cross-collateral",
            "interest rate model", "utilization rate",
            "supply apy", "borrow apy",
            "credit market", "debt market",
            
            # DEXs
            "decentralized exchange", "dex",
            "uniswap", "pancakeswap", "sushiswap",
            "curve", "balancer", "bancor",
            "1inch", "matcha", "paraswap",
            "amm", "automated market maker",
            "constant product", "x*y=k",
            "concentrated liquidity", "v3",
            "range order", "tick",
            "liquidity pool", "lp", "lp token",
            "liquidity provider", "yield farmer",
            "slippage", "price impact",
            "constant sum", "stableswap",
            "orderbook dex", "order book",
            "limit order", "market order",
            "hybrid dex", "amm aggregator",
            
            # Derivatives
            "perpetual", "perp", "perpetual futures",
            "dydx", "gmx", "hyperliquid", "gains network",
            "funding rate", "index price", "mark price",
            "margin trading", "isolated margin", "cross margin",
            "liquidation price", "maintenance margin",
            "open interest", "oi",
            "options protocol", "opyn", "lyra", "premia",
            "structured product", "ribbon", "thetanuts",
            "delta neutral", "market neutral",
            
            # Yield
            "yield farming", "liquidity mining",
            "vault", "yearn", "convex",
            "boosted yield", "native yield",
            "restaking", "eigenlayer", "pendle",
            "liquid staking", "lido", "rocket pool",
            "staking derivative", "lst", "lsd",
            "real yield", "protocol revenue",
            "yield aggregator", "yield optimizer",
            
            # Bridges
            "bridge", "cross chain",
            "layerzero", "wormhole", "axelar",
            "multichain", "stargate", "across",
            "wrapped token", "wbtc", "weth",
            "canonical bridge", "native bridge",
            "lock and mint", "burn and mint",
            
            # Oracles
            "oracle", "chainlink", "link",
            "price feed", "data feed",
            "pyth", "api3", "band protocol",
            "oracle manipulation", "flash loan attack",
        ],
        "entity_types": [
            "DeFiProtocol",
            "DEX",
            "LendingProtocol",
            "DerivativesProtocol",
            "YieldAggregator",
            "Bridge",
            "Oracle",
            "SmartContract",
            "LiquidityPool",
            "GovernanceToken",
        ],
        "relationship_types": [
            "LENDS",
            "BORROWS",
            "PROVIDES_LIQUIDITY",
            "SWAPS",
            "STAKES",
            "GOVERNS",
            "BRIDGES_TO",
            "FEEDS_PRICE",
            "COLLATERALIZES",
        ],
    },
    
    "CRYPTO_REGULATION": {
        "keywords": [
            # US Regulators
            "securities and exchange commission", "sec",
            "commodity futures trading commission", "cftc",
            "financial crimes enforcement network", "fincen",
            "office of the comptroller of the currency", "occ",
            "federal reserve", "fed",
            "department of justice", "doj",
            "internal revenue service", "irs",
            "department of treasury", "treasury",
            "gary gensler", "rostin behnam", "hester peirce",
            "mark uyeda", "paul atkins",
            
            # International
            "financial action task force", "fatf",
            "european union", "eu", "mica",
            "markets in crypto assets",
            "esma", "eba",
            "united kingdom", "fca",
            "singapore", "mas",
            "hong kong", "sfc",
            "japan", "fsa",
            "switzerland", "finma",
            "dubai", "vara",
            
            # Key Regulations
            "securities act", "securities exchange act",
            "commodity exchange act", "cea",
            "bank secrecy act", "bsa",
            "howey test", "reves test",
            "investment company act", "40 act",
            "investment advisers act",
            "sarbanes oxley", "sox",
            "patriot act",
            "dodd frank",
            
            # 2025-2026 Specific
            "genius act", "stablecoin regulation",
            "clarity act", "financial innovation",
            "market structure legislation",
            "sab 121", "staff accounting bulletin",
            "digital asset framework",
            "fit21", "financial innovation",
            "responsible financial innovation act",
            
            # Enforcement & Compliance
            "enforcement action", "settlement",
            "no action letter", "safe harbor",
            "regulation by enforcement",
            "compliance program", "aml", "kyc",
            "know your customer", "customer identification",
            "suspicious activity report", "sar",
            "travel rule", "beneficial ownership",
            "sanctions", "ofac", "tornado cash",
            "securities fraud", "market manipulation",
            "insider trading", "wash trading",
            "unregistered securities",
            "broker dealer", "bd",
            "transfer agent",
            "clearing agency",
            "alternative trading system", "ats",
            
            # Legal Cases
            "sec v ripple", "sec v coinbase",
            "sec v binance", "sec v kraken",
            "sec v uniswap", "sec v consensys",
            "lbry", "telegram", "kik",
            "terraform labs", "do kwon",
            "ftx", "sbf", "sam bankman fried",
            "celsius", "voyager", "blockfi",
        ],
        "entity_types": [
            "RegulatoryBody",
            "Regulation",
            "LegalCase",
            "ComplianceFramework",
            "EnforcementAction",
            "Legislation",
            "Regulator",
        ],
        "relationship_types": [
            "REGULATES",
            "ENFORCES",
            "COMPLIES_WITH",
            "CHALLENGES",
            "DEFINES",
            "AMENDS",
        ],
    },
    
    "STABLECOINS": {
        "keywords": [
            # Types
            "stablecoin", "stable coin",
            "fiat backed", "fiat collateralized",
            "crypto collateralized", "over collateralized",
            "algorithmic stablecoin", "algo stable",
            "commodity backed", "gold backed",
            "synthetic dollar",
            
            # Major Stablecoins
            "tether", "usdt",
            "usd coin", "usdc", "circle",
            "binance usd", "busd",
            "dai", "makerdao", "maker",
            "paypal usd", "pyusd",
            "first digital usd", "fdusd",
            "trueusd", "tusd",
            "pax dollar", "usdp",
            "gemini dollar", "gusd",
            "liquity usd", "lusd",
            "frax", "frax usd",
            "ethena", "usde", "synthetic dollar",
            
            # Mechanics
            "peg", "depeg", "repeg",
            "collateral ratio", "cr",
            "redemption", "mint", "burn",
            "attestation", "proof of reserves",
            "reserve composition", "treasury holdings",
            "yield bearing stablecoin",
            "interest bearing",
            
            # GENIUS Act Terms
            "genius act", "payment stablecoin",
            "qualified payment stablecoin",
            "permitted payment stablecoin issuer",
            "nonbank issuer", "bank issuer",
            "1:1 reserve", "reserve requirements",
            "monthly disclosure", "quarterly audit",
            "redemption rights", "same day redemption",
            
            # Use Cases
            "remittance", "cross border payment",
            "settlement", "trading pair",
            "quote currency", "base currency",
            "collateral", "margin",
            "yield farming", "lending",
            "treasury management", "cash equivalent",
            "offshore dollar", "eurodollar",
            
            # Market Data
            "supply", "circulating supply",
            "market cap", "dominance",
            "transfer volume", "transaction count",
            "active addresses", "velocity",
            "premium", "discount", "basis",
            "curve pool", "3pool", "lusd pool",
            "liquidity fragmentation",
        ],
        "entity_types": [
            "Stablecoin",
            "StablecoinIssuer",
            "ReserveAsset",
            "PaymentCorridor",
            "RedemptionMechanism",
        ],
        "relationship_types": [
            "ISSUES",
            "BACKS",
            "COLLATERALIZES",
            "FACILITATES",
            "PEGS_TO",
            "REDEEMS",
        ],
    },
    
    "BLOCKCHAIN_INFRA": {
        "keywords": [
            # Layer 1s
            "layer 1", "l1", "base layer",
            "ethereum", "eth",
            "solana", "sol",
            "avalanche", "avax",
            "cardano", "ada",
            "polkadot", "dot",
            "cosmos", "atom",
            "near protocol", "near",
            "aptos", "sui", "monad",
            "tron", "trx",
            "bnb chain", "binance smart chain", "bsc",
            "fantom", "ftm",
            "polygon", "matic",
            "arbitrum", "optimism", "base",
            
            # Consensus
            "proof of work", "pow",
            "proof of stake", "pos",
            "delegated proof of stake", "dpos",
            "proof of authority", "poa",
            "proof of history", "poh",
            "byzantine fault tolerance", "bft",
            "tendermint", "hotstuff",
            "validator", "validator set",
            "block producer", "slot leader",
            "finality", "probabilistic finality",
            "epoch", "slot", "era",
            "attestation", "committee",
            "slashing", "jailing",
            "unbonding period", "warmup",
            
            # Scaling
            "layer 2", "l2",
            "rollup", "rollups",
            "optimistic rollup",
            "zero knowledge rollup", "zk rollup",
            "validity rollup",
            "sequencer", "proposer",
            "data availability", "da",
            "data availability sampling", "das",
            "eigenlayer", "celestia",
            "avail", "near da",
            "settlement layer", "execution layer",
            "sharding", "danksharding",
            "proto danksharding", "eip 4844", "blob",
            "state channel", "payment channel",
            "sidechain", "validium",
            "volition", "starkex",
            
            # Cross-chain
            "interoperability", "cross chain",
            "bridge", "bridging",
            "atomic swap", "htlc",
            "ibc", "inter blockchain communication",
            "polkadot xcm", "xcm",
            "layerzero", "omnichain",
            "wrapped asset", "canonical bridge",
            "native bridge", "third party bridge",
            "bridge hack", "bridge exploit",
            "lock and mint", "burn and mint",
            "liquidity network", "bridge aggregation",
            
            # MEV
            "mev", "maximal extractable value",
            "miner extractable value",
            "sandwich attack", "frontrunning",
            "backrunning", "arbitrage",
            "liquidation", "searcher",
            "builder", "proposer builder separation",
            "pbs", "mempool", "dark forest",
            "flashbots", "mev boost",
            "mev relay", "censorship",
            "priority fee", "tip",
            
            # Cryptography
            "cryptography", "cryptographic",
            "public key", "private key",
            "elliptic curve", "secp256k1",
            "eddsa", "bls signature",
            "merkle tree", "merkle root",
            "patricia tree", "trie",
            "zero knowledge proof", "zk proof",
            "zk snark", "zk stark",
            "circuit", "trusted setup",
            "homomorphic encryption",
            "threshold signature", "mpc",
            "multi party computation",
            "quantum resistant", "post quantum",
            
            # Network
            "peer to peer", "p2p",
            "gossip protocol", "libp2p",
            "node", "full node",
            "archive node", "pruned node",
            "light client", "client",
            "geth", "nethermind", "besu", "erigon",
            "consensus client", "execution client",
            "prysm", "lighthouse", "teku", "nimbus", "lodestar",
            "rpc", "json rpc", "websocket",
            "api", "endpoint",
        ],
        "entity_types": [
            "Blockchain",
            "Layer2",
            "ConsensusMechanism",
            "Validator",
            "Bridge",
            "ScalingSolution",
            "CryptographicPrimitive",
            "Node",
            "Client",
        ],
        "relationship_types": [
            "VALIDATES",
            "BUILDS_ON",
            "BRIDGES_TO",
            "SCALES",
            "USES_CONSENSUS",
            "OPERATES",
        ],
    },
    
    "CRYPTO_AI": {
        "keywords": [
            # General
            "ai", "artificial intelligence",
            "machine learning", "ml",
            "blockchain ai", "crypto ai",
            "decentralized ai", "deai",
            "agent", "ai agent",
            "autonomous agent", "onchain agent",
            
            # Projects
            "bittensor", "tao",
            "fetch.ai", "fet",
            "singularitynet", "agi",
            "ocean protocol", "ocean",
            "world coin", "world", "world app",
            "aioz", "render", "rndr",
            "golem", "glm",
            "akash", "akt",
            "near ai", "near",
            "story protocol", "ip",
            
            # Concepts
            "decentralized compute", "gpu mining",
            "distributed training", "federated learning",
            "model marketplace", "ai marketplace",
            "training data", "data marketplace",
            "inference", "model inference",
            "ai agent economy", "agent to agent",
            "proof of personhood", "pop",
            "human verification", "sybil resistance",
            "verifiable compute", "zkml",
            "zero knowledge machine learning",
            "on chain inference",
            "ai oracle", "model oracle",
            
            # Integration
            "chatgpt", "openai", "claude",
            "llm", "large language model",
            "copilot", "coding assistant",
            "smart contract ai", "ai auditing",
            "ai trading", "ai bot",
        ],
        "entity_types": [
            "AIModel",
            "AIAgent",
            "ComputeProvider",
            "DataMarketplace",
            "DecentralizedAIProtocol",
            "ProofOfPersonhood",
        ],
        "relationship_types": [
            "TRAINS",
            "INFERENCES",
            "PROVIDES_COMPUTE",
            "VERIFIES",
            "GOVERNS",
        ],
    },
    
    "TOKENIZATION": {
        "keywords": [
            # General
            "tokenization", "asset tokenization",
            "rwa", "real world assets",
            "digital asset securities",
            "security token", "tokenized security",
            "digital twin", "digital representation",
            
            # Asset Classes
            "tokenized treasury", "tokenized bond",
            "tokenized equity", "tokenized stock",
            "private credit", "tokenized credit",
            "real estate tokenization",
            "commodity tokenization", "gold token",
            "tokenized fund", "tokenized etf",
            "carbon credit", "tokenized carbon",
            "nft", "non fungible token",
            "soulbound token", "sbt",
            
            # Infrastructure
            "tokenization platform",
            "securitize", "polymath", "tokensoft",
            "centrifuge", "goldfinch",
            "ondo finance", "maple",
            "chainlink ccip", "swift integration",
            "dlt", "distributed ledger technology",
            
            # Regulation
            "reg d", "reg s", "reg a+",
            "accredited investor",
            "qualified purchaser",
            "transfer agent",
            "cap table", "on chain cap table",
            "24/7 trading", "t plus zero",
            "atomic settlement", "instant settlement",
            
            # Markets
            "addx", "tzero",
            "dxyn", "ownera",
            "private market", "illiquid asset",
            "fractional ownership",
        ],
        "entity_types": [
            "TokenizedAsset",
            "TokenizationPlatform",
            "SecurityToken",
            "RWACategory",
            "DigitalSecurity",
        ],
        "relationship_types": [
            "TOKENIZES",
            "REPRESENTS",
            "TRADES_ON",
            "CUSTODIES",
            "FRACTIONALIZES",
        ],
    },
}
```

### 3.3 Phase 3: Entity Relationship Taxonomy

#### 3.3.1 Crypto-Specific Edge Types

**File**: Update in `persistence/v1/schema.py` or extraction configurations:

```python
# Hierarchical Relationships
"LAYER_OF",           # L2 builds on L1
"PART_OF_ECOSYSTEM",  # Protocol part of DeFi ecosystem
"FORK_OF",           # One chain/protocol forks from another

# Financial Relationships
"BANKS_WITH",        # Institution uses specific custody
"ISSUED_BY",         # ETF issued by issuer
"BACKED_BY",         # Stablecoin backed by reserves
"COLLATERALIZED_BY", # Loan collateralized by asset
"GENERATES_YIELD",   # Strategy generates yield
"PAYS_INTEREST",     # Protocol pays interest

# Market Relationships
"TRADES_ON",         # Asset trades on exchange
"PAIRED_WITH",       # Trading pair relationship
"LISTED_ON",         # ETF listed on exchange
"FLOWS_INTO",        # Capital flows into product

# Regulatory Relationships
"REGULATED_BY",      # Entity regulated by body
"COMPLIANT_WITH",    # Entity compliant with regulation
"LICENSED_IN",       # Entity licensed in jurisdiction
"SUBJECT_TO",        # Entity subject to enforcement

# Technical Relationships
"UPGRADED_TO",       # Protocol upgrade
"DEPLOYS_ON",        # Contract deploys on chain
"BRIDGES_BETWEEN",   # Bridge connects chains
"VALIDATES_FOR",     # Validator validates chain
"PROVIDES_DATA_TO",  # Oracle provides data

# Temporal Relationships (keep existing + add)
"HALVED_ON",         # Bitcoin halving date
"FORKED_ON",         # Chain fork date
"LAUNCHED_ON",       # Product/protocol launch
"EXPIRES_ON",        # Option/future expiration
```

### 3.4 Phase 4: Extraction Prompts Configuration

#### 3.4.1 Bitcoin-Specific Extraction Prompt

**File**: Create new extraction templates in `extraction/template_registry.py`:

```python
BITCOIN_EXTRACTION_PROMPT = """
You are a Bitcoin knowledge extraction specialist. Extract entities and relationships 
from cryptocurrency content with focus on:

**Entity Types to Extract:**
1. Bitcoin (the asset itself)
2. BitcoinUpgrades (BIPs, soft forks, hard forks)
3. MiningPools (Foundry USA, Antpool, F2Pool, etc.)
4. BitcoinETFs (IBIT, GBTC, FBTC, ARKB, etc.)
5. DigitalAssetTreasuries (Strategy, Metaplanet, etc.)
6. NationStates (El Salvador, Bhutan, etc. adopting Bitcoin)
7. BitcoinAnalysts (Michael Saylor, etc.)
8. BitcoinDevelopers (Core devs, BIP authors)

**Key Relationships:**
- MINES: Mining pool mines Bitcoin
- HOLDS: Entity holds Bitcoin (quantity if specified)
- ISSUES: ETF issuer issues ETF
- OPERATES: Entity operates mining pool/node
- ADOPTED_BY: Nation state adopts Bitcoin
- UPGRADED_TO: Protocol upgrade

**Special Instructions:**
- Extract specific Bitcoin quantities (e.g., "holds 100,000 BTC")
- Capture ETF flow data (inflows/outflows)
- Identify halving dates and impacts
- Extract mining difficulty and hash rate metrics
- Capture on-chain metrics (MVRV, NUPL, HODL waves)

Output format: JSON with entities and relationships arrays.
"""

INSTITUTIONAL_EXTRACTION_PROMPT = """
Extract institutional cryptocurrency adoption entities:

**Entity Types:**
1. ETFIssuer (BlackRock, Grayscale, Fidelity, etc.)
2. CryptoETF (IBIT, GBTC, spot ETF products)
3. CryptoCustodian (Coinbase, BitGo, Fidelity, etc.)
4. InstitutionalInvestor (pension funds, endowments, SWFs)
5. Wirehouse (Morgan Stanley, Merrill Lynch, etc.)
6. PlanSponsor (companies offering 401k)

**Key Data Points:**
- AUM (Assets Under Management)
- ETF flows (daily/weekly inflows/outflows)
- Expense ratios and fees
- Custody arrangements
- Launch dates and approvals

**Relationships:**
- ISSUES: Issuer issues ETF
- CUSTODIES: Custodian holds assets
- ALLOCATES_TO: Institution allocates to crypto
- ENABLES: Platform enables advisor recommendations
"""

DEFI_EXTRACTION_PROMPT = """
Extract DeFi protocol entities and TVL data:

**Entity Types:**
1. DeFiProtocol (Aave, Compound, Uniswap, etc.)
2. LiquidityPool (specific pools with APY)
3. GovernanceToken (UNI, AAVE, COMP, etc.)
4. YieldStrategy (specific vaults/strategies)
5. SmartContract (contract addresses if mentioned)

**Metrics to Extract:**
- TVL (Total Value Locked)
- APY/APR rates
- Token prices
- Market caps / FDV
- Trading volumes

**Relationships:**
- PROVIDES_LIQUIDITY: User/entity provides liquidity
- STAKES_IN: Token staked in protocol
- GOVERNS: Token holder governs protocol
- YIELDS: Strategy generates yield
"""
```

### 3.5 Phase 5: Data Model Updates

#### 3.5.1 Enhanced Entity Properties

**File**: `src/knowledge_base/persistence/v1/schema.py`

Update the `Entity` model to include crypto-specific properties:

```python
class Entity(Base):
    """Knowledge graph entity with crypto-specific properties."""
    
    __tablename__ = "entities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    entity_type = Column(String, nullable=False, index=True)
    
    # Enhanced properties for crypto entities
    properties = Column(JSONB, default={})
    
    # Crypto-specific indexed fields
    symbol = Column(String, nullable=True, index=True)  # BTC, ETH, etc.
    chain = Column(String, nullable=True, index=True)   # bitcoin, ethereum, etc.
    contract_address = Column(String, nullable=True, index=True)
    
    # Temporal tracking
    first_seen_at = Column(DateTime(timezone=True), nullable=True)
    last_updated_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Financial metrics (for relevant entities)
    market_cap = Column(Numeric, nullable=True)
    price_usd = Column(Numeric, nullable=True)
    tvl = Column(Numeric, nullable=True)  # Total Value Locked
    
    # Vector embedding for semantic search
    embedding = Column(Vector(1024), nullable=True)
    
    # Relationships
    chunks = relationship("ChunkEntity", back_populates="entity")
    source_edges = relationship("Edge", foreign_keys="Edge.source_id", back_populates="source")
    target_edges = relationship("Edge", foreign_keys="Edge.target_id", back_populates="target")

# Properties JSON structure for different entity types:
ENTITY_PROPERTY_SCHEMAS = {
    "Cryptocurrency": {
        "symbol": "string",
        "circulating_supply": "number",
        "max_supply": "number",
        "consensus_mechanism": "string",
        "launch_date": "date",
        "whitepaper_url": "string",
        "github_url": "string",
        "website": "string",
        "market_data": {
            "price": "number",
            "market_cap": "number",
            "volume_24h": "number",
            "price_change_24h": "number"
        }
    },
    "BitcoinETF": {
        "ticker": "string",
        "issuer": "string",
        "etf_type": "spot|futures",
        "expense_ratio": "number",
        "aum": "number",
        "bitcoin_holdings": "number",
        "launch_date": "date",
        "exchange": "string",
        "inception_nav": "number",
        "current_nav": "number",
        "premium_discount": "number",
        "creation_redemption": "in_kind|cash"
    },
    "DeFiProtocol": {
        "protocol_type": "lending|dex|yield|derivative|bridge",
        "chain": "string",
        "tvl": "number",
        "mcap": "number",
        "fdv": "number",
        "revenue_24h": "number",
        "revenue_30d": "number",
        "token_symbol": "string",
        "governance_token": "string",
        "launch_date": "date",
        "audit_status": "string",
        "bug_bounty": "boolean"
    },
    "MiningPool": {
        "hash_rate": "number",
        "hash_rate_percentage": "number",
        "blocks_mined_24h": "number",
        "reward_method": "string",
        "location": "string",
        "foundry_usa": "boolean",
        "antpool": "boolean",
    },
    "Stablecoin": {
        "ticker": "string",
        "issuer": "string",
        "backing_type": "fiat|crypto|commodity|algorithmic",
        "collateral_ratio": "number",
        "market_cap": "number",
        "circulating_supply": "number",
        "reserves_composition": "object",
        "attestation_url": "string",
        "redemption_mechanism": "string",
        "genius_act_compliant": "boolean"
    }
}
```

### 3.6 Phase 6: API Enhancements

#### 3.6.1 Crypto-Specific Query Endpoints

**File**: Enhance `query_api.py` with crypto-specific endpoints:

```python
@router.post("/crypto/market-intelligence")
async def crypto_market_intelligence(
    query: MarketIntelligenceQuery,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Query for cryptocurrency market intelligence combining:
    - ETF flows and institutional movements
    - On-chain metrics
    - Market sentiment
    - Protocol TVL and fundamentals
    """
    pass

@router.post("/crypto/onchain-analysis")
async def onchain_analysis(
    metric: OnChainMetric,
    timeframe: TimeFrame,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Retrieve on-chain metrics for Bitcoin and other assets:
    - MVRV ratio
    - NUPL (Net Unrealized Profit/Loss)
    - HODL waves
    - Exchange flows
    - Realized price
    """
    pass

@router.get("/crypto/etf/flows")
async def etf_flows(
    etf_ticker: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get ETF flow data for specified ETF or all ETFs.
    Supports IBIT, GBTC, FBTC, ARKB, etc.
    """
    pass

@router.get("/crypto/treasuries/holdings")
async def treasury_holdings(
    company: Optional[str] = None,
    min_btc: Optional[float] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Query Digital Asset Treasury (DAT) holdings.
    Filter by company or minimum BTC holdings.
    """
    pass

@router.post("/crypto/defi/tvl")
async def defi_tvl(
    protocol: Optional[str] = None,
    chain: Optional[str] = None,
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Query DeFi TVL data.
    Aggregate by protocol, chain, or category.
    """
    pass
```

### 3.7 Phase 7: Search Optimization

#### 3.7.1 Crypto-Specific BM25 Indexing

**File**: Update `storage/bm25_index.py` with crypto stopwords and boosts:

```python
CRYPTO_STOPWORDS = {
    # Common crypto words that shouldn't be indexed
    "blockchain", "cryptocurrency", "crypto", "digital", "token",
    "decentralized", "network", "protocol",
}

CRYPTO_TERM_BOOSTS = {
    # Boost specific crypto terms for relevance
    "bitcoin": 2.0,
    "btc": 2.0,
    "ethereum": 1.8,
    "eth": 1.8,
    "etf": 1.5,
    "halving": 1.5,
    "mining": 1.3,
    "defi": 1.4,
    "tvl": 1.3,
    "stablecoin": 1.3,
    "smart contract": 1.4,
}

CRYPTO_SYNONYMS = {
    # Synonym expansion for better recall
    "btc": ["bitcoin", "satoshi"],
    "eth": ["ethereum", "ether"],
    "ln": ["lightning network", "lightning"],
    "pos": ["proof of stake"],
    "pow": ["proof of work"],
    "dat": ["digital asset treasury", "bitcoin treasury"],
    "lst": ["liquid staking token", "staking derivative"],
    "amm": ["automated market maker"],
    "dex": ["decentralized exchange"],
}
```

### 3.8 Phase 8: Ingestion Pipeline Customization

#### 3.8.1 Crypto Document Classifier

**File**: Enhance `domain/detection.py`:

```python
class CryptoDomainDetector:
    """Specialized detector for cryptocurrency content domains."""
    
    def __init__(self):
        self.bitcoin_keywords = set(BITCOIN_KEYWORDS)
        self.etf_keywords = set(ETF_KEYWORDS)
        self.defi_keywords = set(DEFI_KEYWORDS)
        self.regulatory_keywords = set(REGULATORY_KEYWORDS)
        self.stablecoin_keywords = set(STABLECOIN_KEYWORDS)
        
    async def detect_crypto_domain(self, text: str) -> DomainDetectionResult:
        """
        Specialized detection for crypto sub-domains.
        Returns specific domain (BITCOIN, DEFI, etc.) vs generic CRYPTO.
        """
        scores = {
            Domain.BITCOIN: self._score_bitcoin(text),
            Domain.INSTITUTIONAL_CRYPTO: self._score_institutional(text),
            Domain.DEFI: self._score_defi(text),
            Domain.CRYPTO_REGULATION: self._score_regulation(text),
            Domain.STABLECOINS: self._score_stablecoins(text),
        }
        
        # Return highest scoring domain above threshold
        max_domain = max(scores, key=scores.get)
        if scores[max_domain] > 0.6:
            return DomainDetectionResult(
                primary_domain=max_domain.value,
                confidence=scores[max_domain]
            )
        
        return DomainDetectionResult(
            primary_domain=Domain.GENERAL.value,
            confidence=1.0 - max(scores.values())
        )
```

---

## Part 4: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Tasks:**
1. Update domain models (`domain_models.py`)
   - Replace generic domains with crypto-specific domains
   - Add Domain enum values for BITCOIN, DEFI, etc.

2. Expand ontology snippets (`ontology_snippets.py`)
   - Add comprehensive keyword lists for each domain
   - Define entity types and relationship types

3. Update database schema
   - Add crypto-specific columns to Entity table
   - Create indexes for symbol, chain, contract_address
   - Add JSONB property schemas

**Verification:**
```bash
# Run domain detection tests
uv run pytest tests/unit/domain/ -v

# Verify ontology loading
uv run python -c "from knowledge_base.domain.ontology_snippets import DOMAIN_ONTOLOGIES; print('Domains:', list(DOMAIN_ONTOLOGIES.keys()))"
```

### Phase 2: Extraction Layer (Weeks 3-4)

**Tasks:**
1. Create crypto-specific extraction prompts
2. Update template registry with Bitcoin, DeFi, Institutional prompts
3. Enhance entity extraction for crypto metrics (TVL, APY, etc.)
4. Add relationship extraction for crypto-specific edges

**Verification:**
```bash
# Test extraction on sample crypto documents
uv run python scripts/test_crypto_extraction.py
```

### Phase 3: Search & Retrieval (Weeks 5-6)

**Tasks:**
1. Update BM25 index with crypto boosts and synonyms
2. Configure cross-encoder for crypto domain
3. Implement crypto-specific query endpoints
4. Add on-chain metric aggregation

**Verification:**
```bash
# Run search benchmarks
uv run pytest tests/integration/test_crypto_search.py -v
```

### Phase 4: Data Ingestion (Weeks 7-8)

**Tasks:**
1. Build crypto document processors
2. Implement real-time ETF flow ingestion
3. Add on-chain data connectors
4. Create DeFi protocol data sync

**Verification:**
```bash
# Test ingestion pipeline
uv run python scripts/test_ingestion.py --domain BITCOIN
```

### Phase 5: API & Frontend (Weeks 9-10)

**Tasks:**
1. Implement crypto-specific API endpoints
2. Create market intelligence dashboard
3. Build on-chain metrics visualization
4. Add ETF flow tracking interface

**Verification:**
```bash
# Run API tests
uv run pytest tests/e2e/test_crypto_api.py -v
```

### Phase 6: Documentation & Deployment (Weeks 11-12)

**Tasks:**
1. Write comprehensive documentation
2. Create crypto knowledgebase examples
3. Build deployment scripts
4. Set up monitoring for crypto data freshness

---

## Part 5: Data Sources Integration

### 5.1 Real-Time Data Feeds

| Data Type | Source | Frequency | Endpoint |
|-----------|--------|-----------|----------|
| ETF Flows | Bloomberg, ETF.com | Daily | `/api/v1/crypto/etf/flows` |
| On-Chain | Amberdata, Glassnode | Hourly | `/api/v1/crypto/onchain` |
| DeFi TVL | DeFiLlama | Hourly | `/api/v1/crypto/defi/tvl` |
| Prices | CoinGecko, CoinMarketCap | 5 min | `/api/v1/crypto/prices` |
| News | CryptoPanic, TheBlock | Real-time | WebSocket feed |
| Regulations | SEC, CFTC filings | Daily | RSS/Web scraping |

### 5.2 Document Sources

1. **Research Reports**
   - Grayscale Research
   - Pantera Capital Blockchain Letter
   - Coinbase Institutional Research
   - Amberdata Research
   - Bitwise Research

2. **Protocol Documentation**
   - Bitcoin Improvement Proposals (BIPs)
   - Ethereum Improvement Proposals (EIPs)
   - DeFi protocol docs (Aave, Uniswap, etc.)
   - L2 documentation (Arbitrum, Optimism, Base)

3. **Regulatory Documents**
   - SEC filings (10-K, 10-Q for crypto companies)
   - CFTC regulations
   - Congressional bills (GENIUS Act, etc.)
   - International regulations (MiCA, etc.)

4. **Institutional Communications**
   - ETF prospectuses
   - Custody agreements
   - Treasury disclosures (Strategy, etc.)
   - Investor letters

---

## Part 6: Testing Strategy

### 6.1 Unit Tests

```python
# Test domain detection
def test_bitcoin_domain_detection():
    text = "Bitcoin's 2024 halving reduced block rewards to 3.125 BTC"
    result = detector.detect(text)
    assert result.primary_domain == "BITCOIN"
    assert result.confidence > 0.8

# Test entity extraction
def test_etf_entity_extraction():
    text = "BlackRock's IBIT holds 500,000 BTC with $45B AUM"
    entities = extractor.extract(text)
    assert any(e.entity_type == "ETFIssuer" and e.name == "BlackRock" for e in entities)
    assert any(e.entity_type == "BitcoinETF" and e.name == "IBIT" for e in entities)
```

### 6.2 Integration Tests

```python
# Test full ingestion pipeline
def test_bitcoin_document_ingestion():
    doc = Document(content="Bitcoin mining difficulty reached all-time high...")
    result = await pipeline.ingest(doc)
    assert result.domain == "BITCOIN"
    assert len(result.entities) > 5
    assert any(e.entity_type == "MiningPool" for e in result.entities)
```

### 6.3 E2E Tests

```python
# Test query API
def test_etf_flow_query():
    response = client.get("/api/v1/crypto/etf/flows?etf=IBIT")
    assert response.status_code == 200
    assert "inflows" in response.json()
```

---

## Part 7: Monitoring & Maintenance

### 7.1 Data Freshness

- ETF flows: Update daily after market close
- On-chain metrics: Update hourly
- DeFi TVL: Update every 6 hours
- Prices: Real-time via WebSocket

### 7.2 Quality Metrics

- Entity extraction accuracy > 90%
- Relationship extraction precision > 85%
- Search relevance (NDCG@10) > 0.8
- Query latency P95 < 500ms

### 7.3 Alerts

- New ETF approval announcements
- Regulatory filing alerts
- Unusual on-chain movements
- Protocol exploit/hack notifications
- Halving countdown milestones

---

## Part 8: Conclusion

This refactoring plan transforms KBv2 into a specialized cryptocurrency and Bitcoin knowledgebase optimized for the 2026 institutional adoption era. Key achievements:

1. **Comprehensive Domain Coverage**: 10 crypto-specific domains covering all major aspects of the industry
2. **2026-Ready**: Incorporates latest trends (ETF flows, institutional adoption, GENIUS Act)
3. **Rich Entity Taxonomy**: 50+ entity types with structured property schemas
4. **Optimized Search**: Crypto-specific BM25 boosts, synonyms, and ranking
5. **Real-Time Data**: Integration points for live market data, on-chain metrics, and regulatory feeds
6. **Institutional Focus**: Special emphasis on ETFs, custody, treasuries, and compliance

The refactored system will serve as a definitive knowledge management platform for cryptocurrency research, institutional due diligence, regulatory compliance, and market intelligence.

---

## Appendices

### Appendix A: Entity Type Reference

[Full entity type definitions with properties]

### Appendix B: API Endpoint Reference

[Complete API specification]

### Appendix C: Data Source Configuration

[Detailed integration guides for each data source]

### Appendix D: Ontology Keyword Lists

[Complete keyword lists for all domains]

---

*Document Version: 1.0*
*Last Updated: 2026-02-06*
*Author: KBv2 Refactoring Team*
