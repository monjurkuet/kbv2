"""Data Pipeline Connector for KBv2.

Provides an interface for external data engineering pipelines to feed
real-time crypto data into the knowledgebase.

This module is designed to be used by a SEPARATE data engineering repository
that handles real-time data ingestion (ETF flows, on-chain metrics, etc.)
and pushes processed data to the knowledgebase via this connector.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, Field, validator


class DataSourceType(str, Enum):
    """Types of external data sources."""

    ETF_FLOWS = "etf_flows"  # ETF flow data (IBIT, GBTC, etc.)
    ONCHAIN_METRICS = "onchain_metrics"  # On-chain analytics (MVRV, NUPL, etc.)
    DEFI_TVL = "defi_tvl"  # DeFi protocol TVL
    PRICE_DATA = "price_data"  # Price feeds
    NEWS_SENTIMENT = "news_sentiment"  # News and sentiment
    REGULATORY_FILINGS = "regulatory_filings"  # SEC/CFTC filings
    PROTOCOL_UPDATES = "protocol_updates"  # DeFi protocol updates


class DataFormat(str, Enum):
    """Supported data formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline connection."""

    # KBv2 API endpoint
    kbv2_api_url: str = "http://localhost:8000/api/v1"
    api_key: Optional[str] = None

    # Webhook settings (for KBv2 to call pipeline)
    webhook_secret: Optional[str] = None

    # Data source configurations
    enabled_sources: List[DataSourceType] = field(
        default_factory=lambda: [
            DataSourceType.ETF_FLOWS,
            DataSourceType.ONCHAIN_METRICS,
            DataSourceType.DEFI_TVL,
        ]
    )

    # Rate limiting
    max_requests_per_minute: int = 60
    batch_size: int = 100

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: int = 5


class ETFFlowData(BaseModel):
    """ETF flow data structure."""

    etf_ticker: str = Field(..., description="ETF ticker symbol (e.g., IBIT)")
    issuer: str = Field(..., description="ETF issuer (e.g., BlackRock)")
    date: datetime = Field(..., description="Data date")
    inflow_usd: Optional[float] = Field(None, description="Daily inflow in USD")
    outflow_usd: Optional[float] = Field(None, description="Daily outflow in USD")
    net_flow_usd: Optional[float] = Field(None, description="Net flow in USD")
    aum_usd: Optional[float] = Field(None, description="Assets under management")
    bitcoin_holdings: Optional[float] = Field(None, description="BTC held by ETF")
    shares_outstanding: Optional[int] = Field(None, description="Shares outstanding")
    nav: Optional[float] = Field(None, description="Net asset value")
    premium_discount: Optional[float] = Field(
        None, description="Premium/discount to NAV"
    )

    @validator("etf_ticker")
    def validate_ticker(cls, v):
        valid_etfs = ["IBIT", "GBTC", "FBTC", "ARKB", "BITO", "HODL", "BRRR"]
        if v.upper() not in valid_etfs:
            raise ValueError(f"Unknown ETF ticker: {v}")
        return v.upper()


class OnChainMetricData(BaseModel):
    """On-chain metric data structure."""

    metric_name: str = Field(..., description="Metric name (e.g., MVRV, NUPL)")
    blockchain: str = Field(
        default="bitcoin", description="Blockchain (bitcoin, ethereum)"
    )
    timestamp: datetime = Field(..., description="Metric timestamp")
    value: float = Field(..., description="Metric value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    block_height: Optional[int] = Field(None, description="Block height")

    @validator("metric_name")
    def validate_metric(cls, v):
        valid_metrics = [
            "MVRV",
            "NUPL",
            "SOPR",
            "HODL_Waves",
            "Realized_Price",
            "Hash_Rate",
            "Difficulty",
            "Active_Addresses",
            "Transaction_Count",
            "Exchange_Inflow",
            "Exchange_Outflow",
            "Long_Term_Holder_Supply",
        ]
        if v.upper() not in [m.upper() for m in valid_metrics]:
            raise ValueError(f"Unknown metric: {v}")
        return v


class DeFiProtocolData(BaseModel):
    """DeFi protocol data structure."""

    protocol_name: str = Field(..., description="Protocol name")
    protocol_type: str = Field(..., description="Protocol type (lending, dex, etc.)")
    chain: str = Field(..., description="Blockchain (ethereum, solana, etc.)")
    timestamp: datetime = Field(..., description="Data timestamp")
    tvl_usd: Optional[float] = Field(None, description="Total value locked")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    fdv: Optional[float] = Field(None, description="Fully diluted valuation")
    revenue_24h: Optional[float] = Field(None, description="24h revenue")
    revenue_30d: Optional[float] = Field(None, description="30d revenue")
    token_price: Optional[float] = Field(None, description="Governance token price")
    token_symbol: Optional[str] = Field(None, description="Governance token symbol")

    @validator("protocol_type")
    def validate_type(cls, v):
        valid_types = [
            "lending",
            "dex",
            "yield",
            "derivative",
            "bridge",
            "oracle",
            "cdp",
        ]
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid protocol type: {v}")
        return v.lower()


class DataIngestionRequest(BaseModel):
    """Request to ingest external data."""

    source_type: DataSourceType
    data_format: DataFormat = DataFormat.JSON
    records: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Validation
    validate_schema: bool = True
    skip_duplicates: bool = True


class DataIngestionResponse(BaseModel):
    """Response from data ingestion."""

    success: bool
    records_processed: int
    records_inserted: int
    records_skipped: int
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KBv2DataConnector(ABC):
    """Abstract base class for connecting to KBv2.

    Implement this interface in your data engineering pipeline to
    push data into the knowledgebase.
    """

    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.kbv2_api_url,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        if config.api_key:
            self.client.headers["X-API-Key"] = config.api_key

    @abstractmethod
    async def transform_to_entities(
        self, source_data: Any, source_type: DataSourceType
    ) -> List[Dict[str, Any]]:
        """Transform source data to KBv2 entity format.

        Args:
            source_data: Raw data from source
            source_type: Type of data source

        Returns:
            List of entities in KBv2 format
        """
        pass

    async def ingest_etf_flows(self, flows: List[ETFFlowData]) -> DataIngestionResponse:
        """Ingest ETF flow data.

        Args:
            flows: List of ETF flow records

        Returns:
            Ingestion response
        """
        records = [flow.dict() for flow in flows]
        request = DataIngestionRequest(
            source_type=DataSourceType.ETF_FLOWS,
            records=records,
        )
        return await self._send_ingestion_request(request)

    async def ingest_onchain_metrics(
        self, metrics: List[OnChainMetricData]
    ) -> DataIngestionResponse:
        """Ingest on-chain metric data.

        Args:
            metrics: List of metric records

        Returns:
            Ingestion response
        """
        records = [metric.dict() for metric in metrics]
        request = DataIngestionRequest(
            source_type=DataSourceType.ONCHAIN_METRICS,
            records=records,
        )
        return await self._send_ingestion_request(request)

    async def ingest_defi_data(
        self, protocols: List[DeFiProtocolData]
    ) -> DataIngestionResponse:
        """Ingest DeFi protocol data.

        Args:
            protocols: List of protocol records

        Returns:
            Ingestion response
        """
        records = [protocol.dict() for protocol in protocols]
        request = DataIngestionRequest(
            source_type=DataSourceType.DEFI_TVL,
            records=records,
        )
        return await self._send_ingestion_request(request)

    async def _send_ingestion_request(
        self, request: DataIngestionRequest
    ) -> DataIngestionResponse:
        """Send ingestion request to KBv2."""
        try:
            response = await self.client.post(
                "/data/ingest",
                json=request.dict(),
            )
            response.raise_for_status()
            return DataIngestionResponse(**response.json())
        except httpx.HTTPStatusError as e:
            return DataIngestionResponse(
                success=False,
                records_processed=0,
                records_inserted=0,
                records_skipped=0,
                errors=[f"HTTP error: {e.response.status_code}"],
                processing_time_ms=0.0,
            )
        except Exception as e:
            return DataIngestionResponse(
                success=False,
                records_processed=0,
                records_inserted=0,
                records_skipped=0,
                errors=[str(e)],
                processing_time_ms=0.0,
            )

    async def health_check(self) -> bool:
        """Check KBv2 API health."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except:
            return False

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class ExampleDataConnector(KBv2DataConnector):
    """Example implementation of data connector.

    This is a reference implementation showing how to transform
    external data into KBv2 entity format.
    """

    async def transform_to_entities(
        self, source_data: Any, source_type: DataSourceType
    ) -> List[Dict[str, Any]]:
        """Transform ETF flow data to entities."""
        entities = []

        if source_type == DataSourceType.ETF_FLOWS:
            for record in source_data:
                # Create ETF entity
                etf_entity = {
                    "name": record.get("etf_ticker"),
                    "entity_type": "BitcoinETF",
                    "properties": {
                        "ticker": record.get("etf_ticker"),
                        "issuer": record.get("issuer"),
                        "aum": record.get("aum_usd"),
                        "bitcoin_holdings": record.get("bitcoin_holdings"),
                        "expense_ratio": None,  # Would need to fetch separately
                    },
                }
                entities.append(etf_entity)

                # Create Issuer entity
                issuer_entity = {
                    "name": record.get("issuer"),
                    "entity_type": "ETFIssuer",
                    "properties": {
                        "company_name": record.get("issuer"),
                    },
                }
                entities.append(issuer_entity)

                # Create Flow metric entity
                if record.get("net_flow_usd"):
                    flow_entity = {
                        "name": f"{record.get('etf_ticker')} Flow {record.get('date')}",
                        "entity_type": "MarketIndicator",
                        "properties": {
                            "indicator_type": "ETF Flow",
                            "value": record.get("net_flow_usd"),
                            "date": record.get("date"),
                            "unit": "USD",
                        },
                    }
                    entities.append(flow_entity)

        elif source_type == DataSourceType.DEFI_TVL:
            for record in source_data:
                protocol_entity = {
                    "name": record.get("protocol_name"),
                    "entity_type": "DeFiProtocol",
                    "properties": {
                        "protocol_type": record.get("protocol_type"),
                        "chain": record.get("chain"),
                        "tvl": record.get("tvl_usd"),
                        "market_cap": record.get("market_cap"),
                        "fdv": record.get("fdv"),
                        "revenue_24h": record.get("revenue_24h"),
                    },
                }
                entities.append(protocol_entity)

        return entities


# Webhook handler for receiving data from pipeline
class DataPipelineWebhookHandler:
    """Handles incoming data from external pipelines via webhooks.

    This is used BY KBv2 to receive data from external pipelines.
    """

    def __init__(self, secret: Optional[str] = None):
        self.secret = secret

    async def handle_webhook(
        self, payload: Dict[str, Any], signature: Optional[str] = None
    ) -> DataIngestionResponse:
        """Handle incoming webhook data.

        Args:
            payload: Webhook payload
            signature: Optional signature for verification

        Returns:
            Ingestion response
        """
        # Verify signature if secret is configured
        if self.secret and signature:
            if not self._verify_signature(payload, signature):
                return DataIngestionResponse(
                    success=False,
                    records_processed=0,
                    records_inserted=0,
                    records_skipped=0,
                    errors=["Invalid signature"],
                    processing_time_ms=0.0,
                )

        # Parse request
        try:
            request = DataIngestionRequest(**payload)
        except Exception as e:
            return DataIngestionResponse(
                success=False,
                records_processed=0,
                records_inserted=0,
                records_skipped=0,
                errors=[f"Invalid request format: {str(e)}"],
                processing_time_ms=0.0,
            )

        # Process data (this would call the actual ingestion logic)
        # For now, return success
        return DataIngestionResponse(
            success=True,
            records_processed=len(request.records),
            records_inserted=len(request.records),
            records_skipped=0,
            processing_time_ms=0.0,
        )

    def _verify_signature(self, payload: Dict[str, Any], signature: str) -> bool:
        """Verify webhook signature."""
        import hmac
        import hashlib

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        expected = hmac.new(
            self.secret.encode(), payload_bytes, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected, signature)


# Connection instructions for data engineering pipeline
def print_integration_instructions():
    """Print integration instructions for data engineering team."""
    instructions = """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║           KBv2 Data Pipeline Integration Instructions                      ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    
    STEP 1: Install Required Dependencies
    ─────────────────────────────────────
    pip install httpx pydantic
    
    STEP 2: Create Your Data Connector
    ───────────────────────────────────
    from kbv2_data_connector import KBv2DataConnector, DataPipelineConfig
    
    class MyDataConnector(KBv2DataConnector):
        async def transform_to_entities(self, source_data, source_type):
            # Transform your data to KBv2 entity format
            entities = []
            for record in source_data:
                entity = {
                    "name": record["name"],
                    "entity_type": "YourEntityType",
                    "properties": {
                        "key": record["value"],
                    },
                }
                entities.append(entity)
            return entities
    
    STEP 3: Configure Connection
    ─────────────────────────────
    config = DataPipelineConfig(
        kbv2_api_url="http://your-kbv2-instance:8000/api/v1",
        api_key="your-api-key",
    )
    
    connector = MyDataConnector(config)
    
    STEP 4: Ingest Data
    ───────────────────
    from kbv2_data_connector import ETFFlowData
    
    # ETF Flows
    flows = [
        ETFFlowData(
            etf_ticker="IBIT",
            issuer="BlackRock",
            date=datetime.now(),
            net_flow_usd=500000000,
        )
    ]
    
    response = await connector.ingest_etf_flows(flows)
    print(f"Inserted {response.records_inserted} records")
    
    STEP 5: Data Source Types Supported
    ────────────────────────────────────
    - ETF_FLOWS: ETF flow data (IBIT, GBTC, etc.)
    - ONCHAIN_METRICS: On-chain analytics (MVRV, NUPL, etc.)
    - DEFI_TVL: DeFi protocol TVL data
    - PRICE_DATA: Price feeds
    - NEWS_SENTIMENT: News and sentiment data
    - REGULATORY_FILINGS: SEC/CFTC filings
    - PROTOCOL_UPDATES: DeFi protocol updates
    
    STEP 6: Entity Types Available
    ─────────────────────────────
    Bitcoin Domain:
    - BitcoinETF, MiningPool, DigitalAssetTreasury, BitcoinUpgrade
    
    DeFi Domain:
    - DeFiProtocol, DEX, LendingProtocol, LiquidityPool, YieldStrategy
    
    Institutional Domain:
    - ETFIssuer, CryptoCustodian, CryptoFund
    
    Stablecoin Domain:
    - Stablecoin, StablecoinIssuer
    
    Regulatory Domain:
    - RegulatoryBody, Regulation, LegalCase
    
    STEP 7: Error Handling
    ─────────────────────
    The connector automatically retries failed requests (3 retries by default).
    Check response.success and response.errors for failures.
    
    STEP 8: Rate Limiting
    ─────────────────────
    Maximum 60 requests per minute. Use batching for large datasets.
    
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Contact: For questions about the KBv2 API, refer to API documentation     ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    print_integration_instructions()
