"""Domain detection endpoints for KBV2."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from knowledge_base.routes.dependencies import get_dependencies


# Create router
router = APIRouter(tags=["domain"])


# Request/Response models
class DomainDetectionRequest(BaseModel):
    """Domain detection request."""

    text: str
    top_k: int = 3


class DomainDetectionResponse(BaseModel):
    """Domain detection response."""

    primary_domain: str
    all_domains: list
    is_multi_domain: bool
    detection_method: str
    processing_time_ms: float


@router.post("/detect-domain", response_model=dict)
async def detect_domain(request: DomainDetectionRequest):
    """Detect the domain of the given text."""
    deps = get_dependencies()
    if not deps.domain_detector:
        raise HTTPException(status_code=503, detail="Domain detector not initialized")

    result = await deps.domain_detector.detect_domain(request.text, top_k=request.top_k)

    return {
        "primary_domain": result.primary_domain,
        "all_domains": [
            {
                "domain": p.domain,
                "confidence": p.confidence,
                "reasoning": p.reasoning,
            }
            for p in result.all_domains
        ],
        "is_multi_domain": result.is_multi_domain,
        "detection_method": result.detection_method,
        "processing_time_ms": result.processing_time_ms,
    }
