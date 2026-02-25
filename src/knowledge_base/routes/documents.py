"""Document endpoints for KBV2."""

from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from knowledge_base.routes.dependencies import get_dependencies
from knowledge_base.storage.portable.sqlite_store import Document


# Create router
router = APIRouter(tags=["documents"])


# Request/Response models
class DocumentCreate(BaseModel):
    """Document creation request."""

    name: str
    source_uri: Optional[str] = None
    content: Optional[str] = None
    domain: Optional[str] = None
    metadata: dict = {}


class DocumentResponse(BaseModel):
    """Document response."""

    id: str
    name: str
    status: str
    domain: Optional[str] = None
    created_at: str


@router.post("/documents", response_model=dict)
async def create_document(doc: DocumentCreate):
    """Create a new document."""
    deps = get_dependencies()
    if not deps.sqlite:
        raise RuntimeError("Storage not initialized")

    document = Document(
        name=doc.name,
        source_uri=doc.source_uri,
        content=doc.content,
        domain=doc.domain,
        metadata=doc.metadata,
    )

    doc_id = await deps.sqlite.add_document(document)

    return {
        "id": doc_id,
        "name": doc.name,
        "status": "pending",
        "domain": doc.domain,
        "created_at": document.created_at.isoformat(),
    }


@router.get("/documents")
async def list_documents(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List documents."""
    deps = get_dependencies()
    if not deps.sqlite:
        raise RuntimeError("Storage not initialized")

    docs = await deps.sqlite.list_documents(
        status=status,
        domain=domain,
        limit=limit,
        offset=offset,
    )

    return {
        "documents": [
            {
                "id": d.id,
                "name": d.name,
                "status": d.status,
                "domain": d.domain,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "count": len(docs),
    }


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get a document by ID."""
    deps = get_dependencies()
    if not deps.sqlite:
        raise RuntimeError("Storage not initialized")

    doc = await deps.sqlite.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "name": doc.name,
        "source_uri": doc.source_uri,
        "content": doc.content,
        "status": doc.status,
        "domain": doc.domain,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat(),
    }
