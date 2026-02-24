"""RAG (Retrieval-Augmented Generation) module for Knowledge Base.

This module provides:
- Query pipeline for RAG-based question answering
- Context retrieval and assembly
- Answer generation with source attribution
- Multiple RAG strategies (LightRAG, HippoRAG, CRAG)
"""

from knowledge_base.rag.query_pipeline import RAGQueryPipeline
from knowledge_base.rag.context_builder import ContextBuilder
from knowledge_base.rag.answer_generator import AnswerGenerator

__all__ = [
    "RAGQueryPipeline",
    "ContextBuilder",
    "AnswerGenerator",
]
