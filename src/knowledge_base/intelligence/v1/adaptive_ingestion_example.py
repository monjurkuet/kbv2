"""Example integration of adaptive ingestion engine."""

from knowledge_base.intelligence.v1.adaptive_ingestion_engine import (
    AdaptiveIngestionEngine,
    PipelineRecommendation,
)

# Example usage in orchestrator:

async def optimized_knowledge_extraction(orchestrator, document, chunks):
    """Optimized knowledge extraction using adaptive engine."""

    # Initialize adaptive engine
    adaptive_engine = AdaptiveIngestionEngine(orchestrator._gateway)

    # Combine chunk texts for analysis
    document_text = " ".join([chunk.text for chunk in chunks[:3]])  # Sample first 3 chunks

    # Analyze and get recommendation (1 LLM call)
    recommendation = await adaptive_engine.analyze_document(
        document_text=document_text,
        document_name=document.name,
        file_size_bytes=document.size if hasattr(document, 'size') else 5000
    )

    print(f"🤖 Adaptive Analysis:")
    print(f"   Complexity: {recommendation.complexity}")
    print(f"   Approach: {recommendation.approach}")
    print(f"   Chunk size: {recommendation.chunk_size}")
    print(f"   Estimated entities: {recommendation.expected_entity_count}")
    print(f"   Processing time: {recommendation.estimated_processing_time}")
    print(f"   LLM calls: {adaptive_engine._estimate_llm_calls(recommendation)}")

    # For the current document (5KB, financial content, structured):
    # Expected recommendation:
    #   complexity: "moderate"
    #   approach: "multi_agent"
    #   chunk_size: 1536  [Current chunks: 175, 290, 195 tokens]
    #   expected_entity_count: 25-30
    #   estimated_calls: ~15-20 (down from 25-30!)

    # Apply recommendation
    if recommendation.approach.value == "gleaning":
        # Only 3-5 LLM calls total
        await use_gleaning_only(orchestrator, chunks)

    elif recommendation.approach.value == "gleaning_enhanced":
        # 6-8 LLM calls
        await use_enhanced_gleaning(orchestrator, chunks)

    else:  # multi_agent
        # ~15-20 LLM calls instead of 25-30
        strategy = adaptive_engine.get_ingestion_strategy(recommendation)
        await use_optimized_multi_agent(
            orchestrator,
            chunks,
            max_iterations=recommendation.max_enhancement_iterations
        )


"""
## ESTIMATED TIME SAVINGS

Original approach:
- Always use multi-agent
- 3 chunks × (1 perception + 3 enhancement + 1 eval) = 15 calls minimum
- Plus fallback gleaning = 3 more calls
- **Total: 18-30 LLM calls**
- **Time: 15-25 minutes** with proxy

Adaptive approach:
- 1 analysis call
- Then optimized calls based on complexity
- For this document: ~15-18 calls (10-20% reduction)
- **Time: 12-20 minutes** (saves 3-5 minutes)

For simple documents (blog posts, news):
- Only 3-5 calls total
- **Time: 2-4 minutes** (80% faster!)

For complex documents (research papers):
- Still use full multi-agent: ~25-30 calls
- But with tuned parameters (chunk size, iterations)
- Better quality without wasting calls on simple docs
"""

# Port your current ingestion to use this when it completes!
