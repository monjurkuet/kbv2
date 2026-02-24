"""Answer generator for RAG queries.

This module generates answers from retrieved context using:
1. LLM-based answer synthesis
2. Source attribution
3. Confidence scoring
4. Follow-up question suggestions
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from knowledge_base.ingestion.vision_client import VisionModelClient
from knowledge_base.rag.context_builder import ContextWindow

logger = logging.getLogger(__name__)


@dataclass
class RAGAnswer:
    """A RAG-generated answer."""

    question: str
    answer: str
    sources: list[dict[str, Any]]
    confidence: float = 0.0
    reasoning: Optional[str] = None
    follow_up_questions: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.follow_up_questions is None:
            self.follow_up_questions = []
        if self.metadata is None:
            self.metadata = {}


class AnswerGenerator:
    """Generates answers using retrieved context.

    This class takes a context window and question, then generates
    a comprehensive answer with source attribution.

    Example:
        >>> generator = AnswerGenerator(vision_client)
        >>> answer = await generator.generate(question, context)
        >>> print(answer.answer)
    """

    SYSTEM_PROMPT = """You are a knowledgeable trading and financial analysis assistant.
You answer questions based on the provided context from trading documents, video transcripts, and analysis materials.

Guidelines:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Include specific numbers, prices, and levels when mentioned in the context
4. Cite sources using [Source N] notation when referencing specific information
5. Be precise about timeframes and conditions mentioned
6. If there are conflicting views in the context, present both sides
7. Provide a confidence level (high/medium/low) based on how well the context addresses the question"""

    ANSWER_PROMPT = """## Question
{question}

## Context
{context}

## Instructions
Based on the provided context, answer the question comprehensively.

Your response should include:
1. A direct answer to the question
2. Supporting evidence from the context (with source citations)
3. Any relevant caveats or conditions mentioned
4. A confidence assessment (high/medium/low)

If the context doesn't contain enough information to fully answer the question, clearly state what information is available and what is missing.

## Response"""

    FOLLOW_UP_PROMPT = """Based on this Q&A:
Question: {question}
Answer: {answer}

Generate 2-3 relevant follow-up questions that a user might want to ask next.
Format as a JSON array of strings."""

    def __init__(
        self,
        vision_client: VisionModelClient,
        model: str = "qwen3-max",
        max_tokens: int = 2048,
    ) -> None:
        """Initialize answer generator.

        Args:
            vision_client: Vision model client for LLM calls.
            model: Model to use for generation.
            max_tokens: Maximum tokens in response.
        """
        self._client = vision_client
        self._model = model
        self._max_tokens = max_tokens

    async def generate(
        self,
        question: str,
        context: ContextWindow,
        include_follow_ups: bool = True,
    ) -> RAGAnswer:
        """Generate an answer from context.

        Args:
            question: User question.
            context: Context window with retrieved chunks.
            include_follow_ups: Whether to generate follow-up questions.

        Returns:
            RAGAnswer with the response.
        """
        # Build context text
        context_text = context.to_prompt_text(include_metadata=True)

        # Format prompt
        prompt = self.ANSWER_PROMPT.format(
            question=question,
            context=context_text,
        )

        # Call LLM
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._client._call_chat_api(
            messages,
            model=self._model,
        )

        answer_text = response["choices"][0]["message"]["content"]

        # Extract confidence from answer
        confidence = self._extract_confidence(answer_text)

        # Generate follow-up questions if requested
        follow_ups = []
        if include_follow_ups:
            follow_ups = await self._generate_follow_ups(question, answer_text)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=context.sources,
            confidence=confidence,
            follow_up_questions=follow_ups,
            metadata={
                "model": self._model,
                "context_chunks": len(context.chunks),
                "context_tokens": context.total_tokens,
            },
        )

    def _extract_confidence(self, answer: str) -> float:
        """Extract confidence level from answer.

        Args:
            answer: Generated answer.

        Returns:
            Confidence score (0-1).
        """
        answer_lower = answer.lower()

        # Look for explicit confidence mentions
        if "high confidence" in answer_lower or "confidence: high" in answer_lower:
            return 0.9
        elif "medium confidence" in answer_lower or "confidence: medium" in answer_lower:
            return 0.6
        elif "low confidence" in answer_lower or "confidence: low" in answer_lower:
            return 0.3

        # Look for uncertainty indicators
        uncertainty_phrases = [
            "i don't have enough information",
            "the context doesn't mention",
            "not specified in the context",
            "unclear from the provided",
        ]

        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                return 0.4

        # Default confidence
        return 0.7

    async def _generate_follow_ups(
        self,
        question: str,
        answer: str,
    ) -> list[str]:
        """Generate follow-up questions.

        Args:
            question: Original question.
            answer: Generated answer.

        Returns:
            List of follow-up questions.
        """
        try:
            result = await self._client.extract_structured(
                self.FOLLOW_UP_PROMPT.format(question=question, answer=answer[:1000]),
                {"type": "array", "items": {"type": "string"}},
                model=self._model,
            )

            follow_ups = result.data
            if isinstance(follow_ups, list):
                return [q for q in follow_ups if isinstance(q, string)][:3]
        except Exception as e:
            logger.debug(f"Failed to generate follow-ups: {e}")

        return []

    async def generate_with_reasoning(
        self,
        question: str,
        context: ContextWindow,
    ) -> RAGAnswer:
        """Generate answer with step-by-step reasoning.

        Args:
            question: User question.
            context: Context window.

        Returns:
            RAGAnswer with reasoning included.
        """
        reasoning_prompt = """## Question
{question}

## Context
{context}

## Instructions
First, think through this step-by-step:
1. What is the core question being asked?
2. What relevant information exists in the context?
3. Are there any numbers, prices, or levels mentioned?
4. Are there any conditions or caveats?
5. How confident can we be in the answer?

Then provide your answer with source citations.

## Response
**Reasoning:**
[Your step-by-step analysis]

**Answer:**
[Your comprehensive answer]

**Confidence:** [high/medium/low]"""

        context_text = context.to_prompt_text(include_metadata=True)
        prompt = reasoning_prompt.format(question=question, context=context_text)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self._client._call_chat_api(
            messages,
            model=self._model,
        )

        full_response = response["choices"][0]["message"]["content"]

        # Parse reasoning and answer
        reasoning = None
        answer_text = full_response

        if "**Reasoning:**" in full_response and "**Answer:**" in full_response:
            parts = full_response.split("**Answer:**")
            if len(parts) == 2:
                reasoning_part = parts[0].replace("**Reasoning:**", "").strip()
                answer_text = parts[1].strip()

                # Extract confidence from answer section
                if "**Confidence:**" in answer_text:
                    answer_text = answer_text.split("**Confidence:**")[0].strip()

                reasoning = reasoning_part

        confidence = self._extract_confidence(full_response)

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=context.sources,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "model": self._model,
                "context_chunks": len(context.chunks),
                "context_tokens": context.total_tokens,
            },
        )


# Import for type hints
import string
