#!/usr/bin/env python3
import re

# Read the gleaning service
with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'r') as f:
    content = f.read()

# Add logging to extract method
old_extract = '''    async def extract(
        self,
        text: str,
        context: str | None = None,
    ) -> ExtractionResult:
        """Perform adaptive gleaning extraction.

        Args:
            text: Text to extract from.
            context: Optional surrounding context.

        Returns:
            Complete extraction result.
        """
        results: list[ExtractionResult] = []

        # Pass 1 always runs
        pass_result = await self._extract_pass(text, 1, context, [])
        results.append(pass_result)'''

new_extract = '''    async def extract(
        self,
        text: str,
        context: str | None = None,
    ) -> ExtractionResult:
        """Perform adaptive gleaning extraction.

        Args:
            text: Text to extract from.
            context: Optional surrounding context.

        Returns:
            Complete extraction result.
        """
        logger.info(f"🔍 EXTRACT: Starting extraction for text length {len(text)}")
        logger.info(f"🔍 EXTRACT: Text preview: {text[:100]}...")
        
        results: list[ExtractionResult] = []

        # Pass 1 always runs
        logger.info(f"🔍 EXTRACT: Starting Pass 1")
        pass_result = await self._extract_pass(text, 1, context, [])
        logger.info(f"🔍 EXTRACT: Pass 1 complete - entities: {len(pass_result.entities)}, edges: {len(pass_result.edges)}")
        logger.info(f"🔍 EXTRACT: Pass 1 info density: {pass_result.information_density}")
        results.append(pass_result)'''

content = content.replace(old_extract, new_extract)

# Add logging to _extract_pass method
old_extract_pass = '''    async def _extract_pass(
        self,
        text: str,
        pass_num: int,
        context: str | None,
        previous_results: list[ExtractionResult],
    ) -> ExtractionResult:'''

new_extract_pass = '''    async def _extract_pass(
        self,
        text: str,
        pass_num: int,
        context: str | None,
        previous_results: list[ExtractionResult],
    ) -> ExtractionResult:
        logger.info(f"🔍 PASS {pass_num}: Starting extraction pass")
        logger.info(f"🔍 PASS {pass_num}: Text length: {len(text)}")
        if context:
            logger.info(f"🔍 PASS {pass_num}: Context provided: {context[:50]}...")'''

content = content.replace(old_extract_pass, new_extract_pass)

# Add logging before LLM call
old_llm_call = '''        prompt = self._build_prompt(
            text, pass_num, previous_entities, previous_edges, context
        )

        # Call LLM
        response = await self._gateway.chat.completions.create(
            model=self._config.model,
            messages=[
                ChatMessage(role="system", content="You are a precise information extraction system."),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format={"type": "json_object"},
        )'''

new_llm_call = '''        prompt = self._build_prompt(
            text, pass_num, previous_entities, previous_edges, context
        )
        
        logger.info(f"🔍 PASS {pass_num}: Built prompt, length: {len(prompt)}")
        logger.info(f"🔍 PASS {pass_num}: Prompt preview: {prompt[:200]}...")
        logger.info(f"🔍 PASS {pass_num}: Calling LLM gateway at {self._gateway._config.url}")
        logger.info(f"🔍 PASS {pass_num}: Using model: {self._config.model}")

        # Call LLM
        logger.info(f"🔍 PASS {pass_num}: Sending request to LLM...")
        response = await self._gateway.chat.completions.create(
            model=self._config.model,
            messages=[
                ChatMessage(role="system", content="You are a precise information extraction system."),
                ChatMessage(role="user", content=prompt),
            ],
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            response_format={"type": "json_object"},
        )
        logger.info(f"🔍 PASS {pass_num}: LLM response received!")
        logger.info(f"🔍 PASS {pass_num}: Response type: {type(response)}")
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].get('message', {}).get('content', '')
            logger.info(f"🔍 PASS {pass_num}: Response content length: {len(content)}")
            logger.info(f"🔍 PASS {pass_num}: Response preview: {content[:300]}...")'''

content = content.replace(old_llm_call, new_llm_call)

# Also add logging to the should_continue method
old_should_continue = '''    def should_continue_extraction(
        self, pass_num: int, result: ExtractionResult, previous_results: list[ExtractionResult]
    ) -> tuple[bool, str]:'''

new_should_continue = '''    def should_continue_extraction(
        self, pass_num: int, result: ExtractionResult, previous_results: list[ExtractionResult]
    ) -> tuple[bool, str]:
        logger.info(f"🔍 DECISION: Should we continue to pass {pass_num}?")
        logger.info(f"🔍 DECISION: Current info density: {result.information_density}")
        logger.info(f"🔍 DECISION: Max passes config: {self._config.max_passes}")'''

content = content.replace(old_should_continue, new_should_continue)

# Write back
with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'w') as f:
    f.write(content)

print("✅ Added comprehensive LLM logging to gleaning_service.py")
