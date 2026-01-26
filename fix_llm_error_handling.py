#!/usr/bin/env python3

# Read the file
with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'r') as f:
    content = f.read()

# Fix the exception handling to detect LLM failures
old_code = '''        try:
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
        except Exception as e:
            logger.error(f"Failed to generate text for pass {pass_num}: {e}")
            return ExtractionResult()'''

new_code = '''        try:
            logger.info(f"🔍 PASS {pass_num}: About to call LLM gateway")
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
            logger.info(f"🔍 PASS {pass_num}: LLM call successful!")
        except Exception as e:
            logger.error(f"🔍 PASS {pass_num}: LLM CALL FAILED: {e}")
            logger.error(f"🔍 PASS {pass_num}: This is an LLM error, NOT low information density!")
            logger.error(f"🔍 PASS {pass_num}: Gateway URL: {self._gateway._config.url}")
            logger.error(f"🔍 PASS {pass_num}: Model: {self._config.model}")
            logger.error(f"🔍 PASS {pass_num}: Error type: {type(e).__name__}")
            # Return an extraction result that indicates LLM failure
            result = ExtractionResult()
            result.information_density = -1.0  # Special marker for LLM failure
            return result'''

content = content.replace(old_code, new_code)

# Update the should_continue method to detect LLM failures
old_should = '''        # Check information density of the most recent result
        if result.information_density < self._config.min_density_threshold:
            return False, "low_information_density"'''

new_should = '''        # Check if this is actually an LLM failure
        if result.information_density < 0:
            return False, "llm_failure"
        
        # Check information density of the most recent result
        if result.information_density < self._config.min_density_threshold:
            return False, "low_information_density"'''

content = content.replace(old_should, new_should)

# Write back
with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed LLM error handling - now properly reports 'llm_failure' instead of 'low_information_density'")
