#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'r') as f:
    content = f.read()

# Add more detailed LLM request/response logging
old_call = '''        logger.info(f"🔍 PASS {pass_num}: About to call LLM gateway")
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
        logger.info(f"🔍 PASS {pass_num}: LLM call successful!")'''

new_call = '''        logger.info(f"🔍 PASS {pass_num}: About to call LLM gateway")
        logger.info(f"🔍 PASS {pass_num}: Gateway URL: {self._gateway._config.url}")
        logger.info(f"🔍 PASS {pass_num}: Model: {self._config.model}")
        logger.info(f"🔍 PASS {pass_num}: Temperature: {self._config.temperature}")
        logger.info(f"🔍 PASS {pass_num}: Max tokens: {self._config.max_tokens}")
        
        try:
            logger.info(f"🔍 PASS {pass_num}: === LLM REQUEST START ===")
            logger.info(f"🔍 PASS {pass_num}: Prompt:\n{prompt}")
            logger.info(f"🔍 PASS {pass_num}: === LLM REQUEST END ===")
            
            start_time = time.time()
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
            end_time = time.time()
            
            logger.info(f"🔍 PASS {pass_num}: LLM call successful! ({end_time - start_time:.2f}s)")
            logger.info(f"🔍 PASS {pass_num}: Response type: {type(response).__name__}")
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].get('message', {}).get('content', '')
                logger.info(f"🔍 PASS {pass_num}: === LLM RESPONSE START ===")
                logger.info(f"🔍 PASS {pass_num}: Content length: {len(content)}")
                logger.info(f"🔍 PASS {pass_num}: Content preview:\n{content[:500]}...")
                logger.info(f"🔍 PASS {pass_num}: === LLM RESPONSE END ===")
            
            if hasattr(response, 'usage'):
                logger.info(f"🔍 PASS {pass_num}: Token usage: {response.usage}")
                
        except Exception as e:
            logger.error(f"🔍 PASS {pass_num}: LLM EXCEPTION: {e}")
            logger.error(f"🔍 PASS {pass_num}: Exception type: {type(e).__name__}")
            result = ExtractionResult()
            result.information_density = -1.0
            return result'''

content = content.replace(old_call, new_call)

# Add time import at the top if not present
if 'import time' not in content.split('\n')[:20]:
    # Find the line after imports and add it
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('import') and not line.startswith('from') and not line.startswith('"""'):
            insert_idx = i
            break
    lines.insert(insert_idx, 'import time')
    content = '\n'.join(lines)

with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'w') as f:
    f.write(content)

print("✅ Added comprehensive LLM request/response logging")
