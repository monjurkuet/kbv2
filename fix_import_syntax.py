#!/usr/bin/env python3

with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'r') as f:
    content = f.read()

# Fix the broken import syntax
broken_import = '''from knowledge_base.common.temporal_utils import (
import time
    TemporalClaim,
    TemporalNormalizer,
    TemporalType,
)'''

fixed_import = '''import time
from knowledge_base.common.temporal_utils import (
    TemporalClaim,
    TemporalNormalizer,
    TemporalType,
)'''

content = content.replace(broken_import, fixed_import)

with open('src/knowledge_base/ingestion/v1/gleaning_service.py', 'w') as f:
    f.write(content)

print("✅ Fixed import syntax error")
