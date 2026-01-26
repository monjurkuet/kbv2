#!/usr/bin/env python3

# Store all API keys
keys = [
    "AIzaSyCdAaZU8TnRrZiJ75pIfNBGwwWQWaCbxOk",  # scar.50cent@gmail.com
    "AIzaSyA9_aUQ5lMjed-N7PaEYeBwoRA-vgAT1l4",  # datahiveinfo@gmail.com  
    "AIzaSyCvX779-2d9j2h00oHBcUZrqxz3lPTCqF4",  # evetyler51@gmail.com
    "AIzaSyDQXofNEBXYTwqAn0KJDI7CwRg3NeAeKsc",  # chankiki486@gmail.com
    "AIzaSyD15se_AlK3LLWwr1ZNRNofMF3iOegV19Y",  # orpitamim2024@gmail.com
]

# Write to env file with fallback notation
env_content = """# Google Embeddings Configuration
# Primary API key
GOOGLE_API_KEY=AIzaSyA9_aUQ5lMjed-N7PaEYeBwoRA-vgAT1l4
# Fallback keys (used if primary fails)
GOOGLE_API_KEY_2=AIzaSyCdAaZU8TnRrZiJ75pIfNBGwwWQWaCbxOk
GOOGLE_API_KEY_3=AIzaSyCvX779-2d9j2h00oHBcUZrqxz3lPTCqF4
GOOGLE_API_KEY_4=AIzaSyDQXofNEBXYTwqAn0KJDI7CwRg3NeAeKsc
GOOGLE_API_KEY_5=AIzaSyD15se_AlK3LLWwr1ZNRNofMF3iOegV19Y
GOOGLE_EMBEDDING_URL=https://generativelanguage.googleapis.com
GOOGLE_EMBEDDING_MODEL=embedding-001
"""

with open(".env", "r") as f:
    env = f.read()

# Replace the Google section
import re
env = re.sub(r'GOOGLE_API_KEY=.*', 'GOOGLE_API_KEY=AIzaSyA9_aUQ5lMjed-N7PaEYeBwoRA-vgAT1l4', env)
env = re.sub(r'GOOGLE_EMBEDDING_MODEL=.*', 'GOOGLE_EMBEDDING_MODEL=embedding-001', env)

# Add fallback keys if not present
if "GOOGLE_API_KEY_2" not in env:
    env += """\n# Fallback API keys\n"""
    env += "\n".join([f"GOOGLE_API_KEY_{i+1}={key}" for i, key in enumerate(keys)])

with open(".env", "w") as f:
    f.write(env)

print("✅ Updated .env with fallback API keys")
print("\nAPI Keys configured:")
for i, key in enumerate(keys, 1):
    print(f"  {i}. {key[:15]}...{' ' * 5}({['Primary'] + ['Fallback'] * 4}[i-1]})")
