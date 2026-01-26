#!/usr/bin/env python3

# Store all API keys  
GOOGLE_API_KEY=AIzaSyA9_aUQ5lMjed-N7PaEYeBwoRA-vgAT1l4
GOOGLE_API_KEY_2=AIzaSyCdAaZU8TnRrZiJ75pIfNBGwwWQWaCbxOk
GOOGLE_API_KEY_3=AIzaSyCvX779-2d9j2h00oHBcUZrqxz3lPTCqF4
GOOGLE_API_KEY_4=AIzaSyDQXofNEBXYTwqAn0KJDI7CwRg3NeAeKsc
GOOGLE_API_KEY_5=AIzaSyD15se_AlK3LLWwr1ZNRNofMF3iOegV19Y

with open(".env", "r") as f:
    env = f.read()

# Update primary key
env = env.replace("GOOGLE_API_KEY=", "# OLD KEY\n# GOOGLE_API_KEY=")
if "GOOGLE_API_KEY=" not in env:
    env = env + f"\n# Google API Keys (with fallback)\nGOOGLE_API_KEY={GOOGLE_API_KEY}\n"

# Add fallback keys
for i, var in enumerate(["GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GOOGLE_API_KEY_4", "GOOGLE_API_KEY_5"], 2):
    value = globals()[var]
    if var not in env:
        env += f"{var}={value}\n"

with open(".env", "w") as f:
    f.write(env)

print("✅ Updated .env with Google API keys")
