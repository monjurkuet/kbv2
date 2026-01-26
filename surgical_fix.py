#!/usr/bin/env python3

# Read file line by line and replace at exact position
with open('src/knowledge_base/orchestrator.py', 'r') as f:
    lines = f.readlines()

# Find line 400 (index 399) which has on_conflict_do_update
for i, line in enumerate(lines):
    if i == 399 and 'on_conflict_do_update' in line:
        print(f"Line {i+1}: {line.rstrip()}")
        print(f"Changing line {i+1} to on_conflict_do_nothing()")
        lines[i] = "                    ).on_conflict_do_nothing()\n"
        print(f"New line: {lines[i].rstrip()}")
        break
    elif 'on_conflict_do_update' in line:
        print(f"Found on_conflict_do_update at line {i+1}: {line.rstrip()}")

with open('src/knowledge_base/orchestrator.py', 'w') as f:
    f.writelines(lines)

print("✅ Applied surgical fix at line 400")
