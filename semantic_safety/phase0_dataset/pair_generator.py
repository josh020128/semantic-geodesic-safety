import itertools
import json

group_a = ["cup of water", "kitchen knife", "hot soldering iron", "heavy metal wrench"]
group_b = ["open laptop", "wine glass", "balloon", "power strip"]
group_c = ["kitchen sink", "plastic tray"]

# Define M (Manipulated) and S (Scene)
M = group_a + group_b
S = group_a + group_b + group_c

# Generate Cartesian Product M x S
all_pairs = list(itertools.product(M, S))

print(f"Total pairs generated: {len(all_pairs)}\n")

# Format into the prompt structure
prompt_payload = "Calculate risk for the following pairs:\n"
for i, (manip, scene) in enumerate(all_pairs):
    prompt_payload += f"{i+1}. Manipulated: '{manip}', Scene: '{scene}'\n"

print(prompt_payload[:300]) # Preview the first few lines
print("...\n[Truncated for display]")