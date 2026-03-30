import sys
from pathlib import Path

# Ensure `semantic_safety/` is importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_safety.semantic_router.router import SemanticRouter
import time

router = SemanticRouter(json_path="data/semantic_risk_demo.json")

# 1. Test a Known Object (Should be instant and return your generated weights)
print("TEST 1: Known Object")
res = router.get_risk_parameters("cup of water", "open laptop")
print(res["weights"])

# 2. Test an Unknown Object (Should instantly return 1.0s, then print an update 3 secs later)
print("\nTEST 2: Unknown Object")
res = router.get_risk_parameters("cup of water", "expensive microscope")
print(res["weights"]) # Will print the 1.0 Isotropic fallback

# Keep the main thread alive for a few seconds so the background thread can finish
time.sleep(4) 

# 3. Test the Unknown Object AGAIN (Should now be instantly known!)
print("\nTEST 3: Checking the previously unknown object")
res = router.get_risk_parameters("cup of water", "expensive microscope")
print(res["weights"]) # Will now print the simulated LLM weights

