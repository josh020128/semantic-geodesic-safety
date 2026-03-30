import concurrent.futures
import json
import os
import threading
from pathlib import Path
from typing import Any


class SemanticRouter:
    def __init__(self, json_path: str = "data/semantic_risk_demo.json"):
        """
        Tier 1: O(1) dictionary lookup for known (manipulated, scene) pairs.
        Tier 2: conservative isotropic fallback for unknown pairs (returns immediately).
        Tier 3: background thread to query an LLM and update the in-memory cache.
        """
        self.json_path = json_path
        self._lock = threading.Lock()
        self.knowledge_base: dict[tuple[str, str], dict[str, Any]] = self._load_knowledge_base()

        # Thread pool for Tier 3 "Slow-Brain" calls
        self.slow_brain_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.pending_queries: set[tuple[str, str]] = set()  # prevents spamming the API for the same pair

    def _resolve_json_path(self) -> str:
        """
        Resolve json_path robustly.

        If json_path is relative, treat it as relative to the repo root (parent of /scripts and /semantic_safety).
        """
        p = Path(self.json_path)
        if p.is_absolute():
            return str(p)
        root = Path(__file__).resolve().parent.parent.parent
        return str((root / p).resolve())

    def _load_knowledge_base(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Loads the Phase 0 JSON array into an O(1) dictionary lookup."""
        resolved = self._resolve_json_path()
        if not os.path.exists(resolved):
            print(f"Warning: {resolved} not found. Starting with empty knowledge base.")
            return {}

        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)

        kb: dict[tuple[str, str], dict[str, Any]] = {}
        if not isinstance(data, list):
            print(f"Warning: {resolved} did not contain a JSON array. Starting with empty knowledge base.")
            return kb

        for entry in data:
            if not isinstance(entry, dict):
                continue
            manipulated = str(entry.get("manipulated", "")).strip().lower()
            scene = str(entry.get("scene", "")).strip().lower()
            if not manipulated or not scene:
                continue
            key = (manipulated, scene)
            kb[key] = entry
        return kb

    def get_risk_parameters(self, manipulated_label: str, scene_label: str) -> dict[str, Any]:
        """
        Main Loop 1 entry point (callable at ~30Hz).
        Returns the cached entry when known, otherwise returns a conservative fallback immediately
        and triggers an async Tier 3 fetch (once per unique pair).
        """
        manipulated = str(manipulated_label).strip().lower()
        scene = str(scene_label).strip().lower()
        key = (manipulated, scene)

        # ==========================================
        # TIER 1: FAST-BRAIN MATCH (O(1) Lookup)
        # ==========================================
        with self._lock:
            hit = self.knowledge_base.get(key)
        if hit is not None:
            return hit

        # ==========================================
        # TIER 2 & 3: UNKNOWN OBJECT (Parallel Split)
        # ==========================================
        # If we haven't already asked the LLM about this pair, spin up Tier 3
        should_submit = False
        with self._lock:
            if key not in self.pending_queries:
                self.pending_queries.add(key)
                should_submit = True
        if should_submit:
            self.slow_brain_pool.submit(self._async_llm_query, manipulated_label, scene_label)

        # Immediately return the Tier 2 conservative fallback to keep the robot safe
        print(f"[Tier 2 Fallback] Unknown pair: {key}. Deploying Isotropic Sphere.")
        return {
            "manipulated": manipulated_label,
            "scene": scene_label,
            "topology_template": "isotropic_sphere",
            "weights": {
                "w_+x": 1.0,
                "w_-x": 1.0,
                "w_+y": 1.0,
                "w_-y": 1.0,
                "w_+z": 1.0,
                "w_-z": 1.0,
            },
            "receptacle_attenuation": 1.0,
        }

    def _async_llm_query(self, manipulated_label: str, scene_label: str) -> None:
        """
        TIER 3: SLOW-BRAIN THREAD
        Runs in the background. Calls Gemini (or other LLM), then updates the in-memory cache.
        """
        manipulated = str(manipulated_label).strip().lower()
        scene = str(scene_label).strip().lower()
        key = (manipulated, scene)
        print(f"[Tier 3 Thread] Querying LLM for {key}...")

        # TODO: Insert your Gemini API call here (using the exact same prompt from Phase 0).
        # For now, simulate a network delay and return a mock response.
        import time

        time.sleep(3)
        mock_response = {
            "manipulated": manipulated_label,
            "scene": scene_label,
            "topology_template": "upward_vertical_cone",
            "weights": {
                "w_+x": 0.2,
                "w_-x": 0.2,
                "w_+y": 0.2,
                "w_-y": 0.2,
                "w_+z": 0.9,
                "w_-z": 0.0,
            },
            "receptacle_attenuation": 0.5,
        }

        with self._lock:
            self.knowledge_base[key] = mock_response
            self.pending_queries.discard(key)
        print(f"[Tier 3 Thread] Update complete! {key} is now safely mapped in Tier 1.")
