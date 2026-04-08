from __future__ import annotations

import concurrent.futures
import difflib
import json
import os
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


def _dot(a: Any, b: Any) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


class SemanticRouter:
    """
    Final semantic router.

    Resolution order:
      1) exact canonicalized cache hit
      2) nearest-neighbor canonicalization
      3) family-level fallback
      4) async LLM batch expansion
      5) conservative fallback

    Runtime design:
      - hot path never blocks on LLM
      - background dispatcher batches unknown pairs
      - single writer thread persists JSON updates
    """

    DEFAULT_ALIASES: dict[str, str] = {
        # tools
        "drill": "power drill",
        "electric drill": "power drill",
        "cordless drill": "power drill",
        "hand drill": "power drill",
        "rotary tool": "power drill",
        "screw driver": "screwdriver",
        "box cutter": "knife",

        # bowls / containers
        "mixing bowl": "bowl",
        "ceramic bowl": "bowl",
        "plastic bowl": "bowl",
        "ceramic vessel": "bowl",
        "vessel": "container",

        # laptops / electronics
        "notebook": "laptop",
        "notebook computer": "laptop",
        "computer": "laptop",
        "electronic device": "device",

        # cups / mugs
        "mug": "cup",
        "coffee mug": "cup",
        "travel mug": "cup",

        # receptacles / supports
        "trash can": "bin",
        "garbage can": "bin",
        "countertop": "counter",
    }

    SAFE_RECEPTACLES = {
        "sink",
        "bowl",
        "tray",
        "bucket",
        "bin",
        "container",
        "plate",
        "cup",
    }

    MANIPULATED_FAMILY_KEYWORDS: dict[str, set[str]] = {
        "liquid": {"water", "coffee", "tea", "juice", "milk", "soup", "liquid", "drink"},
        "thermal": {"hot", "pan", "pot", "kettle", "iron", "soldering iron"},
        "sharp": {"knife", "scissors", "blade", "box cutter", "screwdriver"},
        "impact": {"hammer", "wrench", "drill", "power drill", "tool"},
        "contamination": {"trash", "waste", "dirty", "chemical", "paint", "oil"},
        "electrical": {"battery", "charger", "power strip", "cable", "adapter"},
    }

    SCENE_FAMILY_KEYWORDS: dict[str, set[str]] = {
        "electronics_family": {
            "laptop", "computer", "monitor", "keyboard", "phone", "tablet",
            "power strip", "charger", "cable", "adapter", "device",
        },
        "receptacle_family": {
            "sink", "bowl", "tray", "bucket", "bin", "container", "cup", "plate", "glass",
        },
        "fragile_family": {
            "wine glass", "glass", "vase", "ceramic", "plate", "bowl",
        },
        "support_family": {
            "table", "shelf", "counter", "wall", "floor", "desk",
        },
        "tool_family": {
            "power drill", "drill", "screwdriver", "hammer", "wrench", "pliers", "scissors", "knife",
        },
    }

    def __init__(
        self,
        json_path: str = "data/semantic_risk_demo.json",
        system_instruction: str = "",
        llm_callback: Optional[Callable[[str, str], dict[str, Any]]] = None,
        llm_batch_callback: Optional[Callable[[str, list[str], str], list[dict[str, Any]]]] = None,
        persist_updates: bool = True,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        nearest_threshold: float = 0.68,
        batch_window_s: float = 0.15,
        max_batch_size: int = 8,
        max_workers: int = 2,
    ):
        self.json_path = json_path
        self.system_instruction = system_instruction.strip()
        self.llm_callback = llm_callback
        self.llm_batch_callback = llm_batch_callback
        self.persist_updates = persist_updates

        self.nearest_threshold = float(nearest_threshold)
        self.batch_window_s = float(batch_window_s)
        self.max_batch_size = int(max_batch_size)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self.knowledge_base: dict[tuple[str, str], dict[str, Any]] = self._load_knowledge_base()

        # Fast indices
        self.scene_labels: set[str] = set()
        self.manipulated_labels: set[str] = set()
        self.scenes_by_manipulated: dict[str, set[str]] = defaultdict(set)
        self.manipulateds_by_scene: dict[str, set[str]] = defaultdict(set)
        self._rebuild_indices()

        # Optional local embedding model
        self._embedder = None
        self._sentence_transformers_available = False
        self._scene_embedding_cache: dict[str, Any] = {}
        self._manip_embedding_cache: dict[str, Any] = {}
        self._try_init_embedder(embedding_model_name)
        self._rebuild_embedding_cache()

        # Async machinery
        self.slow_brain_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.pending_queries: set[tuple[str, str]] = set()
        self.request_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.writer_queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()

        self.dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            name="semantic-router-dispatcher",
            daemon=True,
        )
        self.dispatcher_thread.start()

        self.writer_thread: Optional[threading.Thread] = None
        if self.persist_updates:
            self.writer_thread = threading.Thread(
                target=self._writer_loop,
                name="semantic-router-writer",
                daemon=True,
            )
            self.writer_thread.start()

    # ------------------------------------------------------------------
    # Path / load / save
    # ------------------------------------------------------------------

    def _resolve_json_path(self) -> str:
        p = Path(self.json_path)
        if p.is_absolute():
            return str(p)
        root = Path(__file__).resolve().parent.parent.parent
        return str((root / p).resolve())

    def _load_json_array(self) -> list[dict[str, Any]]:
        resolved = self._resolve_json_path()
        if not os.path.exists(resolved):
            print(f"Warning: {resolved} not found. Starting with empty knowledge base.")
            return []

        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"Warning: {resolved} did not contain a JSON array. Starting with empty knowledge base.")
            return []
        return [x for x in data if isinstance(x, dict)]

    def _load_knowledge_base(self) -> dict[tuple[str, str], dict[str, Any]]:
        kb: dict[tuple[str, str], dict[str, Any]] = {}

        for entry in self._load_json_array():
            normalized = self._normalize_entry(entry)
            manipulated = normalized["manipulated"]
            scene = normalized["scene"]
            if manipulated and scene:
                kb[(manipulated, scene)] = normalized

        return kb

    def _persist_entry_sync(self, entry: dict[str, Any]) -> None:
        resolved = self._resolve_json_path()
        os.makedirs(os.path.dirname(resolved), exist_ok=True)

        data = self._load_json_array()
        normalized = self._normalize_entry(entry)
        key = (normalized["manipulated"], normalized["scene"])

        updated = False
        for i, row in enumerate(data):
            row_norm = self._normalize_entry(row)
            row_key = (row_norm["manipulated"], row_norm["scene"])
            if row_key == key:
                data[i] = normalized
                updated = True
                break

        if not updated:
            data.append(normalized)

        with open(resolved, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break

            try:
                self._persist_entry_sync(item)
            except Exception as e:
                print(f"[Writer Thread] Failed to persist entry: {e}")

    # ------------------------------------------------------------------
    # Index / embedding init
    # ------------------------------------------------------------------

    def _rebuild_indices(self) -> None:
        self.scene_labels.clear()
        self.manipulated_labels.clear()
        self.scenes_by_manipulated.clear()
        self.manipulateds_by_scene.clear()

        for manipulated, scene in self.knowledge_base.keys():
            self.manipulated_labels.add(manipulated)
            self.scene_labels.add(scene)
            self.scenes_by_manipulated[manipulated].add(scene)
            self.manipulateds_by_scene[scene].add(manipulated)

    def _try_init_embedder(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(model_name)
            self._sentence_transformers_available = True
            print(f"[Router] Loaded embedding model: {model_name}")
        except Exception:
            self._embedder = None
            self._sentence_transformers_available = False
            print("[Router] sentence-transformers not available. Falling back to lexical nearest matching.")

    def _rebuild_embedding_cache(self) -> None:
        if not self._sentence_transformers_available:
            return

        scene_list = sorted(self.scene_labels)
        manip_list = sorted(self.manipulated_labels)

        if scene_list:
            scene_vecs = self._embedder.encode(scene_list, normalize_embeddings=True)
            self._scene_embedding_cache = {label: vec for label, vec in zip(scene_list, scene_vecs)}
        else:
            self._scene_embedding_cache = {}

        if manip_list:
            manip_vecs = self._embedder.encode(manip_list, normalize_embeddings=True)
            self._manip_embedding_cache = {label: vec for label, vec in zip(manip_list, manip_vecs)}
        else:
            self._manip_embedding_cache = {}

    # ------------------------------------------------------------------
    # Label normalization / canonicalization
    # ------------------------------------------------------------------

    def _canonicalize_label(self, label: str) -> str:
        s = str(label).strip().lower()
        s = " ".join(s.split())
        return self.DEFAULT_ALIASES.get(s, s)

    def _normalize_key(self, manipulated_label: str, scene_label: str) -> tuple[str, str]:
        manipulated = self._canonicalize_label(manipulated_label)
        scene = self._canonicalize_label(scene_label)
        return manipulated, scene

    def _normalize_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        manipulated = self._canonicalize_label(entry.get("manipulated", ""))
        scene = self._canonicalize_label(entry.get("scene", ""))

        weights = entry.get("weights", {}) or {}
        normalized = {
            "manipulated": manipulated,
            "scene": scene,
            "families": list(entry.get("families", [])),
            "topology_template": entry.get("topology_template", "isotropic_sphere"),
            "weights": {
                "w_+x": float(weights.get("w_+x", 0.0)),
                "w_-x": float(weights.get("w_-x", 0.0)),
                "w_+y": float(weights.get("w_+y", 0.0)),
                "w_-y": float(weights.get("w_-y", 0.0)),
                "w_+z": float(weights.get("w_+z", 0.0)),
                "w_-z": float(weights.get("w_-z", 0.0)),
            },
            "vertical_rule": entry.get("vertical_rule", "standard_decay"),
            "lateral_decay": entry.get("lateral_decay", "moderate"),
            "receptacle_attenuation": float(entry.get("receptacle_attenuation", 1.0)),
        }
        return normalized

    # ------------------------------------------------------------------
    # Nearest-neighbor matching
    # ------------------------------------------------------------------

    def _lexical_similarity(self, a: str, b: str) -> float:
        a = self._canonicalize_label(a)
        b = self._canonicalize_label(b)

        if a == b:
            return 1.0

        seq = difflib.SequenceMatcher(None, a, b).ratio()

        toks_a = set(a.split())
        toks_b = set(b.split())
        if not toks_a or not toks_b:
            jaccard = 0.0
        else:
            jaccard = len(toks_a & toks_b) / len(toks_a | toks_b)

        return 0.6 * seq + 0.4 * jaccard

    def _embedding_similarity(self, query: str, label: str, kind: str) -> float:
        if not self._sentence_transformers_available:
            return self._lexical_similarity(query, label)

        query = self._canonicalize_label(query)
        label = self._canonicalize_label(label)

        cache = self._scene_embedding_cache if kind == "scene" else self._manip_embedding_cache
        if label not in cache:
            return self._lexical_similarity(query, label)

        query_vec = self._embedder.encode([query], normalize_embeddings=True)[0]
        label_vec = cache[label]
        return _dot(query_vec, label_vec)

    def _nearest_label(
        self,
        query: str,
        candidates: Sequence[str],
        kind: str,
    ) -> tuple[Optional[str], float]:
        if not candidates:
            return None, 0.0

        best_label = None
        best_score = -1.0
        for cand in candidates:
            score = self._embedding_similarity(query, cand, kind=kind)
            if score > best_score:
                best_score = score
                best_label = cand

        return best_label, float(best_score)

    def _nearest_pair_lookup(
        self,
        manipulated_label: str,
        scene_label: str,
    ) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
        manipulated = self._canonicalize_label(manipulated_label)
        scene = self._canonicalize_label(scene_label)

        # nearest scene given exact manipulated
        exact_manip_scenes = sorted(self.scenes_by_manipulated.get(manipulated, []))
        if exact_manip_scenes:
            nn_scene, s_score = self._nearest_label(scene, exact_manip_scenes, kind="scene")
            if nn_scene is not None and s_score >= self.nearest_threshold:
                hit = self.knowledge_base.get((manipulated, nn_scene))
                if hit is not None:
                    return self._attach_meta(
                        hit,
                        match_type="nearest_scene",
                        raw_manipulated=manipulated_label,
                        raw_scene=scene_label,
                        score=s_score,
                        resolved_manipulated=manipulated,
                        resolved_scene=nn_scene,
                    ), {
                        "match_type": "nearest_scene",
                        "score": s_score,
                    }

        # nearest manipulated given exact scene
        exact_scene_manips = sorted(self.manipulateds_by_scene.get(scene, []))
        if exact_scene_manips:
            nn_manip, m_score = self._nearest_label(manipulated, exact_scene_manips, kind="manipulated")
            if nn_manip is not None and m_score >= self.nearest_threshold:
                hit = self.knowledge_base.get((nn_manip, scene))
                if hit is not None:
                    return self._attach_meta(
                        hit,
                        match_type="nearest_manipulated",
                        raw_manipulated=manipulated_label,
                        raw_scene=scene_label,
                        score=m_score,
                        resolved_manipulated=nn_manip,
                        resolved_scene=scene,
                    ), {
                        "match_type": "nearest_manipulated",
                        "score": m_score,
                    }

        # global nearest on both sides
        nn_manip, m_score = self._nearest_label(manipulated, sorted(self.manipulated_labels), kind="manipulated")
        nn_scene, s_score = self._nearest_label(scene, sorted(self.scene_labels), kind="scene")

        if (
            nn_manip is not None
            and nn_scene is not None
            and m_score >= self.nearest_threshold
            and s_score >= self.nearest_threshold
        ):
            hit = self.knowledge_base.get((nn_manip, nn_scene))
            if hit is not None:
                return self._attach_meta(
                    hit,
                    match_type="nearest_pair",
                    raw_manipulated=manipulated_label,
                    raw_scene=scene_label,
                    score=0.5 * (m_score + s_score),
                    resolved_manipulated=nn_manip,
                    resolved_scene=nn_scene,
                ), {
                    "match_type": "nearest_pair",
                    "score": 0.5 * (m_score + s_score),
                }

        return None, {"match_type": "none", "score": 0.0}

    # ------------------------------------------------------------------
    # Family inference / family fallback
    # ------------------------------------------------------------------

    def _infer_manipulated_family(self, manipulated: str) -> Optional[str]:
        s = self._canonicalize_label(manipulated)
        best_family = None
        best_hits = 0

        for fam, keywords in self.MANIPULATED_FAMILY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in s)
            if hits > best_hits:
                best_hits = hits
                best_family = fam

        return best_family

    def _infer_scene_family(self, scene: str) -> Optional[str]:
        s = self._canonicalize_label(scene)
        best_family = None
        best_hits = 0

        for fam, keywords in self.SCENE_FAMILY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in s)
            if hits > best_hits:
                best_hits = hits
                best_family = fam

        return best_family

    def _build_family_fallback(
        self,
        manipulated_label: str,
        scene_label: str,
    ) -> Optional[dict[str, Any]]:
        manipulated, scene = self._normalize_key(manipulated_label, scene_label)

        manip_family = self._infer_manipulated_family(manipulated)
        scene_family = self._infer_scene_family(scene)

        if manip_family is None and scene_family is None:
            return None

        receptacle_factor = 0.25 if scene in self.SAFE_RECEPTACLES or scene_family == "receptacle_family" else 1.0

        if manip_family == "liquid":
            attenuation = 1.0
            if scene_family == "receptacle_family":
                attenuation = 0.20
            elif scene_family == "support_family":
                attenuation = 0.50
            elif scene_family == "electronics_family":
                attenuation = 1.0
            elif scene_family == "fragile_family":
                attenuation = 0.85

            return self._attach_meta(
                {
                    "manipulated": manipulated,
                    "scene": scene,
                    "families": ["liquid"] + (["electrical"] if scene_family == "electronics_family" else []),
                    "topology_template": "upward_vertical_cone",
                    "weights": {
                        "w_+x": 0.25 if attenuation > 0.4 else 0.15,
                        "w_-x": 0.25 if attenuation > 0.4 else 0.15,
                        "w_+y": 0.25 if attenuation > 0.4 else 0.15,
                        "w_-y": 0.25 if attenuation > 0.4 else 0.15,
                        "w_+z": 0.95 if attenuation > 0.5 else 0.55,
                        "w_-z": 0.05,
                    },
                    "vertical_rule": "gravity_column",
                    "lateral_decay": "moderate",
                    "receptacle_attenuation": attenuation * receptacle_factor,
                },
                match_type="family",
                raw_manipulated=manipulated_label,
                raw_scene=scene_label,
                score=None,
                resolved_manipulated=manipulated,
                resolved_scene=scene,
            )

        if manip_family == "sharp":
            attenuation = 0.75
            if scene_family == "fragile_family":
                attenuation = 0.95
            elif scene_family == "receptacle_family":
                attenuation = 0.35

            return self._attach_meta(
                {
                    "manipulated": manipulated,
                    "scene": scene,
                    "families": ["sharp"] + (["fragile"] if scene_family == "fragile_family" else []),
                    "topology_template": "forward_directional_cone",
                    "weights": {
                        "w_+x": 0.55,
                        "w_-x": 0.25,
                        "w_+y": 0.55,
                        "w_-y": 0.25,
                        "w_+z": 0.15,
                        "w_-z": 0.10,
                    },
                    "vertical_rule": "none",
                    "lateral_decay": "narrow",
                    "receptacle_attenuation": attenuation,
                },
                match_type="family",
                raw_manipulated=manipulated_label,
                raw_scene=scene_label,
                score=None,
                resolved_manipulated=manipulated,
                resolved_scene=scene,
            )

        if manip_family == "thermal":
            attenuation = 0.75
            if scene_family == "electronics_family":
                attenuation = 0.90
            elif scene_family == "receptacle_family":
                attenuation = 0.35

            return self._attach_meta(
                {
                    "manipulated": manipulated,
                    "scene": scene,
                    "families": ["thermal"],
                    "topology_template": "isotropic_sphere",
                    "weights": {
                        "w_+x": 0.60,
                        "w_-x": 0.60,
                        "w_+y": 0.60,
                        "w_-y": 0.60,
                        "w_+z": 0.60,
                        "w_-z": 0.45,
                    },
                    "vertical_rule": "standard_decay",
                    "lateral_decay": "moderate",
                    "receptacle_attenuation": attenuation,
                },
                match_type="family",
                raw_manipulated=manipulated_label,
                raw_scene=scene_label,
                score=None,
                resolved_manipulated=manipulated,
                resolved_scene=scene,
            )

        if manip_family == "impact":
            attenuation = 0.70
            if scene_family == "fragile_family":
                attenuation = 0.95
            elif scene_family == "receptacle_family":
                attenuation = 0.30

            return self._attach_meta(
                {
                    "manipulated": manipulated,
                    "scene": scene,
                    "families": ["impact"] + (["fragile"] if scene_family == "fragile_family" else []),
                    "topology_template": "isotropic_sphere",
                    "weights": {
                        "w_+x": 0.55,
                        "w_-x": 0.55,
                        "w_+y": 0.55,
                        "w_-y": 0.55,
                        "w_+z": 0.70,
                        "w_-z": 0.25,
                    },
                    "vertical_rule": "standard_decay",
                    "lateral_decay": "moderate",
                    "receptacle_attenuation": attenuation,
                },
                match_type="family",
                raw_manipulated=manipulated_label,
                raw_scene=scene_label,
                score=None,
                resolved_manipulated=manipulated,
                resolved_scene=scene,
            )

        return None

    # ------------------------------------------------------------------
    # Conservative fallback
    # ------------------------------------------------------------------

    def _build_conservative_fallback(self, manipulated_label: str, scene_label: str) -> dict[str, Any]:
        manipulated, scene = self._normalize_key(manipulated_label, scene_label)

        receptacle_factor = 0.3 if scene in self.SAFE_RECEPTACLES else 0.8

        return self._attach_meta(
            {
                "manipulated": manipulated,
                "scene": scene,
                "families": [],
                "topology_template": "isotropic_sphere",
                "weights": {
                    "w_+x": 0.50,
                    "w_-x": 0.50,
                    "w_+y": 0.50,
                    "w_-y": 0.50,
                    "w_+z": 0.50,
                    "w_-z": 0.50,
                },
                "vertical_rule": "standard_decay",
                "lateral_decay": "moderate",
                "receptacle_attenuation": receptacle_factor,
            },
            match_type="conservative",
            raw_manipulated=manipulated_label,
            raw_scene=scene_label,
            score=None,
            resolved_manipulated=manipulated,
            resolved_scene=scene,
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get_risk_parameters(self, manipulated_label: str, scene_label: str) -> dict[str, Any]:
        """
        Hot-path lookup.
        """
        manipulated, scene = self._normalize_key(manipulated_label, scene_label)
        key = (manipulated, scene)

        # Tier 1: exact canonicalized hit
        with self._lock:
            hit = self.knowledge_base.get(key)
        if hit is not None:
            return self._attach_meta(
                hit,
                match_type="exact",
                raw_manipulated=manipulated_label,
                raw_scene=scene_label,
                score=1.0,
                resolved_manipulated=manipulated,
                resolved_scene=scene,
            )

        # Tier 2: nearest
        nn_hit, _ = self._nearest_pair_lookup(manipulated_label, scene_label)
        if nn_hit is not None:
            self._enqueue_llm_pair(manipulated_label, scene_label)
            return nn_hit

        # Tier 3: family fallback
        family_entry = self._build_family_fallback(manipulated_label, scene_label)
        if family_entry is not None:
            self._enqueue_llm_pair(manipulated_label, scene_label)
            return family_entry

        # Tier 4/5: conservative + async LLM
        self._enqueue_llm_pair(manipulated_label, scene_label)
        print(f"[Router] Unknown pair {(manipulated, scene)} -> conservative fallback + async LLM.")
        return self._build_conservative_fallback(manipulated_label, scene_label)

    def prefetch_scene_pairs(self, manipulated_label: str, scene_labels: Sequence[str]) -> None:
        """
        Non-blocking prefetch.
        Call this right after perception to warm the cache for the next frame.
        """
        for scene_label in scene_labels:
            manipulated, scene = self._normalize_key(manipulated_label, scene_label)
            key = (manipulated, scene)

            with self._lock:
                if key in self.knowledge_base:
                    continue

            nn_hit, _ = self._nearest_pair_lookup(manipulated_label, scene_label)
            if nn_hit is not None:
                continue

            self._enqueue_llm_pair(manipulated_label, scene_label)

    # ------------------------------------------------------------------
    # Async LLM batching
    # ------------------------------------------------------------------

    def _enqueue_llm_pair(self, manipulated_label: str, scene_label: str) -> None:
        manipulated, scene = self._normalize_key(manipulated_label, scene_label)
        if not manipulated or not scene:
            return

        key = (manipulated, scene)

        with self._lock:
            if key in self.knowledge_base or key in self.pending_queries:
                return
            self.pending_queries.add(key)

        self.request_queue.put(key)

    def _dispatcher_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not first[0] or not first[1]:
                continue

            batch = [first]
            start = time.time()

            while len(batch) < self.max_batch_size and (time.time() - start) < self.batch_window_s:
                try:
                    item = self.request_queue.get(timeout=0.02)
                    if item[0] and item[1]:
                        batch.append(item)
                except queue.Empty:
                    pass

            grouped: dict[str, list[str]] = defaultdict(list)
            for manipulated, scene in batch:
                grouped[manipulated].append(scene)

            for manipulated, scenes in grouped.items():
                scenes = sorted(set(scenes))
                self.slow_brain_pool.submit(self._run_llm_batch, manipulated, scenes)

    def _run_llm_batch(self, manipulated_label: str, scene_labels: list[str]) -> None:
        keys = [(self._canonicalize_label(manipulated_label), self._canonicalize_label(s)) for s in scene_labels]
        print(f"[Tier 3 Batch] Querying LLM for manipulated='{manipulated_label}' scenes={scene_labels}")

        try:
            results = self._query_llm_batch(manipulated_label, scene_labels)
            normalized_entries = [self._normalize_entry(r) for r in results if isinstance(r, dict)]

            with self._lock:
                for entry in normalized_entries:
                    key = (entry["manipulated"], entry["scene"])
                    self.knowledge_base[key] = entry

                for key in keys:
                    self.pending_queries.discard(key)

                self._rebuild_indices()

            self._rebuild_embedding_cache()

            if self.persist_updates:
                for entry in normalized_entries:
                    self.writer_queue.put(entry)

            print(f"[Tier 3 Batch] Cached {len(normalized_entries)} entries.")

        except Exception as e:
            with self._lock:
                for key in keys:
                    self.pending_queries.discard(key)
            print(f"[Tier 3 Batch] Failed for manipulated='{manipulated_label}' scenes={scene_labels}: {e}")

    def _query_llm_batch(self, manipulated_label: str, scene_labels: list[str]) -> list[dict[str, Any]]:
        """
        Preferred callback signature:
            llm_batch_callback(manipulated_label, scene_labels, system_instruction) -> list[dict]

        Fallback:
            use llm_callback pair-by-pair
        """
        if self.llm_batch_callback is not None:
            raw = self.llm_batch_callback(manipulated_label, scene_labels, self.system_instruction)
            if not isinstance(raw, list):
                raise ValueError("llm_batch_callback must return a list[dict].")
            return [x for x in raw if isinstance(x, dict)]

        if self.llm_callback is not None:
            out: list[dict[str, Any]] = []
            for scene_label in scene_labels:
                raw = self.llm_callback(manipulated_label, scene_label)
                if isinstance(raw, dict):
                    out.append(raw)
            return out

        # Mock fallback for development
        manipulated = self._canonicalize_label(manipulated_label)
        out = []
        for scene_label in scene_labels:
            scene = self._canonicalize_label(scene_label)
            out.append(
                {
                    "manipulated": manipulated,
                    "scene": scene,
                    "families": ["liquid", "electrical"] if "water" in manipulated and "laptop" in scene else ["liquid"],
                    "topology_template": "upward_vertical_cone",
                    "weights": {
                        "w_+x": 0.20,
                        "w_-x": 0.20,
                        "w_+y": 0.20,
                        "w_-y": 0.20,
                        "w_+z": 0.90,
                        "w_-z": 0.00,
                    },
                    "vertical_rule": "gravity_column",
                    "lateral_decay": "moderate",
                    "receptacle_attenuation": 0.70 if scene not in self.SAFE_RECEPTACLES else 0.20,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Meta helpers
    # ------------------------------------------------------------------

    def _attach_meta(
        self,
        entry: dict[str, Any],
        *,
        match_type: str,
        raw_manipulated: str,
        raw_scene: str,
        score: Optional[float],
        resolved_manipulated: str,
        resolved_scene: str,
    ) -> dict[str, Any]:
        out = dict(entry)
        out["_router_meta"] = {
            "match_type": match_type,
            "raw_manipulated": raw_manipulated,
            "raw_scene": raw_scene,
            "resolved_manipulated": resolved_manipulated,
            "resolved_scene": resolved_scene,
            "score": score,
        }
        return out

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._stop_event.set()

        try:
            self.request_queue.put_nowait(("", ""))
        except Exception:
            pass

        if self.persist_updates:
            self.writer_queue.put(None)

        self.slow_brain_pool.shutdown(wait=False)