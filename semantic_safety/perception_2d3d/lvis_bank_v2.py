from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional


# ---------------------------------------------------------------------
# Small robotics / tabletop extensions that are useful even if the
# official LVIS categories are unavailable locally.
# ---------------------------------------------------------------------

DEFAULT_CUSTOM_LABELS = [
    "power drill",
    "hot soldering iron",
    "bleach cleanser",
    "shelf",
    "table",
    "desk",
    "counter",
    "bowl",
    "mug",
    "laptop",
    "cup of water",
    "soccer ball",
]

DEFAULT_ALIASES = {
    # drill variants
    "drill": "power drill",
    "electric drill": "power drill",
    "corded drill": "power drill",
    "cordless drill": "power drill",
    "hand drill": "power drill",

    # soldering variants
    "soldering iron": "hot soldering iron",
    "hot iron": "hot soldering iron",
    "solder iron": "hot soldering iron",

    # bleach variants
    "bleach": "bleach cleanser",
    "bleach cleaner": "bleach cleanser",
    "cleanser": "bleach cleanser",

    # structural variants
    "partition": "wall",
    "panel wall": "wall",
    "partition wall": "wall",

    # support variants
    "workbench": "table",
    "countertop": "counter",
}


def _normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


@dataclass(frozen=True)
class LabelEntryV2:
    label: str
    canonical_label: str
    source: str  # e.g. "lvis", "lvis_synonym", "custom"


class LVISBankV2:
    """
    Canonical object-name bank for SigLIP2-based open-vocabulary labeling.

    Main responsibilities
    ---------------------
    - Load official LVIS category names from a local LVIS annotation JSON.
    - Optionally include LVIS synonyms as extra text-bank entries.
    - Merge custom robotics / tabletop labels.
    - Canonicalize aliases (e.g. 'electric drill' -> 'power drill').
    - Produce a clean text bank for SigLIP2 / OpenCLIP scoring.

    Notes
    -----
    - This module does NOT depend on any specific model.
    - It is meant to be used by SigLIP2LabelerV2 later.
    - If no LVIS JSON is available, it still works using only custom labels.
    """

    def __init__(
        self,
        entries: list[LabelEntryV2],
        alias_to_canonical: Optional[dict[str, str]] = None,
    ) -> None:
        self.entries = entries

        alias_to_canonical = alias_to_canonical or {}
        self.alias_to_canonical: dict[str, str] = {
            _normalize_label(k): _normalize_label(v)
            for k, v in alias_to_canonical.items()
        }

        # Make sure every explicit entry label resolves to its canonical form.
        for e in self.entries:
            self.alias_to_canonical[_normalize_label(e.label)] = _normalize_label(e.canonical_label)

        # Ordered unique entry labels for scoring bank
        seen = set()
        self.text_bank_labels: list[str] = []
        for e in self.entries:
            lbl = _normalize_label(e.label)
            if lbl not in seen:
                seen.add(lbl)
                self.text_bank_labels.append(lbl)

        # Ordered unique canonical labels
        seen_canon = set()
        self.canonical_labels: list[str] = []
        for e in self.entries:
            canon = _normalize_label(e.canonical_label)
            if canon not in seen_canon:
                seen_canon.add(canon)
                self.canonical_labels.append(canon)

        # Reverse map: canonical -> aliases in bank
        self.canonical_to_aliases: dict[str, list[str]] = {}
        for e in self.entries:
            canon = _normalize_label(e.canonical_label)
            lbl = _normalize_label(e.label)
            self.canonical_to_aliases.setdefault(canon, [])
            if lbl not in self.canonical_to_aliases[canon]:
                self.canonical_to_aliases[canon].append(lbl)

    # -----------------------------------------------------------------
    # Construction helpers
    # -----------------------------------------------------------------

    @classmethod
    def from_lvis_json(
        cls,
        lvis_json_path: str,
        *,
        include_synonyms: bool = True,
        custom_labels: Optional[Iterable[str]] = None,
        custom_aliases: Optional[dict[str, str]] = None,
    ) -> "LVISBankV2":
        """
        Build a label bank from an LVIS annotation JSON.

        Expected LVIS JSON structure:
          {
            "categories": [
              {"id": 1, "name": "...", "synonyms": ["...", ...]},
              ...
            ]
          }
        """
        if not os.path.exists(lvis_json_path):
            raise FileNotFoundError(f"LVIS JSON not found: {lvis_json_path}")

        with open(lvis_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", None)
        if categories is None or not isinstance(categories, list):
            raise ValueError(
                f"Expected 'categories' list in LVIS JSON: {lvis_json_path}"
            )

        entries: list[LabelEntryV2] = []
        for cat in categories:
            if not isinstance(cat, dict):
                continue

            name = cat.get("name", None)
            if not name:
                continue

            canon = _normalize_label(name)
            entries.append(
                LabelEntryV2(
                    label=canon,
                    canonical_label=canon,
                    source="lvis",
                )
            )

            if include_synonyms:
                synonyms = cat.get("synonyms", [])
                if isinstance(synonyms, list):
                    for s in synonyms:
                        s_norm = _normalize_label(s)
                        if not s_norm or s_norm == canon:
                            continue
                        entries.append(
                            LabelEntryV2(
                                label=s_norm,
                                canonical_label=canon,
                                source="lvis_synonym",
                            )
                        )

        return cls.from_entries(
            entries,
            custom_labels=custom_labels,
            custom_aliases=custom_aliases,
        )

    @classmethod
    def from_label_list(
        cls,
        labels: Iterable[str],
        *,
        custom_labels: Optional[Iterable[str]] = None,
        custom_aliases: Optional[dict[str, str]] = None,
    ) -> "LVISBankV2":
        entries = [
            LabelEntryV2(
                label=_normalize_label(lbl),
                canonical_label=_normalize_label(lbl),
                source="label_list",
            )
            for lbl in labels
            if str(lbl).strip()
        ]
        return cls.from_entries(
            entries,
            custom_labels=custom_labels,
            custom_aliases=custom_aliases,
        )

    @classmethod
    def from_entries(
        cls,
        base_entries: Iterable[LabelEntryV2],
        *,
        custom_labels: Optional[Iterable[str]] = None,
        custom_aliases: Optional[dict[str, str]] = None,
    ) -> "LVISBankV2":
        entries = list(base_entries)

        # Add default custom labels first
        for lbl in DEFAULT_CUSTOM_LABELS:
            norm = _normalize_label(lbl)
            entries.append(
                LabelEntryV2(
                    label=norm,
                    canonical_label=norm,
                    source="custom_default",
                )
            )

        # Add user custom labels
        if custom_labels is not None:
            for lbl in custom_labels:
                norm = _normalize_label(lbl)
                if not norm:
                    continue
                entries.append(
                    LabelEntryV2(
                        label=norm,
                        canonical_label=norm,
                        source="custom_user",
                    )
                )

        # Deduplicate identical (label, canonical, source) triples conservatively
        deduped: list[LabelEntryV2] = []
        seen = set()
        for e in entries:
            key = (_normalize_label(e.label), _normalize_label(e.canonical_label), e.source)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                LabelEntryV2(
                    label=_normalize_label(e.label),
                    canonical_label=_normalize_label(e.canonical_label),
                    source=e.source,
                )
            )

        alias_map = dict(DEFAULT_ALIASES)
        if custom_aliases is not None:
            alias_map.update(custom_aliases)

        return cls(deduped, alias_to_canonical=alias_map)

    @classmethod
    def build_default(
        cls,
        *,
        lvis_json_path: Optional[str] = None,
        include_synonyms: bool = True,
        custom_labels: Optional[Iterable[str]] = None,
        custom_aliases: Optional[dict[str, str]] = None,
    ) -> "LVISBankV2":
        """
        Preferred constructor.

        Behavior:
        - If lvis_json_path is given and exists: load official LVIS categories.
        - Otherwise: fall back to custom/tabletop labels only.
        """
        if lvis_json_path is not None and os.path.exists(lvis_json_path):
            return cls.from_lvis_json(
                lvis_json_path,
                include_synonyms=include_synonyms,
                custom_labels=custom_labels,
                custom_aliases=custom_aliases,
            )

        # Fallback: use only custom labels.
        return cls.from_entries(
            [],
            custom_labels=custom_labels,
            custom_aliases=custom_aliases,
        )

    # -----------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------

    def canonicalize(self, label: str) -> str:
        """
        Map a possibly non-canonical label to canonical form.
        If not found, returns normalized label itself.
        """
        norm = _normalize_label(label)
        return self.alias_to_canonical.get(norm, norm)

    def is_known_label(self, label: str) -> bool:
        norm = _normalize_label(label)
        return norm in self.text_bank_labels or norm in self.alias_to_canonical

    def get_text_bank(
        self,
        *,
        canonical_only: bool = False,
        prepend_photo_prompt: bool = False,
    ) -> list[str]:
        """
        Return text bank strings for SigLIP2/OpenCLIP scoring.
        """
        labels = self.canonical_labels if canonical_only else self.text_bank_labels
        if prepend_photo_prompt:
            return [f"a photo of {x}" for x in labels]
        return list(labels)

    def get_aliases_for_canonical(self, canonical_label: str) -> list[str]:
        canon = self.canonicalize(canonical_label)
        return list(self.canonical_to_aliases.get(canon, [canon]))

    def describe(self) -> dict:
        source_counts: dict[str, int] = {}
        for e in self.entries:
            source_counts[e.source] = source_counts.get(e.source, 0) + 1

        return {
            "num_entries": len(self.entries),
            "num_text_bank_labels": len(self.text_bank_labels),
            "num_canonical_labels": len(self.canonical_labels),
            "num_aliases": len(self.alias_to_canonical),
            "source_counts": source_counts,
        }

    def save_text_bank_txt(
        self,
        path: str,
        *,
        canonical_only: bool = False,
    ) -> None:
        labels = self.canonical_labels if canonical_only else self.text_bank_labels
        with open(path, "w", encoding="utf-8") as f:
            for lbl in labels:
                f.write(lbl + "\n")

    def save_debug_json(self, path: str) -> None:
        payload = {
            "entries": [
                {
                    "label": e.label,
                    "canonical_label": e.canonical_label,
                    "source": e.source,
                }
                for e in self.entries
            ],
            "alias_to_canonical": self.alias_to_canonical,
            "describe": self.describe(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Optional convenience helpers
# ---------------------------------------------------------------------

def load_custom_labels_from_txt(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    labels: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                labels.append(s)
    return labels


def load_aliases_from_json(path: str) -> dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON for aliases: {path}")

    return {str(k): str(v) for k, v in data.items()}