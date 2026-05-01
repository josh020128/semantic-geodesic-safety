#!/usr/bin/env python3
"""
Merge Wavefront OBJ meshes listed in a semantic hierarchy JSON (e.g. result_after_merging.json).

The JSON is a tree: each node may have ``objs`` (list of stems without ``.obj``) and ``children``.
All unique stems are collected in depth-first order, then each ``textured_objs/<stem>.obj`` is
merged into one OBJ with consistent vertex indexing and a single combined MTL (material names
prefixed by stem to avoid collisions). Texture paths ``../images/...`` from per-part MTLs are
rewritten to ``images/...`` relative to the output directory (sibling layout to ``textured_objs``).

Example (laptop asset)::

    python scripts/merge_asset_textured_objs.py \\
        --asset-dir data/assets/laptop \\
        --json result_after_merging.json \\
        --out merged.obj
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

# Parsed OBJ: each face is a list of corners (vi, vti, vni) 0-based; segments group by usemtl.
FaceCorner = tuple[int, int | None, int | None]
Face = list[FaceCorner]
FaceSegment = tuple[str | None, list[Face]]


def collect_obj_stems_dfs(node: dict | list) -> list[str]:
    """Depth-first traversal; ``objs`` on a node appear before recursing into ``children``."""
    out: list[str] = []

    def walk(n: dict) -> None:
        for stem in n.get("objs", []):
            out.append(stem)
        for child in n.get("children", []):
            walk(child)

    if isinstance(node, list):
        for item in node:
            walk(item)
    else:
        walk(node)
    return list(OrderedDict.fromkeys(out))


def _face_vertex_indices(tok: str) -> tuple[int | None, int | None, int | None]:
    """Parse one corner of ``f v/vt/vn``; returns 0-based indices (OBJ file is 1-based)."""
    parts = tok.split("/")
    while len(parts) < 3:
        parts.append("")

    def to_idx(s: str) -> int | None:
        if not s:
            return None
        i = int(s)
        if i < 0:
            raise ValueError(f"Negative OBJ index not supported in merge: {tok!r}")
        return i - 1

    vi = to_idx(parts[0]) if parts[0] else None
    vti = to_idx(parts[1]) if len(parts) > 1 and parts[1] else None
    vni = to_idx(parts[2]) if len(parts) > 2 and parts[2] else None
    return vi, vti, vni


def parse_obj(path: Path) -> tuple[list[str], list[str], list[str], list[FaceSegment]]:
    """
    Returns (v_lines, vt_lines, vn_lines, segments) where each segment is
    (usemtl_name_or_None, faces) and each face is a list of (vi, vti, vni) 0-based.
    """
    v_lines: list[str] = []
    vt_lines: list[str] = []
    vn_lines: list[str] = []
    segments: list[FaceSegment] = []

    current_mtl: str | None = None
    current_faces: list[list[tuple[int, int | None, int | None]]] = []

    def flush_segment() -> None:
        nonlocal current_faces
        if not current_faces:
            return
        segments.append((current_mtl, current_faces))
        current_faces = []

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("v "):
            v_lines.append(line[2:].strip())
        elif line.startswith("vt "):
            vt_lines.append(line[3:].strip())
        elif line.startswith("vn "):
            vn_lines.append(line[3:].strip())
        elif line.startswith("usemtl "):
            flush_segment()
            current_mtl = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            toks = line.split()[1:]
            face: list[tuple[int, int | None, int | None]] = []
            for t in toks:
                vi, vti, vni = _face_vertex_indices(t)
                if vi is None:
                    raise ValueError(f"Bad face corner in {path}: {line!r}")
                face.append((vi, vti, vni))
            current_faces.append(face)
        # ignore mtllib, o, g, s, etc.

    flush_segment()
    return v_lines, vt_lines, vn_lines, segments


def _format_face_corner(
    vi: int,
    vti: int | None,
    vni: int | None,
    v_off: int,
    vt_off: int,
    vn_off: int,
) -> str:
    """Emit one OBJ ``f`` corner (1-based indices after merge offsets)."""
    a = vi + v_off + 1
    if vti is not None and vni is not None:
        return f"{a}/{vti + vt_off + 1}/{vni + vn_off + 1}"
    if vti is not None and vni is None:
        return f"{a}/{vti + vt_off + 1}"
    if vti is None and vni is not None:
        return f"{a}//{vni + vn_off + 1}"
    return str(a)


def rewrite_mtl_paths(line: str) -> str:
    """``../images/foo`` -> ``images/foo`` when merged MTL lives in asset root."""
    stripped = line.strip()
    for prefix in ("map_Kd ", "map_Ka ", "map_Ks ", "map_Ke ", "map_d ", "map_bump ", "bump ", "disp "):
        if stripped.startswith(prefix):
            rest = stripped[len(prefix) :].strip()
            if rest.startswith("../images/"):
                rest = "images/" + rest[len("../images/") :]
            return prefix + rest
    return line.rstrip("\n")


def merge_mtl_for_stem(mtl_path: Path, stem: str) -> str:
    """Prefix ``newmtl`` names with ``stem__``; fix texture paths."""
    body = mtl_path.read_text(encoding="utf-8", errors="replace")
    out_lines: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("newmtl "):
            name = s.split(None, 1)[1].strip()
            out_lines.append(f"newmtl {stem}__{name}")
        else:
            out_lines.append(rewrite_mtl_paths(line))
    return "\n".join(out_lines).rstrip() + "\n"


def merge_asset(
    asset_dir: Path,
    json_name: str,
    textured_subdir: str,
    out_obj: Path,
    out_mtl: Path,
) -> None:
    json_path = asset_dir / json_name
    textured = asset_dir / textured_subdir
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stems = collect_obj_stems_dfs(data)
    if not stems:
        raise SystemExit(f"No object stems found in {json_path}")

    all_v: list[str] = []
    all_vt: list[str] = []
    all_vn: list[str] = []
    mtl_chunks: list[str] = []

    obj_header = "\n".join(
        [
            "# Merged OBJ from JSON + textured_objs/",
            f"# Source: {json_path.name}",
            f"# Parts: {len(stems)}",
            f"mtllib {out_mtl.name}",
            "",
        ]
    )
    obj_body: list[str] = []

    for stem in stems:
        obj_path = textured / f"{stem}.obj"
        if not obj_path.is_file():
            raise FileNotFoundError(f"Missing mesh file: {obj_path}")
        v_lines, vt_lines, vn_lines, segments = parse_obj(obj_path)
        v_off = len(all_v)
        vt_off = len(all_vt)
        vn_off = len(all_vn)

        all_v.extend(v_lines)
        all_vt.extend(vt_lines)
        all_vn.extend(vn_lines)

        mtl_name = None
        for line in obj_path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("mtllib "):
                mtl_name = s.split(None, 1)[1].strip()
                break
        if mtl_name:
            mtl_path = textured / mtl_name
            if mtl_path.is_file():
                mtl_chunks.append(f"# --- {stem} ({mtl_name}) ---\n")
                mtl_chunks.append(merge_mtl_for_stem(mtl_path, stem))
            else:
                mtl_chunks.append(f"# --- {stem}: missing {mtl_path} ---\n")

        obj_body.append(f"g {stem}\n")
        for mtl, faces in segments:
            if mtl is not None:
                obj_body.append(f"usemtl {stem}__{mtl}\n")
            for face in faces:
                corners = []
                for vi, vti, vni in face:
                    corners.append(_format_face_corner(vi, vti, vni, v_off, vt_off, vn_off))
                obj_body.append("f " + " ".join(corners) + "\n")

    out_mtl.parent.mkdir(parents=True, exist_ok=True)
    out_mtl.write_text("".join(mtl_chunks), encoding="utf-8")

    lines_out = (
        obj_header
        + "\n".join(f"v {s}" for s in all_v)
        + "\n"
        + ("\n".join(f"vt {s}" for s in all_vt) + "\n" if all_vt else "")
        + ("\n".join(f"vn {s}" for s in all_vn) + "\n" if all_vn else "")
        + "".join(obj_body)
    )
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    out_obj.write_text(lines_out, encoding="utf-8")

    print(f"Wrote {out_obj} ({len(all_v)} vertices, {len(stems)} parts)")
    print(f"Wrote {out_mtl}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--asset-dir",
        type=Path,
        default=Path("data/assets/bottle"),
        help="Directory containing the JSON and textured_objs/",
    )
    p.add_argument(
        "--json",
        default="result_after_merging.json",
        help="JSON filename inside asset-dir",
    )
    p.add_argument(
        "--textured-subdir",
        default="textured_objs",
        help="Subdirectory with <stem>.obj files",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output OBJ path (default: asset-dir/merged.obj)",
    )
    p.add_argument(
        "--out-mtl",
        type=Path,
        default=None,
        help="Output MTL path (default: next to --out as merged.mtl)",
    )
    args = p.parse_args()
    asset_dir = args.asset_dir.resolve()
    out_obj = (args.out or (asset_dir / "merged.obj")).resolve()
    out_mtl = (args.out_mtl or (out_obj.with_name(out_obj.stem + ".mtl"))).resolve()
    merge_asset(asset_dir, args.json, args.textured_subdir, out_obj, out_mtl)


if __name__ == "__main__":
    main()
