#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Saliency-based candidate bounding-box detection and filtering.

Pipeline:
    1.  LLM detects salient objects → raw bboxes
    2.  Area filter   (remove < 1% or > 95% image area)
    3.  NMS           (merge overlapping boxes, IoU > threshold)
    4.  Saliency rank (LLM ranks remaining candidates)
    5.  Top-K keep
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from config import (
    Box,
    Config,
    call_llm_vision,
    compute_iou,
    extract_json,
    file_to_data_url,
    with_retry,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateBBox:
    """A detected candidate with saliency metadata."""
    bbox_xyxy: List[int]
    label: str
    saliency_score: float = 0.0
    area_ratio: float = 0.0
    candidate_id: str = ""


# ---------------------------------------------------------------------------
# Step 1 — LLM object detection
# ---------------------------------------------------------------------------

_DETECTION_PROMPT = """\
You are a visual object detector.  Identify ALL visually salient foreground \
objects in this image.  For each object output a tight bounding box.

Return strict JSON only:
{{"detections": [{{"label": "short label", "bbox": [x1, y1, x2, y2]}}]}}

Rules:
1. bbox uses absolute pixel coordinates, format xyxy.
2. One entry per visible object instance.
3. Labels should be short and concrete, e.g. phone, person, car, router, laptop.
4. At most {max_boxes} boxes.
5. Boxes must be tight — do NOT include large background margins.
6. Do not output reasoning or markdown, only JSON.
"""


def _parse_detections(raw_json: Any, img_w: int, img_h: int) -> List[CandidateBBox]:
    """Parse LLM detection output into candidate bboxes."""
    # Handle top-level list (model returned [{...}, ...] directly)
    if isinstance(raw_json, list):
        dets = raw_json
    elif isinstance(raw_json, dict):
        dets = raw_json.get("detections", [])
        if not isinstance(dets, list):
            return []
    else:
        return []

    candidates: List[CandidateBBox] = []
    for item in dets:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "object")).strip()
        # Qwen3.5 uses "bbox_2d" instead of "bbox"
        bbox_raw = item.get("bbox") or item.get("bbox_2d", [])
        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            continue
        try:
            coords = [float(v) for v in bbox_raw]
        except (ValueError, TypeError):
            continue
        # handle Qwen-style 0-1000 relative coords
        if all(0 <= c <= 1000 for c in coords) and max(coords) <= 1000:
            if img_w > 1000 or img_h > 1000:
                coords = [
                    coords[0] / 1000.0 * img_w,
                    coords[1] / 1000.0 * img_h,
                    coords[2] / 1000.0 * img_w,
                    coords[3] / 1000.0 * img_h,
                ]
        box = Box(*coords).normalize().clip(img_w, img_h)
        img_area = max(1, img_w * img_h)
        candidates.append(CandidateBBox(
            bbox_xyxy=box.to_int_list(),
            label=label,
            area_ratio=box.area / img_area,
        ))
    return candidates


def detect_objects(client, cfg: Config, image_path: Path,
                   max_boxes: Optional[int] = None) -> List[CandidateBBox]:
    """Run LLM-based object detection on an image. Never raises."""
    prompt = _DETECTION_PROMPT.format(max_boxes=max_boxes or cfg.max_boxes)
    try:
        raw_text = with_retry(
            lambda: call_llm_vision(client, cfg, prompt, [image_path],
                                     max_tokens=2048),
            retries=cfg.retry_times,
        )
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [saliency] detect_objects API error: {exc}")
        return []

    if cfg.print_raw:
        has_think = "<think>" in raw_text
        print(f"=== SALIENCY DETECT RAW (len={len(raw_text)}, think={has_think}) ===")
        print(raw_text[:300])
        print()
    try:
        payload = extract_json(raw_text)
    except ValueError as exc:
        if cfg.print_raw:
            print(f"  [saliency] JSON parse failed: {exc}")
            print(f"  [saliency] raw first 500: {repr(raw_text[:500])}")
        return []

    try:
        with Image.open(image_path) as img:
            w, h = img.size
        return _parse_detections(payload, w, h)
    except Exception as exc:
        if cfg.print_raw:
            print(f"  [saliency] detection parse error: {exc}")
        return []


# ---------------------------------------------------------------------------
# Step 2 — Area filter
# ---------------------------------------------------------------------------

def filter_by_area(candidates: List[CandidateBBox],
                   cfg: Config,
                   min_area_ratio: Optional[float] = None,
                   max_area_ratio: Optional[float] = None) -> List[CandidateBBox]:
    """Remove boxes that are too small or too large."""
    min_ratio = cfg.min_box_area_ratio if min_area_ratio is None else min_area_ratio
    max_ratio = cfg.max_box_area_ratio if max_area_ratio is None else max_area_ratio
    return [
        c for c in candidates
        if min_ratio <= c.area_ratio <= max_ratio
    ]


# ---------------------------------------------------------------------------
# Step 3 — NMS
# ---------------------------------------------------------------------------

def nms(candidates: List[CandidateBBox],
        iou_threshold: float = 0.65) -> List[CandidateBBox]:
    """Greedy non-maximum suppression sorted by area (larger first)."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: c.area_ratio, reverse=True)
    keep: List[CandidateBBox] = []
    for cand in sorted_cands:
        box_a = Box(*[float(v) for v in cand.bbox_xyxy]).normalize()
        suppressed = False
        for kept in keep:
            box_b = Box(*[float(v) for v in kept.bbox_xyxy]).normalize()
            if compute_iou(box_a, box_b) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(cand)
    return keep


# ---------------------------------------------------------------------------
# Step 4 — Saliency ranking
# ---------------------------------------------------------------------------

_SALIENCY_RANK_PROMPT = """\
You are ranking detected objects by visual saliency (how visually prominent \
and attention-grabbing each object is).

The image contains these detected objects:
{candidate_list}

For each candidate, assign a saliency_score between 0.0 (background clutter) \
and 1.0 (most prominent foreground object).

Return strict JSON only:
{{"scores": [{{"id": "candidate_1", "saliency_score": 0.0}}, ...]}}

Rules:
1. Return only JSON.
2. Every candidate id must appear exactly once.
3. Do not output reasoning.
"""


def rank_by_saliency(client, cfg: Config, image_path: Path,
                      candidates: List[CandidateBBox]) -> List[CandidateBBox]:
    """Use LLM to rank candidates by visual saliency."""
    if not candidates:
        return []
    # assign ids
    for i, c in enumerate(candidates):
        c.candidate_id = f"candidate_{i + 1}"

    desc_lines = []
    for c in candidates:
        desc_lines.append(
            f"  {c.candidate_id}: label={c.label}, bbox={c.bbox_xyxy}"
        )
    candidate_list = "\n".join(desc_lines)
    prompt = _SALIENCY_RANK_PROMPT.format(candidate_list=candidate_list)

    try:
        raw_text = with_retry(
            lambda: call_llm_vision(client, cfg, prompt, [image_path], max_tokens=512),
            retries=cfg.retry_times,
        )
        if cfg.print_raw:
            print("=== SALIENCY RANK RAW ===")
            print(raw_text)
            print()
        payload = extract_json(raw_text)
        scores_list = payload.get("scores", []) if isinstance(payload, dict) else []
        score_map: Dict[str, float] = {}
        for entry in scores_list:
            if isinstance(entry, dict):
                cid = str(entry.get("id", ""))
                sc = entry.get("saliency_score", 0.0)
                if isinstance(sc, (int, float)):
                    score_map[cid] = max(0.0, min(1.0, float(sc)))
        for c in candidates:
            c.saliency_score = score_map.get(c.candidate_id, 0.5)
    except Exception:
        # fallback: rank by area (larger = more salient)
        max_area = max(c.area_ratio for c in candidates) if candidates else 1.0
        for c in candidates:
            c.saliency_score = c.area_ratio / max_area if max_area > 0 else 0.5

    candidates.sort(key=lambda c: c.saliency_score, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Step 5 — Top-K
# ---------------------------------------------------------------------------

def top_k_candidates(candidates: List[CandidateBBox],
                     k: int) -> List[CandidateBBox]:
    """Keep only top-K by saliency score."""
    return candidates[:k]


def _remap_tile_candidates(
    tile_candidates: List[CandidateBBox],
    offset_x: int,
    offset_y: int,
    full_w: int,
    full_h: int,
) -> List[CandidateBBox]:
    """Map tile-local detections back to full-image coordinates."""
    remapped: List[CandidateBBox] = []
    img_area = max(1, full_w * full_h)
    for cand in tile_candidates:
        x1, y1, x2, y2 = cand.bbox_xyxy
        box = Box(
            x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y,
        ).normalize().clip(full_w, full_h)
        remapped.append(CandidateBBox(
            bbox_xyxy=box.to_int_list(),
            label=cand.label,
            area_ratio=box.area / img_area,
        ))
    return remapped


def detect_objects_on_tiles(
    client,
    cfg: Config,
    image_path: Path,
    grid_size: int = 2,
    max_boxes: Optional[int] = None,
) -> List[CandidateBBox]:
    """Run detection on image tiles to recover small or dense objects."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        full_w, full_h = img.size
        tile_w = max(1, full_w // grid_size)
        tile_h = max(1, full_h // grid_size)
        remapped: List[CandidateBBox] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            for row in range(grid_size):
                for col in range(grid_size):
                    x1 = col * tile_w
                    y1 = row * tile_h
                    x2 = full_w if col == grid_size - 1 else (col + 1) * tile_w
                    y2 = full_h if row == grid_size - 1 else (row + 1) * tile_h
                    tile_path = tmp_root / f"tile_{row}_{col}.png"
                    img.crop((x1, y1, x2, y2)).save(tile_path)
                    tile_candidates = detect_objects(
                        client, cfg, tile_path, max_boxes=max_boxes,
                    )
                    remapped.extend(_remap_tile_candidates(
                        tile_candidates, x1, y1, full_w, full_h,
                    ))
    return remapped


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def run_saliency_pipeline(
    client,
    cfg: Config,
    image_path: Path,
    *,
    max_boxes: Optional[int] = None,
    top_k: Optional[int] = None,
    min_area_ratio: Optional[float] = None,
    max_area_ratio: Optional[float] = None,
    use_tiling: bool = False,
    tile_grid_size: int = 2,
) -> List[CandidateBBox]:
    """Full saliency pipeline: detect → filter → NMS → rank → top-K."""
    # Step 1: detect
    raw = detect_objects(client, cfg, image_path, max_boxes=max_boxes)
    if use_tiling:
        raw.extend(detect_objects_on_tiles(
            client, cfg, image_path,
            grid_size=tile_grid_size,
            max_boxes=max_boxes,
        ))
    if cfg.print_raw:
        print(f"  [saliency] {len(raw)} raw detections")

    # Step 2: area filter
    filtered = filter_by_area(
        raw, cfg,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )
    if cfg.print_raw:
        print(f"  [saliency] {len(filtered)} after area filter")

    # Step 3: NMS
    after_nms = nms(filtered, iou_threshold=cfg.nms_iou_threshold)
    if cfg.print_raw:
        print(f"  [saliency] {len(after_nms)} after NMS")

    # Step 4: saliency rank
    ranked = rank_by_saliency(client, cfg, image_path, after_nms)

    # Step 5: top-K
    final = top_k_candidates(ranked, top_k or cfg.saliency_top_k)
    if cfg.print_raw:
        print(f"  [saliency] {len(final)} final candidates")

    return final


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def draw_candidates_on_image(
    image_path: Path,
    candidates: List[CandidateBBox],
    output_path: Path,
    color: str = "lime",
) -> None:
    """Draw candidate bboxes on a copy of the image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for c in candidates:
            x1, y1, x2, y2 = c.bbox_xyxy
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            tag = f"{c.candidate_id} {c.label} ({c.saliency_score:.2f})"
            tag_y = max(0, y1 - 18)
            draw.text((x1 + 2, tag_y), tag, fill=color, font=font)
        img.save(output_path)


def crop_candidate(image_path: Path, bbox_xyxy: List[int],
                   output_path: Path, expand_ratio: float = 0.12) -> Path:
    """Crop a candidate region with slight expansion, save to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        px, py = bw * expand_ratio, bh * expand_ratio
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))
        crop = img.crop((cx1, cy1, cx2, cy2))
        crop.save(output_path)
    return output_path


def render_highlight(image_path: Path, bbox_xyxy: List[int],
                     tag: str, output_path: Path,
                     color: str = "yellow") -> Path:
    """Draw one highlighted bbox with tag on the full image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox_xyxy
        draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        tag_y = max(0, y1 - 22)
        draw.text((x1 + 4, tag_y), tag, fill=color, font=font)
        img.save(output_path)
    return output_path
