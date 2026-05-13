#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command inference and evaluation for WebEyes.

The grounding path intentionally follows the stronger reference baseline under
``C:\\baidunetdiskdownload\\BaiduSyncdisk\\test``:
  - ask for a ``detections`` JSON schema rather than a bare bbox,
  - use model-family-aware coordinate normalization (Gemini/Qwen -> rel1000),
  - parse JSON and text bboxes defensively,
  - optionally filter tiny/huge boxes and apply label-wise NMS before selecting one box.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

import evaluate as ev


CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent


def default_dataset_root() -> Path:
    return CODE_ROOT / "dataset"


DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "outputs" / "eval"


@dataclass
class ApiConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout: float = 120.0
    workers: int = 4
    retry_times: int = 2
    max_boxes: int = 20
    min_box_area_ratio: float = 0.0
    max_box_area_ratio: float = 1.0
    nms_iou_threshold: float = 0.65
    enable_nms: bool = True


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized(self) -> "Box":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return Box(x1=x1, y1=y1, x2=x2, y2=y2)

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0.0, min(float(max(width - 1, 0)), self.x1)),
            y1=max(0.0, min(float(max(height - 1, 0)), self.y1)),
            x2=max(0.0, min(float(max(width - 1, 0)), self.x2)),
            y2=max(0.0, min(float(max(height - 1, 0)), self.y2)),
        )

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def as_int_xyxy(self) -> List[int]:
        return [
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        ]


@dataclass
class Detection:
    label: str
    bbox_xyxy: List[int]
    source_bbox: List[float]
    source_coord_mode: str
    saliency_rank: Optional[int] = None
    score: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenAI-compatible WebEyes grounding/VQA inference and evaluate results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["all", "search-grounding", "search-vqa"],
        help="Run grounding, VQA, or both.",
    )
    parser.add_argument("--env-file", default=".env", help="Optional .env file with OPENAI_* settings.")
    parser.add_argument("--api-key", help="Override OPENAI_API_KEY.")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL.")
    parser.add_argument("--model", help="Override OPENAI_MODEL.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-times", type=int)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root directory.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to Pixel-Searcher/outputs/eval/<model>.")
    parser.add_argument("--grounding-jsonl", type=Path, help="Grounding task JSONL. Overrides --dataset-root.")
    parser.add_argument("--vqa-jsonl", type=Path, help="VQA task JSONL. Overrides --dataset-root.")
    parser.add_argument("--seg-jsonl", type=Path, help="Segmentation task JSONL used when --sam3 is enabled. Overrides --dataset-root.")
    parser.add_argument("--sample-id", help="Only process one sample_id.")
    parser.add_argument("--qa-id", help="Only process one qa_id.")
    parser.add_argument("--limit", type=int, help="Only process first N records after filtering.")
    parser.add_argument(
        "--coord-mode",
        choices=["auto", "abs", "rel1000"],
        default="rel1000",
        help="Coordinate mode for grounding bbox output.",
    )
    parser.add_argument("--max-boxes", type=int, help="Maximum parsed grounding boxes before selecting the top one.")
    parser.add_argument("--min-box-area-ratio", type=float, help="Drop grounding boxes smaller than this image-area ratio.")
    parser.add_argument("--max-box-area-ratio", type=float, help="Drop grounding boxes larger than this image-area ratio.")
    parser.add_argument("--nms-iou-threshold", type=float, help="Label-wise NMS IoU threshold for grounding boxes.")
    parser.add_argument("--no-nms", action="store_true", help="Disable grounding NMS.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw model responses.")
    parser.add_argument("--no-evaluate", action="store_true", help="Only write prediction JSONL files.")
    parser.add_argument("--sam3", action="store_true", help="Convert grounding bboxes to masks with SAM3 and evaluate search-seg.")
    parser.add_argument("--sam-python", default=sys.executable, help="Python executable that can import sam3.")
    parser.add_argument("--sam-worker", type=Path, default=CURRENT_DIR / "sam3_worker.py")
    parser.add_argument("--sam-checkpoint", default="", help="SAM3 checkpoint path.")
    parser.add_argument("--sam-device", default="cuda", help="SAM3 device, e.g. cuda or cpu.")
    return parser.parse_args()


def parse_env_file(path: str) -> dict[str, str]:
    env: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        p = CURRENT_DIR / path
    if not p.exists():
        return env
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def env_float(env: dict[str, str], key: str, default: float) -> float:
    value = env.get(key)
    return default if value in {None, ""} else float(str(value))


def env_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    return default if value in {None, ""} else int(str(value))


def load_api_config(args: argparse.Namespace) -> ApiConfig:
    env = {**os.environ, **parse_env_file(args.env_file)}
    api_key = args.api_key or env.get("OPENAI_API_KEY", "")
    base_url = args.base_url or env.get("OPENAI_BASE_URL", "")
    model = args.model or env.get("OPENAI_MODEL", "")
    missing = [name for name, value in {
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_MODEL": model,
    }.items() if not str(value).strip()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}. Provide them in --env-file or CLI args.")
    return ApiConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=args.temperature if args.temperature is not None else env_float(env, "TEMPERATURE", 0.0),
        top_p=args.top_p if args.top_p is not None else env_float(env, "TOP_P", 1.0),
        max_tokens=args.max_tokens if args.max_tokens is not None else env_int(env, "MAX_TOKENS", 1024),
        timeout=args.timeout if args.timeout is not None else env_float(env, "TIMEOUT", env_float(env, "REQUEST_TIMEOUT", 120.0)),
        workers=max(1, args.workers if args.workers is not None else env_int(env, "MAX_WORKERS", 4)),
        retry_times=max(0, args.retry_times if args.retry_times is not None else env_int(env, "RETRY_TIMES", 2)),
        max_boxes=max(1, args.max_boxes if args.max_boxes is not None else env_int(env, "MAX_BOXES", 20)),
        min_box_area_ratio=(
            args.min_box_area_ratio
            if args.min_box_area_ratio is not None
            else env_float(env, "MIN_BOX_AREA_RATIO", 0.0)
        ),
        max_box_area_ratio=(
            args.max_box_area_ratio
            if args.max_box_area_ratio is not None
            else env_float(env, "MAX_BOX_AREA_RATIO", 1.0)
        ),
        nms_iou_threshold=(
            args.nms_iou_threshold
            if args.nms_iou_threshold is not None
            else env_float(env, "NMS_IOU_THRESHOLD", 0.65)
        ),
        enable_nms=not args.no_nms,
    )


def build_client(cfg: ApiConfig) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "model"


def file_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def extract_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text).strip()
    try:
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif item.get("text") is not None:
                        parts.append(str(item.get("text", "")))
                elif getattr(item, "text", None):
                    parts.append(str(item.text))
            text = "\n".join(part.strip() for part in parts if part).strip()
            if text:
                return text
        reasoning = getattr(message, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    except Exception:
        pass
    return str(response).strip()


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        before, after = cleaned.split("<think>", 1)
        json_start = next((i for i, ch in enumerate(after) if ch in "[{"), -1)
        cleaned = (before + " " + after[json_start:]).strip() if json_start >= 0 else before.strip()

    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    else:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
            return payload
        except Exception:
            continue
    raise ValueError("No valid JSON payload found")


def image_path_from_record(record: Dict[str, Any]) -> Path:
    image = record.get("image", {}) if isinstance(record.get("image"), dict) else {}
    path = Path(str(image.get("path") or image.get("rel_path") or ""))
    if not path.exists() and image.get("rel_path"):
        path = (DEFAULT_DATASET_ROOT / str(image.get("rel_path"))).resolve()
    return path


def image_size(record: Dict[str, Any]) -> Tuple[int, int]:
    with Image.open(image_path_from_record(record)) as img:
        return img.size


def normalize_label(label: str) -> str:
    text = re.sub(r"\s+", " ", label.strip().lower())
    text = re.sub(r"[^a-z0-9 _-]", "", text)
    aliases = {
        "people": "person",
        "persons": "person",
        "men": "person",
        "women": "person",
        "phones": "phone",
        "smartphones": "phone",
        "mobile phone": "phone",
        "mobile phones": "phone",
        "cars": "car",
        "vehicles": "vehicle",
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("s") and len(text) > 3:
        return text[:-1]
    return text or "object"


def resolve_coord_mode(model_name: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lower = model_name.strip().lower()
    if any(name in lower for name in ("gemini", "qwen", "qwq")):
        return "rel1000"
    return "abs"


def rel1000_to_abs_xyxy(bbox: Sequence[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = bbox
    return Box(
        x1=float(x1) / 1000.0 * width,
        y1=float(y1) / 1000.0 * height,
        x2=float(x2) / 1000.0 * width,
        y2=float(y2) / 1000.0 * height,
    ).normalized().clip(width, height)


def abs_to_box(bbox: Sequence[float], width: int, height: int) -> Box:
    return Box(
        x1=float(bbox[0]),
        y1=float(bbox[1]),
        x2=float(bbox[2]),
        y2=float(bbox[3]),
    ).normalized().clip(width, height)


def convert_box(bbox: Sequence[float], coord_mode: str, width: int, height: int) -> Box:
    if coord_mode == "rel1000":
        return rel1000_to_abs_xyxy(bbox, width=width, height=height)
    return abs_to_box(bbox, width=width, height=height)


def box_iou(box_a: Box, box_b: Box) -> float:
    inter_x1 = max(box_a.x1, box_b.x1)
    inter_y1 = max(box_a.y1, box_b.y1)
    inter_x2 = min(box_a.x2, box_b.x2)
    inter_y2 = min(box_a.y2, box_b.y2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    union = box_a.area() + box_b.area() - inter
    return inter / union if union > 0.0 else 0.0


def filter_and_normalize_detections(
    items: Iterable[Dict[str, Any]],
    coord_mode: str,
    width: int,
    height: int,
    cfg: ApiConfig,
) -> List[Detection]:
    image_area = float(width * height)
    detections: List[Detection] = []
    for item in items:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            source_bbox = [float(v) for v in bbox]
        except Exception:
            continue

        box = convert_box(source_bbox, coord_mode=coord_mode, width=width, height=height)
        area_ratio = box.area() / image_area if image_area > 0 else 0.0
        if box.width() < 2 or box.height() < 2:
            continue
        if area_ratio < cfg.min_box_area_ratio or area_ratio > cfg.max_box_area_ratio:
            continue

        rank = item.get("saliency_rank")
        score = item.get("score")
        detections.append(
            Detection(
                label=normalize_label(str(item.get("label", "object"))),
                bbox_xyxy=box.as_int_xyxy(),
                source_bbox=source_bbox,
                source_coord_mode=coord_mode,
                saliency_rank=int(rank) if isinstance(rank, (int, float)) else None,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )

    detections.sort(
        key=lambda det: (
            det.saliency_rank if det.saliency_rank is not None else 10**9,
            -(det.score if det.score is not None else 0.0),
            -((det.bbox_xyxy[2] - det.bbox_xyxy[0]) * (det.bbox_xyxy[3] - det.bbox_xyxy[1])),
        )
    )
    return detections[: cfg.max_boxes]


def apply_nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    kept: List[Detection] = []
    grouped: Dict[str, List[Detection]] = {}
    for det in detections:
        grouped.setdefault(det.label, []).append(det)

    for group in grouped.values():
        remaining = list(group)
        remaining.sort(
            key=lambda det: (
                det.saliency_rank if det.saliency_rank is not None else 10**9,
                -(det.score if det.score is not None else 0.0),
            )
        )
        while remaining:
            current = remaining.pop(0)
            kept.append(current)
            current_box = Box(*current.bbox_xyxy)
            remaining = [
                candidate
                for candidate in remaining
                if box_iou(current_box, Box(*candidate.bbox_xyxy)) < iou_threshold
            ]

    kept.sort(
        key=lambda det: (
            det.saliency_rank if det.saliency_rank is not None else 10**9,
            det.label,
            det.bbox_xyxy[0],
            det.bbox_xyxy[1],
        )
    )
    return kept


def payload_to_detection_items(payload: Any) -> List[Dict[str, Any]]:
    def one_item(item: Dict[str, Any]) -> Dict[str, Any]:
        bbox = item.get("bbox") or item.get("bbox_2d") or item.get("bbox_xyxy") or item.get("box")
        return {**item, "bbox": bbox}

    if isinstance(payload, dict):
        raw_bboxes = payload.get("bboxes")
        raw_label = payload.get("label", "object")
        if isinstance(raw_bboxes, list):
            items = [
                {"label": raw_label, "bbox": bbox}
                for bbox in raw_bboxes
                if isinstance(bbox, list) and len(bbox) == 4
            ]
            if items:
                return items
        detections = payload.get("detections")
        if isinstance(detections, list):
            return [one_item(det) for det in detections if isinstance(det, dict)]
        for key in ("predicted_bbox", "bbox", "bbox_2d", "bbox_xyxy", "box"):
            bbox = payload.get(key)
            if isinstance(bbox, list) and len(bbox) == 4:
                return [{"label": payload.get("label", "object"), "bbox": bbox, "score": payload.get("score")}]
    if isinstance(payload, list):
        if len(payload) == 4 and all(isinstance(value, (int, float, str)) for value in payload):
            return [{"label": "object", "bbox": payload}]
        items: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                items.append(one_item(item))
            elif isinstance(item, list) and len(item) == 4:
                items.append({"label": "object", "bbox": item})
        return items
    return []


def heuristic_parse_detection_text(text: str) -> List[Dict[str, Any]]:
    patterns = [
        r"(?:predicted_bbox|bbox|box|bbox_xyxy|bbox_2d)\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:\])?",
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*(?:\])?",
    ]
    items: List[Dict[str, Any]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            items.append(
                {
                    "label": "object",
                    "bbox": [float(match.group(i)) for i in range(1, 5)],
                }
            )
        if items:
            break
    return items


def parse_raw_detections(raw_text: str) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if raw_text.strip() == "None":
        return [], "none", None
    try:
        payload = extract_json_payload(raw_text)
        items = payload_to_detection_items(payload)
        if items:
            return items, "json", None
    except Exception as exc:
        json_error = str(exc)
    else:
        json_error = "JSON payload did not contain bbox detections"

    items = heuristic_parse_detection_text(raw_text)
    if items:
        return items, "text_bbox", None
    return [], "failed", json_error


def normalize_detections_with_fallback(
    parsed_items: Sequence[Dict[str, Any]],
    parse_mode: str,
    primary_coord_mode: str,
    model_name: str,
    width: int,
    height: int,
    cfg: ApiConfig,
) -> Tuple[List[Detection], str]:
    coord_modes = [primary_coord_mode]
    lower_model = model_name.strip().lower()
    if parse_mode == "text_bbox" and "gemini" not in lower_model and primary_coord_mode != "abs":
        coord_modes = ["abs", primary_coord_mode]
    elif primary_coord_mode == "rel1000":
        coord_modes = ["rel1000", "abs"]

    best_detections: List[Detection] = []
    best_mode = coord_modes[0]
    for coord_mode in coord_modes:
        detections = filter_and_normalize_detections(
            items=parsed_items,
            coord_mode=coord_mode,
            width=width,
            height=height,
            cfg=cfg,
        )
        if cfg.enable_nms:
            detections = apply_nms(detections, iou_threshold=cfg.nms_iou_threshold)
        if len(detections) > len(best_detections):
            best_detections = detections
            best_mode = coord_mode
        elif len(detections) == len(best_detections) and detections and coord_mode == primary_coord_mode:
            best_detections = detections
            best_mode = coord_mode
    return best_detections, best_mode


def grounding_system_prompt() -> str:
    return (
        "You are a pure visual multimodal model under evaluation. "
        "Answer only with information grounded in the provided image. "
        "Do not use internet access, web search, or external lookup. "
        "Return only the requested JSON and nothing else. "
        "ONLY ONE object. DO NOT GIVE MULTIPLE OBJECTS. "
        "Return JSON:\n"
        "{"
        '"detections": ['
        '{"label": "string", "bbox": [x1, y1, x2, y2], "score": 0.0}'
        "], "
        '"format": "xyxy"'
        "}\n"
    )


def grounding_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_path = image_path_from_record(record)
    return [
        {"role": "system", "content": grounding_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": str(record.get("question_en") or record.get("question") or "").strip()},
                {"type": "image_url", "image_url": {"url": file_to_data_url(image_path)}},
            ],
        },
    ]


def abs_bbox_to_ref1000(bbox_xyxy: Sequence[int], width: int, height: int) -> List[int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image width/height must be positive for ref0-1000 conversion.")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return [
        int(round(x1 / width * 1000.0)),
        int(round(y1 / height * 1000.0)),
        int(round(x2 / width * 1000.0)),
        int(round(y2 / height * 1000.0)),
    ]


def vqa_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_path = image_path_from_record(record)
    width, height = image_size(record)
    bbox_ref = abs_bbox_to_ref1000(record.get("bbox_xyxy", []), width=width, height=height)
    options = record.get("options", [])
    option_lines = "\n".join(f"{idx}. {text}" for idx, text in enumerate(options))
    system = (
        "You are a visual grounding multiple-choice model.\n"
        "You are given an image, a target bbox, and several candidate descriptions.\n"
        "Your task is to identify the object inside the bbox and choose the single option that best matches that boxed object.\n\n"
        "Return strict JSON only with this schema:\n"
        '{"selected_index": 0, "confidence": 0.0}\n\n'
        "Rules:\n"
        "1. selected_index must be an integer and must refer to exactly one provided option.\n"
        "2. The bbox is provided in ref0-1000 coordinates relative to the full image.\n"
        "3. Use the object inside the bbox as the primary evidence.\n"
        "4. Compare all options and choose the best match, even if some are similar.\n"
        "5. Do not output reasoning, markdown, or any extra text.\n"
    )
    user_text = (
        f"bbox_ref0_1000_xyxy = {bbox_ref}\n"
        "Choose the correct description for the object inside this bbox.\n"
        f"Options:\n{option_lines}"
    )
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": file_to_data_url(image_path)}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def call_model(client: Any, cfg: ApiConfig, messages: List[Dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
    )
    return extract_text(response)


def with_retries(fn: Any, retry_times: int) -> Any:
    last_error: Optional[Exception] = None
    total_attempts = max(1, retry_times + 1)
    for attempt in range(total_attempts):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
            if attempt >= total_attempts - 1:
                break
            text = str(exc).lower()
            sleep_seconds = min(20, 2 ** attempt)
            if "429" not in text and "rate" not in text:
                sleep_seconds = min(5, sleep_seconds)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def parse_choice_response(text: str, num_options: int) -> Optional[int]:
    payload = None
    try:
        payload = extract_json_payload(text)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("selected_index", "choice_index", "answer_index", "predicted_id", "prediction"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip().upper() == "UNKNOWN":
                return None
            if isinstance(value, (int, float)):
                index = int(value)
                if 0 <= index < num_options:
                    return index
            if isinstance(value, str):
                match = re.search(r"-?\d+", value)
                if match:
                    index = int(match.group(0))
                    if 0 <= index < num_options:
                        return index
    structured_match = re.search(r'"selected_index"\s*:\s*(-?\d+)', text)
    if not structured_match:
        structured_match = re.search(r"\bselected_index\b\s*[:=]\s*(-?\d+)", text)
    if structured_match:
        index = int(structured_match.group(1))
        if 0 <= index < num_options:
            return index
    match = re.search(r"-?\d+", text)
    if match:
        index = int(match.group(0))
        if 0 <= index < num_options:
            return index
    return None


def iter_records(task: str, dataset_jsonl: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    records = list(
        ev.iter_task_records(
            dataset_jsonl=dataset_jsonl,
            task=task,
            sample_id=args.sample_id,
            qa_id=args.qa_id,
        )
    )
    if args.limit is not None:
        records = records[: args.limit]
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_jsonl(args: argparse.Namespace, task: str) -> Path:
    overrides = {
        "search-grounding": args.grounding_jsonl,
        "search-vqa": args.vqa_jsonl,
        "search-seg": args.seg_jsonl,
    }
    override = overrides[task]
    if override is not None:
        return override
    filenames = {
        "search-grounding": "search_grounding.jsonl",
        "search-vqa": "search_vqa.jsonl",
        "search-seg": "search_segmentation.jsonl",
    }
    return args.dataset_root / "data" / filenames[task]


def run_grounding(client: Any, cfg: ApiConfig, records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    primary_coord_mode = resolve_coord_mode(cfg.model, args.coord_mode)

    def process(record: Dict[str, Any]) -> Dict[str, Any]:
        raw = ""
        error = None
        parse_error = None
        parse_mode = "unknown"
        detections: List[Detection] = []
        coord_mode = primary_coord_mode
        try:
            width, height = image_size(record)
            raw = with_retries(lambda: call_model(client, cfg, grounding_messages(record)), cfg.retry_times)
            parsed_items, parse_mode, parse_error = parse_raw_detections(raw)
            detections, coord_mode = normalize_detections_with_fallback(
                parsed_items=parsed_items,
                parse_mode=parse_mode,
                primary_coord_mode=primary_coord_mode,
                model_name=cfg.model,
                width=width,
                height=height,
                cfg=cfg,
            )
            if not detections:
                error = "missing_bbox_prediction"
                if parse_error:
                    error = parse_error
        except Exception as exc:
            error = str(exc)
        if args.print_raw:
            print(f"\n[{record.get('qa_id')}] {raw}")
        selected = detections[0] if detections else None
        return {
            "task": record.get("task", "Search-based Grounding"),
            "qa_id": record["qa_id"],
            "sample_id": record["sample_id"],
            "image_path": record.get("image", {}).get("rel_path", record.get("image", {}).get("path", "")),
            "question_en": record.get("question_en", ""),
            "coord_mode_used": coord_mode,
            "parse_mode": parse_mode,
            "num_predictions": len(detections),
            "predicted_label": selected.label if selected is not None else None,
            "predicted_bbox": selected.bbox_xyxy if selected is not None else None,
            "prediction_score": selected.score if selected is not None else None,
            "normalized_detections": [
                {
                    "label": det.label,
                    "bbox_xyxy": det.bbox_xyxy,
                    "source_bbox": det.source_bbox,
                    "source_coord_mode": det.source_coord_mode,
                    "saliency_rank": det.saliency_rank,
                    "score": det.score,
                }
                for det in detections
            ],
            "raw_response": raw,
            "raw_response_preview": raw[:500],
            "parse_error": parse_error,
            "error": error,
        }

    return run_parallel(records, process, cfg.workers, "grounding")


def run_vqa(client: Any, cfg: ApiConfig, records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    def process(record: Dict[str, Any]) -> Dict[str, Any]:
        raw = ""
        error = None
        selected = None
        try:
            raw = with_retries(lambda: call_model(client, cfg, vqa_messages(record)), cfg.retry_times)
            selected = parse_choice_response(raw, len(record.get("options", [])))
            if selected is None:
                error = "missing_choice_prediction"
        except Exception as exc:
            error = str(exc)
        if args.print_raw:
            print(f"\n[{record.get('qa_id')}] {raw}")
        return {
            "task": record.get("task", "Search-based VQA"),
            "qa_id": record["qa_id"],
            "sample_id": record["sample_id"],
            "bbox_xyxy": record.get("bbox_xyxy", []),
            "options": record.get("options", []),
            "answer_index": record.get("answer_index"),
            "selected_index": selected,
            "raw_response": raw,
            "raw_response_preview": raw[:500],
            "error": error,
        }

    return run_parallel(records, process, cfg.workers, "vqa")


def run_parallel(
    records: Sequence[Dict[str, Any]],
    fn: Any,
    workers: int,
    label: str,
) -> List[Dict[str, Any]]:
    total = len(records)
    if total == 0:
        print(f"{label}: 0/0")
        return []
    completed = 0
    ordered: List[Optional[Dict[str, Any]]] = [None] * total
    with ThreadPoolExecutor(max_workers=max(1, min(workers, total))) as pool:
        future_to_index = {pool.submit(fn, record): index for index, record in enumerate(records)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            ordered[index] = future.result()
            completed += 1
            print(f"\r{label}: {completed}/{total}", end="", flush=True)
    print()
    return [item for item in ordered if item is not None]


class Sam3Worker:
    def __init__(self, python: str, worker: Path, checkpoint: str, device: str) -> None:
        cmd = [python, str(worker), "--device", device]
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        ready = self._read()
        if not ready.get("ok"):
            raise RuntimeError(f"Failed to start SAM3 worker: {ready}")

    def _read(self) -> Dict[str, Any]:
        if self.proc.stdout is None:
            raise RuntimeError("SAM3 worker stdout is not available")
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"SAM3 worker exited: {stderr}")
        return json.loads(line)

    def predict(self, image_path: Path, bbox: Sequence[int]) -> Dict[str, Any]:
        if self.proc.stdin is None:
            raise RuntimeError("SAM3 worker stdin is not available")
        self.proc.stdin.write(json.dumps({"cmd": "predict", "image_path": str(image_path), "box": list(bbox)}) + "\n")
        self.proc.stdin.flush()
        response = self._read()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "SAM3 predict failed")))
        return response

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def save_mask_b64(data: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(BytesIO(base64.b64decode(data))) as image:
        image.convert("L").save(path)


def run_sam3_masks(
    grounding_rows: Sequence[Dict[str, Any]],
    grounding_records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    record_index = {str(record["qa_id"]): record for record in grounding_records}
    rows: List[Dict[str, Any]] = []
    worker = Sam3Worker(args.sam_python, args.sam_worker, args.sam_checkpoint, args.sam_device)
    try:
        total = len(grounding_rows)
        for index, row in enumerate(grounding_rows, start=1):
            qa_id = str(row.get("qa_id", ""))
            bbox = row.get("predicted_bbox")
            pred: Dict[str, Any] = {"qa_id": qa_id, "sample_id": row.get("sample_id", "")}
            try:
                if not bbox:
                    raise ValueError("missing predicted_bbox")
                record = record_index[qa_id]
                image_path = image_path_from_record(record)
                response = worker.predict(image_path, bbox)
                mask_path = output_dir / "sam3_masks" / f"{qa_id}.png"
                save_mask_b64(str(response["mask_png_b64"]), mask_path)
                pred.update({
                    "predicted_mask_path": str(mask_path),
                    "sam3_score": response.get("score"),
                    "sam3_area": response.get("area"),
                    "sam3_bbox_xywh": response.get("bbox_xywh"),
                })
            except Exception as exc:
                pred["error"] = str(exc)
            rows.append(pred)
            print(f"\rsam3: {index}/{total}", end="", flush=True)
    finally:
        worker.close()
    print()
    return rows


def evaluate_and_save(
    task: str,
    prediction_jsonl: Path,
    dataset_jsonl_path: Path,
    model_name: str,
    output_json: Path,
    output_jsonl: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = ev.run_evaluation(
        task_arg=task,
        prediction_jsonl=prediction_jsonl,
        dataset_jsonl=dataset_jsonl_path,
        model_name=model_name,
        output_json=str(output_json),
        output_jsonl=str(output_jsonl),
        sample_id=args.sample_id,
        qa_id=args.qa_id,
        limit=args.limit,
    )
    summary = {key: value for key, value in payload.items() if key != "results"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_api_config(args)
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / sanitize_name(cfg.model))
    output_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(cfg)

    grounding_jsonl = dataset_jsonl(args, "search-grounding")
    vqa_jsonl = dataset_jsonl(args, "search-vqa")
    seg_jsonl = dataset_jsonl(args, "search-seg")
    summary: Dict[str, Any] = {
        "model": cfg.model,
        "output_dir": str(output_dir),
        "dataset_root": str(args.dataset_root),
        "coord_mode_requested": args.coord_mode,
        "tasks": {},
    }

    grounding_rows: List[Dict[str, Any]] = []
    grounding_records: List[Dict[str, Any]] = []
    if args.task in {"all", "search-grounding"} or args.sam3:
        grounding_records = iter_records("search-grounding", grounding_jsonl, args)
        grounding_rows = run_grounding(client, cfg, grounding_records, args)
        prediction_path = output_dir / "grounding_predictions.jsonl"
        write_jsonl(prediction_path, grounding_rows)
        if not args.no_evaluate:
            payload = evaluate_and_save(
                "search-grounding",
                prediction_path,
                grounding_jsonl,
                cfg.model,
                output_dir / "grounding_summary.json",
                output_dir / "grounding_scored.jsonl",
                args,
            )
            summary["tasks"]["search-grounding"] = {k: v for k, v in payload.items() if k != "results"}

    if args.task in {"all", "search-vqa"}:
        vqa_records = iter_records("search-vqa", vqa_jsonl, args)
        vqa_rows = run_vqa(client, cfg, vqa_records, args)
        prediction_path = output_dir / "vqa_predictions.jsonl"
        write_jsonl(prediction_path, vqa_rows)
        if not args.no_evaluate:
            payload = evaluate_and_save(
                "search-vqa",
                prediction_path,
                vqa_jsonl,
                cfg.model,
                output_dir / "vqa_summary.json",
                output_dir / "vqa_scored.jsonl",
                args,
            )
            summary["tasks"]["search-vqa"] = {k: v for k, v in payload.items() if k != "results"}

    if args.sam3:
        sam_rows = run_sam3_masks(grounding_rows, grounding_records, args, output_dir)
        prediction_path = output_dir / "sam3_seg_predictions.jsonl"
        write_jsonl(prediction_path, sam_rows)
        if not args.no_evaluate:
            payload = evaluate_and_save(
                "search-seg",
                prediction_path,
                seg_jsonl,
                cfg.model,
                output_dir / "sam3_seg_summary.json",
                output_dir / "sam3_seg_scored.jsonl",
                args,
            )
            summary["tasks"]["search-seg-sam3"] = {k: v for k, v in payload.items() if k != "results"}

    write_json(output_dir / "run_summary.json", summary)
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
