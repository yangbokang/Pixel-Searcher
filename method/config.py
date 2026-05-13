#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone configuration, utilities, and data loaders for the multi-round
interactive grounding & choice benchmark system.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATASET_ROOT = CODE_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = CODE_ROOT / "outputs"
TASK_SEARCH_GROUNDING = "Search-based Grounding"
TASK_SEARCH_VQA = "Search-based VQA"


# ---------------------------------------------------------------------------
# Box & IoU
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Axis-aligned bounding box in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float

    def normalize(self) -> "Box":
        """Ensure x1<=x2, y1<=y2."""
        lx, rx = sorted((self.x1, self.x2))
        ly, ry = sorted((self.y1, self.y2))
        return Box(lx, ly, rx, ry)

    def clip(self, width: int, height: int) -> "Box":
        """Clip to image bounds."""
        return Box(
            x1=max(0.0, min(float(width - 1), self.x1)),
            y1=max(0.0, min(float(height - 1), self.y1)),
            x2=max(0.0, min(float(width - 1), self.x2)),
            y2=max(0.0, min(float(height - 1), self.y2)),
        )

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_int_list(self) -> List[int]:
        return [int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2))]


def compute_iou(a: Box, b: Box) -> float:
    """Compute Intersection-over-Union between two boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def best_iou_to_targets(pred_box: Optional[List[int]],
                         gt_boxes: List[List[int]]) -> float:
    """Return best IoU between a predicted box and a list of GT boxes."""
    if pred_box is None or len(pred_box) != 4 or not gt_boxes:
        return 0.0
    pred = Box(*[float(v) for v in pred_box]).normalize()
    best = 0.0
    for gt in gt_boxes:
        if len(gt) != 4:
            continue
        gt_box = Box(*[float(v) for v in gt]).normalize()
        best = max(best, compute_iou(pred, gt_box))
    return best


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All configuration for the multi-round system."""
    # API
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = ""
    serper_api_key: str = ""
    # Generation
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout: float = 120.0
    retry_times: int = 2
    # Multi-round
    max_search_rounds: int = 5
    saliency_top_k: int = 8
    max_workers: int = 4
    # Saliency / detection
    max_boxes: int = 20
    min_box_area_ratio: float = 0.01
    max_box_area_ratio: float = 0.95
    nms_iou_threshold: float = 0.65
    # Search
    search_results_per_query: int = 5
    search_max_queries: int = 3
    # Paths
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    # Flags
    save_raw: bool = True
    print_raw: bool = False


def _parse_env_file(path: str) -> Dict[str, str]:
    """Parse a simple KEY=VALUE .env file."""
    env: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        # try relative to this script
        p = Path(__file__).resolve().parent / path
    if not p.exists():
        raise SystemExit(f"[config] .env file not found: {path}")
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        env[key] = val
    return env


def load_config(env_file: str = ".env") -> Config:
    """Load config from .env file."""
    env = _parse_env_file(env_file)
    api_key = env.get("OPENAI_API_KEY", "").strip()
    base_url = env.get("OPENAI_BASE_URL", "").strip()
    model = env.get("OPENAI_MODEL", "").strip()

    missing = []
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not model:
        missing.append("OPENAI_MODEL")
    if missing:
        raise SystemExit(f"[config] Missing: {', '.join(missing)}")

    def _int(k: str, d: int) -> int:
        v = env.get(k)
        return int(v) if v else d

    def _float(k: str, d: float) -> float:
        v = env.get(k)
        return float(v) if v else d

    return Config(
        openai_api_key=api_key,
        openai_base_url=base_url,
        openai_model=model,
        serper_api_key=env.get("SERPER_API_KEY", "").strip(),
        temperature=_float("TEMPERATURE", 0.0),
        top_p=_float("TOP_P", 1.0),
        max_tokens=_int("MAX_TOKENS", 1024),
        timeout=_float("TIMEOUT", 120.0),
        retry_times=_int("RETRY_TIMES", 2),
        max_search_rounds=_int("MAX_SEARCH_ROUNDS", 5),
        saliency_top_k=_int("SALIENCY_TOP_K", 8),
        max_workers=max(1, _int("MAX_WORKERS", 4)),
        max_boxes=_int("MAX_BOXES", 20),
        min_box_area_ratio=_float("MIN_BOX_AREA_RATIO", 0.01),
        max_box_area_ratio=_float("MAX_BOX_AREA_RATIO", 0.95),
        nms_iou_threshold=_float("NMS_IOU_THRESHOLD", 0.65),
        search_results_per_query=_int("SEARCH_RESULTS_PER_QUERY", 5),
        search_max_queries=_int("SEARCH_MAX_QUERIES", 3),
        dataset_root=env.get("DATASET_ROOT", str(DEFAULT_DATASET_ROOT)).strip() or str(DEFAULT_DATASET_ROOT),
        output_root=env.get("OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)).strip() or str(DEFAULT_OUTPUT_ROOT),
        save_raw=env.get("SAVE_RAW", "true").strip().lower() in {"1", "true", "yes"},
        print_raw=False,
    )


# ---------------------------------------------------------------------------
# OpenAI client helpers
# ---------------------------------------------------------------------------

def build_openai_client(cfg: Config):
    """Create an OpenAI-compatible client."""
    from openai import OpenAI
    return OpenAI(
        api_key=cfg.openai_api_key,
        base_url=cfg.openai_base_url,
        timeout=cfg.timeout,
    )


def file_to_data_url(path: Path) -> str:
    """Encode a local file as a data: URL for the vision API."""
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "application/octet-stream"
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def extract_llm_text(response: Any) -> str:
    """Pull the text string from an OpenAI chat response."""
    # responses API
    if hasattr(response, "output_text") and response.output_text:
        return str(response.output_text).strip()
    # chat completions API
    try:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    t = getattr(item, "text", None)
                    if t:
                        parts.append(str(t))
            return "\n".join(p.strip() for p in parts if p).strip()
    except Exception:
        pass
    return str(response).strip()


def extract_json(text: str) -> Any:
    """Robustly extract a JSON object or array from LLM output.

    Handles:
      - Qwen <think>...</think> reasoning blocks (closed AND unclosed)
      - Markdown ```json ... ``` fencing
      - JSON embedded in prose text
      - Truncated JSON (best effort)
    """
    cleaned = text.strip()
    # 1. Strip <think>...</think> blocks (Qwen reasoning mode)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S).strip()
    # 1b. Handle UNCLOSED <think> tags: strip from <think> to end, or to
    #     first JSON-like character
    if "<think>" in cleaned:
        # find the <think> tag and look for JSON after it
        parts = cleaned.split("<think>", 1)
        before = parts[0].strip()
        after = parts[1] if len(parts) > 1 else ""
        # look for JSON start in the after part
        json_start = -1
        for i, ch in enumerate(after):
            if ch in "{[":
                json_start = i
                break
        if json_start >= 0:
            cleaned = before + " " + after[json_start:]
        else:
            cleaned = before
        cleaned = cleaned.strip()
    # 2. Strip markdown code fences
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, flags=re.S
    )
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # 3. Try full parse first
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 4. Scan for first { or [ and attempt parse
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[i:])
            return obj
        except Exception:
            continue
    raise ValueError("No valid JSON found in LLM output.")


def call_llm_text(client, cfg: Config, prompt: str,
                   max_tokens: Optional[int] = None) -> str:
    """Call LLM with a text-only prompt."""
    resp = client.chat.completions.create(
        model=cfg.openai_model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_tokens or min(cfg.max_tokens, 768),
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return extract_llm_text(resp)


def call_llm_vision(client, cfg: Config, prompt: str,
                     image_paths: List[Path],
                     max_tokens: Optional[int] = None) -> str:
    """Call LLM with text + one or more images."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": file_to_data_url(p)},
        })
    resp = client.chat.completions.create(
        model=cfg.openai_model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_tokens or cfg.max_tokens,
        messages=[{"role": "user", "content": content}],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return extract_llm_text(resp)


def with_retry(fn, retries: int = 2):
    """Call fn() with retries on exception."""
    last_err = None
    for _ in range(max(1, retries + 1)):
        try:
            return fn()
        except Exception as e:
            last_err = e
    raise last_err  # type: ignore


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@dataclass
class GroundingEntry:
    """One row from benchmark_grounding.jsonl."""
    task: str
    sample_id: str
    qa_id: str
    image_rel_path: str
    image_width: int
    image_height: int
    question_en: str
    answer_en: str
    target_boxes: List[List[int]] = field(default_factory=list)


@dataclass
class ChoiceEntry:
    """One row from benchmark_grounding_choice.jsonl."""
    task: str
    sample_id: str
    qa_id: str
    image_rel_path: str
    image_width: int
    image_height: int
    bbox_xyxy: List[int] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    answer_index: int = -1
    answer_description_en: str = ""
    answer_en: str = ""


def load_grounding_entries(
    jsonl_path: str,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
) -> List[GroundingEntry]:
    """Parse released Search-based Grounding JSONL or object-level annotations."""
    p = Path(jsonl_path)
    if not p.exists():
        raise SystemExit(f"[data] File not found: {jsonl_path}")
    entries: List[GroundingEntry] = []
    object_level_entries: Dict[str, GroundingEntry] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)

        if isinstance(item.get("object"), dict) and isinstance(item.get("qa_pairs"), list):
            sid = str(item.get("sample_id", "")).strip()
            if sample_id and sid != sample_id:
                continue
            img = item.get("image", {}) if isinstance(item.get("image"), dict) else {}
            obj = item.get("object", {})
            bbox = _normalize_bbox(obj.get("bbox_xyxy"))
            if bbox is None:
                continue
            for qa in item.get("qa_pairs", []):
                if not isinstance(qa, dict):
                    continue
                qid = str(qa.get("qa_id", "")).strip()
                if not qid or (qa_id and qid != qa_id):
                    continue
                entry = object_level_entries.get(qid)
                if entry is None:
                    entry = GroundingEntry(
                        task=TASK_SEARCH_GROUNDING,
                        sample_id=sid,
                        qa_id=qid,
                        image_rel_path=str(img.get("rel_path", "")),
                        image_width=int(img.get("width", 0)),
                        image_height=int(img.get("height", 0)),
                        question_en=str(qa.get("question_en", "")).strip(),
                        answer_en=str(qa.get("answer_en", "")).strip(),
                        target_boxes=[],
                    )
                    object_level_entries[qid] = entry
                if bbox not in entry.target_boxes:
                    entry.target_boxes.append(bbox)
            continue

        sid = str(item.get("sample_id", "")).strip()
        qid = str(item.get("qa_id", "")).strip()
        if sample_id and sid != sample_id:
            continue
        if qa_id and qid != qa_id:
            continue
        if item.get("task") and item.get("task") not in {TASK_SEARCH_GROUNDING, "Search-based Grounding"}:
            continue
        image_rel_path, image_width, image_height = _image_fields(item)
        # parse target boxes
        raw_targets = item.get("target_boxes", [])
        tboxes: List[List[int]] = []
        if isinstance(raw_targets, list):
            for t in raw_targets:
                bbox = t.get("bbox_xyxy") if isinstance(t, dict) else t
                normalized = _normalize_bbox(bbox)
                if normalized is not None:
                    tboxes.append(normalized)
        entries.append(GroundingEntry(
            task=TASK_SEARCH_GROUNDING,
            sample_id=sid,
            qa_id=qid,
            image_rel_path=image_rel_path,
            image_width=image_width,
            image_height=image_height,
            question_en=str(item.get("question_en") or item.get("question") or "").strip(),
            answer_en=str(item.get("answer_en", "")).strip(),
            target_boxes=tboxes,
        ))
    if object_level_entries:
        entries.extend(object_level_entries.values())
    return entries


def load_choice_entries(
    jsonl_path: str,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
) -> List[ChoiceEntry]:
    """Parse released Search-based VQA JSONL or object-level annotations."""
    p = Path(jsonl_path)
    if not p.exists():
        raise SystemExit(f"[data] File not found: {jsonl_path}")
    entries: List[ChoiceEntry] = []
    seen_object_level_qas: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)

        if isinstance(item.get("object"), dict) and isinstance(item.get("qa_pairs"), list):
            sid = str(item.get("sample_id", "")).strip()
            if sample_id and sid != sample_id:
                continue
            img = item.get("image", {}) if isinstance(item.get("image"), dict) else {}
            bbox = _normalize_bbox(item.get("object", {}).get("bbox_xyxy"))
            if bbox is None:
                continue
            for qa in item.get("qa_pairs", []):
                if not isinstance(qa, dict):
                    continue
                qid = str(qa.get("qa_id", "")).strip()
                if not qid or qid in seen_object_level_qas:
                    continue
                if qa_id and qid != qa_id:
                    continue
                choice = qa.get("choice", {})
                if not isinstance(choice, dict):
                    continue
                options = choice.get("options", [])
                if not isinstance(options, list) or not options:
                    continue
                seen_object_level_qas.add(qid)
                entries.append(ChoiceEntry(
                    task=TASK_SEARCH_VQA,
                    sample_id=sid,
                    qa_id=qid,
                    image_rel_path=str(img.get("rel_path", "")),
                    image_width=int(img.get("width", 0)),
                    image_height=int(img.get("height", 0)),
                    bbox_xyxy=bbox,
                    options=[str(o).strip() for o in options],
                    answer_index=int(choice.get("answer_index", -1)),
                    answer_description_en=str(qa.get("description_en", "")).strip(),
                    answer_en=str(qa.get("answer_en", "")).strip(),
                ))
            continue

        sid = str(item.get("sample_id", "")).strip()
        qid = str(item.get("qa_id", "")).strip()
        if sample_id and sid != sample_id:
            continue
        if qa_id and qid != qa_id:
            continue
        if item.get("task") and item.get("task") not in {TASK_SEARCH_VQA, "Search-based VQA"}:
            continue
        image_rel_path, image_width, image_height = _image_fields(item)
        grounding = item.get("grounding", {})
        bbox = item.get("bbox_xyxy") or grounding.get("bbox_xyxy", [])
        options = item.get("options", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        if not isinstance(options, list) or not options:
            continue
        entries.append(ChoiceEntry(
            task=TASK_SEARCH_VQA,
            sample_id=sid,
            qa_id=qid,
            image_rel_path=image_rel_path,
            image_width=image_width,
            image_height=image_height,
            bbox_xyxy=[int(round(float(v))) for v in bbox],
            options=[str(o).strip() for o in options],
            answer_index=int(item.get("answer_index", -1)),
            answer_description_en=str(item.get("answer_description_en", "")).strip(),
            answer_en=str(item.get("answer_en", "")).strip(),
        ))
    return entries


def _normalize_bbox(raw_bbox: Any) -> Optional[List[int]]:
    """Normalize a bbox-like value to integer xyxy."""
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        return [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None


def _image_fields(item: Dict[str, Any]) -> tuple[str, int, int]:
    image = item.get("image", {})
    if isinstance(image, dict):
        return (
            str(image.get("rel_path", "")).strip(),
            int(image.get("width", 0) or 0),
            int(image.get("height", 0) or 0),
        )
    return str(image).strip(), 0, 0


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_image_path(dataset_root: str, rel_path: str) -> Path:
    """Resolve relative image path to absolute."""
    root = Path(dataset_root)
    candidates = [
        root / rel_path,
        CODE_ROOT / root / rel_path,
        CURRENT_DIR / root / rel_path,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Image not found: {rel_path} (tried {tried})")


def sanitize_model_name(name: str) -> str:
    """Make model name safe for filesystem use."""
    s = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    s = re.sub(r"\s+", "_", s)
    return s.strip("._") or "unknown_model"


def ensure_dir(path: Path) -> Path:
    """Create directory if needed, return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
