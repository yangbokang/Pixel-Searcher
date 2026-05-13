#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebEyes prediction evaluator.

This script only scores existing predictions. It does not call any model.
Accepted prediction types:
  - bbox:   {"qa_id": "...", "predicted_bbox": [x1, y1, x2, y2]}
  - mask:   {"qa_id": "...", "predicted_mask_path": "..."} or {"mask_png_b64": "..."}
  - choice: {"qa_id": "...", "selected_index": 0}
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent


def default_dataset_root() -> Path:
    return CODE_ROOT / "dataset"


DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_ANNOTATION_JSONL = DEFAULT_DATASET_ROOT / "annotations" / "dataset.jsonl"
DEFAULT_TASK_JSONLS = {
    "search-grounding": DEFAULT_DATASET_ROOT / "data" / "search_grounding.jsonl",
    "search-seg": DEFAULT_DATASET_ROOT / "data" / "search_segmentation.jsonl",
    "search-vqa": DEFAULT_DATASET_ROOT / "data" / "search_vqa.jsonl",
}

TASK_SEARCH_GROUNDING = "Search-based Grounding"
TASK_SEARCH_SEG = "Search-based Segmentation"
TASK_SEARCH_VQA = "Search-based VQA"
TASKS = {TASK_SEARCH_GROUNDING, TASK_SEARCH_SEG, TASK_SEARCH_VQA}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate bbox, mask, or choice predictions for WebEyes.")
    parser.add_argument(
        "--dataset-jsonl",
        help="Path to a task JSONL or annotation JSONL. Defaults to the matching dataset/data/*.jsonl file.",
    )
    parser.add_argument("--prediction-jsonl", help="Prediction JSONL to score.")
    parser.add_argument(
        "--prediction-mask-dir",
        help="Directory with one predicted mask per qa_id. Only used with --task search-seg.",
    )
    parser.add_argument(
        "--mask-name-template",
        default="{qa_id}.png",
        help="Filename template under --prediction-mask-dir, e.g. '{qa_id}.png'.",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["search-grounding", "search-seg", "search-vqa"],
        help="Which task interface to score.",
    )
    parser.add_argument("--model-name", default="model", help="Name written into the summary.")
    parser.add_argument("--output-json", help="Optional path for JSON summary plus per-item results.")
    parser.add_argument("--output-jsonl", help="Optional path for per-item scored rows.")
    parser.add_argument("--sample-id", help="Only evaluate one sample_id.")
    parser.add_argument("--qa-id", help="Only evaluate one qa_id.")
    parser.add_argument("--limit", type=int, help="Evaluate first N records after filtering.")
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="Mask foreground threshold in [0, 255].")
    return parser.parse_args()


def task_name(task_arg: str) -> str:
    return {
        "search-grounding": TASK_SEARCH_GROUNDING,
        "search-seg": TASK_SEARCH_SEG,
        "search-vqa": TASK_SEARCH_VQA,
    }[task_arg]


def default_dataset_jsonl_for_task(task: str) -> Path:
    return DEFAULT_TASK_JSONLS[task]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"JSONL not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def normalize_bbox(raw_bbox: Any) -> Optional[List[int]]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in raw_bbox]
    except Exception:
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [x1, y1, x2, y2]


def dataset_abs_path(dataset_jsonl: Path, rel_path: str) -> str:
    path = Path(str(rel_path).strip())
    if path.is_absolute():
        return str(path)
    bases = []
    jsonl_parent = dataset_jsonl.resolve().parent
    bases.append(jsonl_parent)
    if jsonl_parent.name in {"jsonl", "annotations", "data"}:
        bases.append(jsonl_parent.parent)
    bases.append(DEFAULT_DATASET_ROOT)
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((bases[-1] / path).resolve())


def is_flat_task_row(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("image"), str) and row.get("task") in TASKS and row.get("qa_id")


def flat_task_record_to_dict(row: Dict[str, Any], dataset_jsonl: Path) -> Dict[str, Any]:
    image_rel_path = str(row.get("image", "")).strip()
    image_path = dataset_abs_path(dataset_jsonl, image_rel_path)
    object_ids = row.get("target_object_ids", [])
    names = row.get("object_names", [])
    categories = row.get("categories", [])
    boxes = row.get("target_boxes") or []
    if row.get("task") == TASK_SEARCH_VQA and row.get("bbox_xyxy"):
        boxes = [row.get("bbox_xyxy")]
    mask_rels = row.get("target_masks", [])

    target_objects = []
    for index, object_id in enumerate(object_ids if isinstance(object_ids, list) else []):
        bbox = boxes[index] if index < len(boxes) else []
        mask_rel = mask_rels[index] if index < len(mask_rels) else ""
        target_objects.append(
            {
                "object_id": str(object_id),
                "bbox_xyxy": normalize_bbox(bbox) or [],
                "mask": {
                    "rel_path": str(mask_rel),
                    "path": dataset_abs_path(dataset_jsonl, str(mask_rel)) if mask_rel else "",
                },
                "category": str(categories[index]) if index < len(categories) else "",
                "name_en": str(names[index]) if index < len(names) else "",
            }
        )

    record = {
        "task": row.get("task"),
        "sample_id": str(row.get("sample_id", "")),
        "qa_id": str(row.get("qa_id", "")),
        "image": {"rel_path": image_rel_path, "path": image_path},
        "question_en": str(row.get("question", "")),
        "target_objects": target_objects,
    }
    if row.get("task") == TASK_SEARCH_GROUNDING:
        record["target_boxes"] = row.get("target_boxes", [])
    elif row.get("task") == TASK_SEARCH_SEG:
        record["target_masks"] = [
            {
                "object_id": str(object_ids[index]) if index < len(object_ids) else "",
                "rel_path": str(mask_rel),
                "path": dataset_abs_path(dataset_jsonl, str(mask_rel)),
            }
            for index, mask_rel in enumerate(mask_rels if isinstance(mask_rels, list) else [])
        ]
    elif row.get("task") == TASK_SEARCH_VQA:
        record["bbox_xyxy"] = normalize_bbox(row.get("bbox_xyxy")) or []
        record["options"] = row.get("options", [])
        record["answer_index"] = row.get("answer_index")
    return record


def iter_flat_task_rows(
    rows: List[Dict[str, Any]],
    dataset_jsonl: Path,
    task: str,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    requested_task = task_name(task)
    for row in rows:
        if row.get("task") != requested_task:
            continue
        if sample_id and str(row.get("sample_id", "")).strip() != sample_id:
            continue
        if qa_id and str(row.get("qa_id", "")).strip() != qa_id:
            continue
        yield flat_task_record_to_dict(row, dataset_jsonl)


def iter_annotation_task_records(
    rows: List[Dict[str, Any]],
    dataset_jsonl: Path,
    task: str,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    requested_task = task_name(task)
    records: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        current_sample_id = str(row.get("sample_id", "")).strip()
        if sample_id and current_sample_id != sample_id:
            continue
        image = row.get("image", {}) if isinstance(row.get("image"), dict) else {}
        obj = row.get("object", {}) if isinstance(row.get("object"), dict) else {}
        object_id = str(obj.get("object_id", "")).strip()
        bbox = normalize_bbox(obj.get("bbox_xyxy"))
        if not object_id or bbox is None:
            continue
        mask = obj.get("mask", {}) if isinstance(obj.get("mask"), dict) else {}
        mask_rel_path = str(mask.get("rel_path", "")).strip()
        target = {
            "object_id": object_id,
            "bbox_xyxy": bbox,
            "mask": {
                "rel_path": mask_rel_path,
                "path": dataset_abs_path(dataset_jsonl, mask_rel_path),
            },
            "category": str(obj.get("category", "")).strip(),
            "name_en": str(obj.get("name_en", "")).strip(),
        }

        for qa in row.get("qa_pairs", []) or []:
            if not isinstance(qa, dict):
                continue
            current_qa_id = str(qa.get("qa_id", "")).strip()
            if not current_qa_id or (qa_id and current_qa_id != qa_id):
                continue
            record = records.get(current_qa_id)
            if record is None:
                record = {
                    "task": requested_task,
                    "sample_id": current_sample_id,
                    "qa_id": current_qa_id,
                    "image": {
                        "rel_path": str(image.get("rel_path", "")).strip(),
                        "path": dataset_abs_path(dataset_jsonl, str(image.get("rel_path", "")).strip()),
                    },
                    "question_en": str(qa.get("question_en", "")).strip(),
                    "answer_en": str(qa.get("answer_en", "")).strip(),
                    "target_objects": [],
                }
                choice = qa.get("choice", {})
                if isinstance(choice, dict):
                    options = choice.get("options", [])
                    if isinstance(options, list):
                        record["options"] = [str(option).strip() for option in options]
                    try:
                        record["answer_index"] = int(choice.get("answer_index", -1))
                    except Exception:
                        record["answer_index"] = -1
                records[current_qa_id] = record
            if all(existing.get("object_id") != object_id for existing in record["target_objects"]):
                record["target_objects"].append(target)

    for record in records.values():
        targets = record["target_objects"]
        if requested_task == TASK_SEARCH_GROUNDING:
            record["target_boxes"] = [target["bbox_xyxy"] for target in targets]
        elif requested_task == TASK_SEARCH_SEG:
            record["target_masks"] = [
                {
                    "object_id": target["object_id"],
                    "rel_path": target["mask"]["rel_path"],
                    "path": target["mask"]["path"],
                }
                for target in targets
            ]
        elif requested_task == TASK_SEARCH_VQA:
            if not record.get("options"):
                continue
            first_target = targets[0] if targets else None
            record["bbox_xyxy"] = first_target["bbox_xyxy"] if first_target else []
        yield record


def iter_task_records(
    dataset_jsonl: Path,
    task: str,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    rows = load_jsonl(dataset_jsonl)
    if rows and is_flat_task_row(rows[0]):
        yield from iter_flat_task_rows(rows, dataset_jsonl, task, sample_id=sample_id, qa_id=qa_id)
    else:
        yield from iter_annotation_task_records(rows, dataset_jsonl, task, sample_id=sample_id, qa_id=qa_id)


def load_prediction_index(path: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        qa_id = str(row.get("qa_id", "")).strip()
        if qa_id:
            index[qa_id] = row
    return index


def load_mask_dir_prediction_index(
    records: Sequence[Dict[str, Any]],
    mask_dir: Path,
    name_template: str,
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for record in records:
        qa_id = str(record.get("qa_id", "")).strip()
        if not qa_id:
            continue
        mask_name = name_template.format(
            qa_id=qa_id,
            sample_id=str(record.get("sample_id", "")).strip(),
        )
        mask_path = (mask_dir / mask_name).resolve()
        index[qa_id] = {"qa_id": qa_id, "predicted_mask_path": str(mask_path)}
    return index


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0.0 else 0.0


def best_iou(pred_box: Optional[Sequence[int]], target_boxes: Sequence[Sequence[int]]) -> float:
    if pred_box is None:
        return 0.0
    return max((box_iou(pred_box, target) for target in target_boxes), default=0.0)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0.0 else 0.0


def mask_intersection_union(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    return int(np.logical_and(mask_a, mask_b).sum()), int(np.logical_or(mask_a, mask_b).sum())


def load_mask(path: Path, target_shape: Optional[tuple[int, int]] = None, threshold: float = 0.0) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        if target_shape is not None:
            target_h, target_w = target_shape
            if gray.size != (target_w, target_h):
                gray = gray.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        return np.asarray(gray) > float(threshold)


def load_mask_b64(data: str, target_shape: tuple[int, int], threshold: float) -> np.ndarray:
    raw = base64.b64decode(data)
    with Image.open(BytesIO(raw)) as image:
        gray = image.convert("L")
        target_h, target_w = target_shape
        if gray.size != (target_w, target_h):
            gray = gray.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        return np.asarray(gray) > float(threshold)


def union_gt_mask(record: Dict[str, Any]) -> np.ndarray:
    masks = []
    for target in record.get("target_masks", []):
        mask_path = Path(str(target.get("path", "")))
        if mask_path.exists():
            masks.append(load_mask(mask_path))
    if not masks:
        raise FileNotFoundError(f"No GT masks found for qa_id={record.get('qa_id')}")
    union = masks[0].copy()
    for mask in masks[1:]:
        if mask.shape != union.shape:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((union.shape[1], union.shape[0]), Image.Resampling.NEAREST)) > 0
        union = np.logical_or(union, mask)
    return union


def resolve_pred_mask_path(raw_path: str, prediction_jsonl: Path) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path
    candidates = [
        prediction_jsonl.resolve().parent / path,
        Path.cwd() / path,
        PROJECT_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def score_grounding(record: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    pred_box = None
    if pred:
        for key in ("predicted_bbox", "bbox_xyxy", "bbox"):
            pred_box = normalize_bbox(pred.get(key))
            if pred_box is not None:
                break
    target_boxes = [target.get("bbox_xyxy", []) for target in record.get("target_objects", [])]
    iou = best_iou(pred_box, target_boxes)
    return {
        "task": TASK_SEARCH_GROUNDING,
        "qa_id": record["qa_id"],
        "sample_id": record["sample_id"],
        "predicted_bbox": pred_box,
        "target_boxes": target_boxes,
        "best_iou": round(iou, 6),
        "is_correct_iou50": iou >= 0.5,
        "error": None if pred_box is not None else "missing_bbox_prediction",
    }


def score_seg(record: Dict[str, Any], pred: Optional[Dict[str, Any]], prediction_jsonl: Path, threshold: float) -> Dict[str, Any]:
    gt_mask = union_gt_mask(record)
    pred_mask = None
    source = ""
    error = None
    if pred:
        if pred.get("mask_png_b64"):
            try:
                pred_mask = load_mask_b64(str(pred["mask_png_b64"]), gt_mask.shape, threshold)
                source = "mask_png_b64"
            except Exception as exc:
                error = str(exc)
        else:
            for key in ("predicted_mask_path", "mask_path", "prediction_mask_path"):
                value = str(pred.get(key, "")).strip()
                if value:
                    path = resolve_pred_mask_path(value, prediction_jsonl)
                    source = str(path)
                    if path.exists():
                        pred_mask = load_mask(path, target_shape=gt_mask.shape, threshold=threshold)
                    else:
                        error = f"mask path not found: {path}"
                    break
    if pred_mask is None and error is None:
        error = "missing_mask_prediction"

    current_iou = mask_iou(pred_mask, gt_mask) if pred_mask is not None else 0.0
    inter, union = mask_intersection_union(pred_mask, gt_mask) if pred_mask is not None else (0, int(gt_mask.sum()))
    return {
        "task": TASK_SEARCH_SEG,
        "qa_id": record["qa_id"],
        "sample_id": record["sample_id"],
        "predicted_mask_source": source,
        "mask_iou": round(current_iou, 6),
        "mask_intersection": inter,
        "mask_union": union,
        "is_correct_mask_iou50": current_iou >= 0.5,
        "error": error,
    }


def score_vqa(record: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    selected = None
    if pred:
        for key in ("selected_index", "choice_index", "answer_index", "predicted_id", "prediction"):
            if pred.get(key) is None:
                continue
            value = pred.get(key)
            if isinstance(value, str) and value.strip().upper() == "UNKNOWN":
                selected = None
                break
            try:
                selected = int(value)
                break
            except Exception:
                match = re.search(r"-?\d+", str(value))
                if match:
                    selected = int(match.group(0))
                    break
    answer_index = record.get("answer_index")
    is_correct = selected == answer_index if selected is not None else False
    return {
        "task": TASK_SEARCH_VQA,
        "qa_id": record["qa_id"],
        "sample_id": record["sample_id"],
        "selected_index": selected,
        "answer_index": answer_index,
        "is_correct": is_correct,
        "error": None if selected is not None else "missing_choice_prediction",
    }


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def summarize(task: str, rows: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    total = len(rows)
    errors = sum(1 for row in rows if row.get("error"))
    if task == TASK_SEARCH_GROUNDING:
        correct = sum(1 for row in rows if row.get("is_correct_iou50"))
        return {
            "task": task,
            "model": model_name,
            "num_items": total,
            "num_questions": total,
            "num_errors": errors,
            "num_questions_with_prediction": total - errors,
            "num_correct_iou50": correct,
            "accuracy_iou50": round(correct / total, 6) if total else 0.0,
            "mean_iou": mean(float(row.get("best_iou", 0.0)) for row in rows),
            "mean_best_iou_all": mean(float(row.get("best_iou", 0.0)) for row in rows),
            "mean_best_iou_predicted_only": mean(
                float(row.get("best_iou", 0.0)) for row in rows if not row.get("error")
            ),
            "recall_iou50": round(correct / total, 6) if total else 0.0,
        }
    if task == TASK_SEARCH_SEG:
        correct = sum(1 for row in rows if row.get("is_correct_mask_iou50"))
        inter = sum(int(row.get("mask_intersection", 0) or 0) for row in rows)
        union = sum(int(row.get("mask_union", 0) or 0) for row in rows)
        return {
            "task": task,
            "model": model_name,
            "num_items": total,
            "num_questions": total,
            "num_errors": errors,
            "mask_g_iou": mean(float(row.get("mask_iou", 0.0)) for row in rows),
            "mask_c_iou": round(inter / union, 6) if union else 0.0,
            "gIoU": mean(float(row.get("mask_iou", 0.0)) for row in rows),
            "cIoU": round(inter / union, 6) if union else 0.0,
            "mask_iou_ge_50_rate": round(correct / total, 6) if total else 0.0,
        }
    correct = sum(1 for row in rows if row.get("is_correct"))
    answered = sum(1 for row in rows if row.get("selected_index") is not None)
    return {
        "task": task,
        "model": model_name,
        "num_items": total,
        "num_questions": total,
        "num_answered": answered,
        "num_correct": correct,
        "num_errors": errors,
        "accuracy": round(correct / total, 6) if total else 0.0,
    }


def write_json(path: str, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_evaluation(
    task_arg: str,
    prediction_jsonl: Optional[Path] = None,
    dataset_jsonl: Optional[Path] = None,
    model_name: str = "model",
    output_json: Optional[str] = None,
    output_jsonl: Optional[str] = None,
    sample_id: Optional[str] = None,
    qa_id: Optional[str] = None,
    limit: Optional[int] = None,
    mask_threshold: float = 0.0,
    prediction_mask_dir: Optional[Path] = None,
    mask_name_template: str = "{qa_id}.png",
) -> Dict[str, Any]:
    dataset_jsonl = dataset_jsonl or default_dataset_jsonl_for_task(task_arg)
    task = task_name(task_arg)
    records = list(
        iter_task_records(
            dataset_jsonl=dataset_jsonl,
            task=task_arg,
            sample_id=sample_id,
            qa_id=qa_id,
        )
    )
    if limit is not None:
        records = records[:limit]

    if prediction_jsonl is not None:
        prediction_index = load_prediction_index(prediction_jsonl)
        prediction_base = prediction_jsonl
    elif task == TASK_SEARCH_SEG and prediction_mask_dir is not None:
        prediction_index = load_mask_dir_prediction_index(records, prediction_mask_dir, mask_name_template)
        prediction_base = prediction_mask_dir
    else:
        raise SystemExit("--prediction-jsonl is required unless --task search-seg uses --prediction-mask-dir.")

    rows: List[Dict[str, Any]] = []
    for record in records:
        pred = prediction_index.get(str(record.get("qa_id", "")))
        if task == TASK_SEARCH_GROUNDING:
            rows.append(score_grounding(record, pred))
        elif task == TASK_SEARCH_SEG:
            rows.append(score_seg(record, pred, prediction_base, mask_threshold))
        else:
            rows.append(score_vqa(record, pred))

    summary = summarize(task, rows, model_name)
    payload = {**summary, "results": rows}
    if output_json:
        write_json(output_json, payload)
    if output_jsonl:
        write_jsonl(output_jsonl, rows)
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    prediction_jsonl = Path(args.prediction_jsonl) if args.prediction_jsonl else None
    prediction_mask_dir = Path(args.prediction_mask_dir) if args.prediction_mask_dir else None
    payload = run_evaluation(
        task_arg=args.task,
        prediction_jsonl=prediction_jsonl,
        dataset_jsonl=Path(args.dataset_jsonl) if args.dataset_jsonl else None,
        model_name=args.model_name,
        output_json=args.output_json,
        output_jsonl=args.output_jsonl,
        sample_id=args.sample_id,
        qa_id=args.qa_id,
        limit=args.limit,
        mask_threshold=args.mask_threshold,
        prediction_mask_dir=prediction_mask_dir,
        mask_name_template=args.mask_name_template,
    )
    summary = {key: value for key, value in payload.items() if key != "results"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
