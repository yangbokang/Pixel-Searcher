#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified benchmark runner for the multi-round interactive grounding & choice
system.  Supports resume, concurrent processing, and both task types.

Usage:
    python run_benchmark.py --task grounding [--limit N] [--print-raw]
    python run_benchmark.py --task choice    [--limit N] [--print-raw]
    python run_benchmark.py --task all
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config import (
    Config,
    ChoiceEntry,
    GroundingEntry,
    TASK_SEARCH_GROUNDING,
    TASK_SEARCH_VQA,
    best_iou_to_targets,
    build_openai_client,
    ensure_dir,
    load_choice_entries,
    load_config,
    load_grounding_entries,
    resolve_image_path,
    sanitize_model_name,
)
from multi_round_agent import run_choice, run_grounding

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_DATASET_ROOT = CODE_ROOT / "dataset"
DEFAULT_GROUNDING_JSONL = DEFAULT_DATASET_ROOT / "data" / "search_grounding.jsonl"
DEFAULT_CHOICE_JSONL = DEFAULT_DATASET_ROOT / "data" / "search_vqa.jsonl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pixel-Search runner for Search-based Grounding and Search-based VQA."
    )
    p.add_argument("--task", required=True,
                   choices=[
                       "grounding", "choice", "all",
                       "search-grounding", "search-vqa",
                   ],
                   help="Task to run: Search-based Grounding, Search-based VQA, or all.")
    p.add_argument("--env-file", default=".env",
                   help="Path to local config file.")
    p.add_argument("--model", help="Override model name.")
    p.add_argument("--limit", type=int,
                   help="Process only first N entries.")
    p.add_argument("--sample-id", help="Process only this sample_id.")
    p.add_argument("--qa-id", help="Process only this qa_id.")
    p.add_argument("--workers", type=int,
                   help="Override max_workers.")
    p.add_argument("--print-raw", action="store_true",
                   help="Print raw LLM outputs.")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable resume (reprocess all).")
    p.add_argument("--grounding-jsonl",
                   default=str(DEFAULT_GROUNDING_JSONL),
                   help="Path to the released Search-based Grounding JSONL.")
    p.add_argument("--choice-jsonl",
                   default=str(DEFAULT_CHOICE_JSONL),
                   help="Path to the released Search-based VQA JSONL.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_existing_qa_ids(jsonl_path: Path) -> Set[str]:
    """Load qa_ids from an existing result JSONL for resume."""
    ids: Set[str] = set()
    if not jsonl_path.exists():
        return ids
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            item = json.loads(raw)
            qid = str(item.get("qa_id", "")).strip()
            if qid:
                ids.add(qid)
        except Exception:
            continue
    return ids


# ---------------------------------------------------------------------------
# Single-entry processors
# ---------------------------------------------------------------------------

def process_grounding_entry(
    entry: GroundingEntry,
    cfg: Config,
    vis_dir: Path,
) -> Dict[str, Any]:
    """Process one grounding entry."""
    image_path = resolve_image_path(cfg.dataset_root, entry.image_rel_path)
    client = build_openai_client(cfg)

    try:
        result = run_grounding(
            client=client,
            cfg=cfg,
            image_path=image_path,
            question_text=entry.question_en,
            artifact_dir=vis_dir / entry.qa_id,
        )
    except Exception as exc:
        result = {
            "task_type": "ground_from_question",
            "predicted_bbox": None,
            "error": str(exc),
            "rounds_used": 0,
        }

    pred_bbox = result.get("predicted_bbox")
    iou = best_iou_to_targets(pred_bbox, entry.target_boxes)

    return {
        "task": TASK_SEARCH_GROUNDING,
        "sample_id": entry.sample_id,
        "qa_id": entry.qa_id,
        "image_path": entry.image_rel_path,
        "question_en": entry.question_en,
        "answer_en": entry.answer_en,
        "predicted_bbox": pred_bbox,
        "ground_truth_boxes": entry.target_boxes,
        "best_iou_to_target": round(iou, 4),
        "is_correct_iou50": iou >= 0.5,
        "confidence": result.get("confidence", 0.0),
        "rounds_used": result.get("rounds_used", 0),
        "resolved_entity": result.get("resolved_entity"),
        "visual_description": result.get("visual_description"),
        "sub_questions": result.get("sub_questions"),
        "search_traces": result.get("traces"),
        "raw_evidence": result.get("raw_evidence"),
        "entity_verification": result.get("entity_verification"),
        "candidate_scores": result.get("candidate_scores"),
        "joint_selection": result.get("joint_selection"),
        "selected_candidate_id": result.get("selected_candidate_id"),
        "num_candidates": result.get("num_candidates", 0),
        "num_ref_images": result.get("num_ref_images", 0),
        "error": result.get("error"),
    }


def process_choice_entry(
    entry: ChoiceEntry,
    cfg: Config,
    vis_dir: Path,
) -> Dict[str, Any]:
    """Process one choice entry."""
    image_path = resolve_image_path(cfg.dataset_root, entry.image_rel_path)
    client = build_openai_client(cfg)

    try:
        result = run_choice(
            client=client,
            cfg=cfg,
            image_path=image_path,
            bbox_xyxy=entry.bbox_xyxy,
            options=entry.options,
            artifact_dir=vis_dir / entry.qa_id,
        )
    except Exception as exc:
        result = {
            "task_type": "choose_option_for_bbox",
            "selected_index": None,
            "error": str(exc),
        }

    selected = result.get("selected_index")
    is_correct = (selected == entry.answer_index) if selected is not None else False

    return {
        "task": TASK_SEARCH_VQA,
        "sample_id": entry.sample_id,
        "qa_id": entry.qa_id,
        "image_path": entry.image_rel_path,
        "bbox_xyxy": entry.bbox_xyxy,
        "options": entry.options,
        "answer_index": entry.answer_index,
        "answer_en": entry.answer_en,
        "selected_index": selected,
        "is_correct": is_correct,
        "confidence": result.get("confidence", 0.0),
        "reason": result.get("reason", ""),
        "option_entities": result.get("option_entities"),
        "raw_output": result.get("raw_output", ""),
        "result_payload": result.get("result_payload"),
        "error": result.get("error"),
    }


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def run_grounding_batch(
    cfg: Config,
    entries: List[GroundingEntry],
    output_json: Path,
    output_jsonl: Path,
    vis_dir: Path,
    resume: bool,
) -> Dict[str, Any]:
    """Run grounding on all entries with concurrency and resume."""
    existing = load_existing_qa_ids(output_jsonl) if resume else set()
    todo = [e for e in entries if e.qa_id not in existing]
    print(f"[grounding] {len(entries)} total, {len(existing)} done, "
          f"{len(todo)} to process")

    results: List[Dict[str, Any]] = []

    # reload existing results
    if resume and output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if raw:
                results.append(json.loads(raw))

    if todo:
        n_workers = min(cfg.max_workers, len(todo))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _safe_grounding, entry, cfg, vis_dir
                ): entry.qa_id
                for entry in todo
            }
            for fut in as_completed(futures):
                qa_id = futures[fut]
                result = fut.result()
                results.append(result)
                # append to JSONL immediately for resume
                with output_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                status = "OK" if result.get("is_correct_iou50") else "FAIL"
                iou = result.get("best_iou_to_target", 0)
                print(f"  [{status}] {qa_id}  iou={iou:.3f}  "
                      f"rounds={result.get('rounds_used', 0)}")

    # summary
    n = len(results)
    correct = sum(1 for r in results if r.get("is_correct_iou50"))
    summary = {
        "task": TASK_SEARCH_GROUNDING,
        "model": cfg.openai_model,
        "num_questions": n,
        "num_correct_iou50": correct,
        "accuracy_iou50": round(correct / n, 4) if n else 0.0,
        "mean_iou": round(
            sum(r.get("best_iou_to_target", 0) for r in results) / n, 4
        ) if n else 0.0,
    }
    output_json.write_text(
        json.dumps({**summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def run_choice_batch(
    cfg: Config,
    entries: List[ChoiceEntry],
    output_json: Path,
    output_jsonl: Path,
    vis_dir: Path,
    resume: bool,
) -> Dict[str, Any]:
    """Run choice on all entries with concurrency and resume."""
    existing = load_existing_qa_ids(output_jsonl) if resume else set()
    todo = [e for e in entries if e.qa_id not in existing]
    print(f"[choice] {len(entries)} total, {len(existing)} done, "
          f"{len(todo)} to process")

    results: List[Dict[str, Any]] = []

    if resume and output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if raw:
                results.append(json.loads(raw))

    if todo:
        n_workers = min(cfg.max_workers, len(todo))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _safe_choice, entry, cfg, vis_dir
                ): entry.qa_id
                for entry in todo
            }
            for fut in as_completed(futures):
                qa_id = futures[fut]
                result = fut.result()
                results.append(result)
                with output_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                status = "OK" if result.get("is_correct") else "FAIL"
                print(f"  [{status}] {qa_id}  "
                      f"sel={result.get('selected_index')} "
                      f"ans={result.get('answer_index')}")

    n = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    summary = {
        "task": TASK_SEARCH_VQA,
        "model": cfg.openai_model,
        "num_questions": n,
        "num_correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
    }
    output_json.write_text(
        json.dumps({**summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _safe_grounding(entry, cfg, vis_dir):
    print(f"  [processing] {entry.qa_id}")
    try:
        return process_grounding_entry(entry, cfg, vis_dir)
    except Exception as exc:
        return {
            "task": entry.task,
            "sample_id": entry.sample_id,
            "qa_id": entry.qa_id,
            "predicted_bbox": None,
            "error": str(exc),
            "best_iou_to_target": 0.0,
            "is_correct_iou50": False,
        }


def _safe_choice(entry, cfg, vis_dir):
    print(f"  [processing] {entry.qa_id}")
    try:
        return process_choice_entry(entry, cfg, vis_dir)
    except Exception as exc:
        return {
            "task": entry.task,
            "sample_id": entry.sample_id,
            "qa_id": entry.qa_id,
            "selected_index": None,
            "error": str(exc),
            "is_correct": False,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    task = {
        "search-grounding": "grounding",
        "search-vqa": "choice",
    }.get(args.task, args.task)
    cfg = load_config(args.env_file)
    if args.model:
        cfg.openai_model = args.model.strip()
    if args.workers:
        cfg.max_workers = max(1, args.workers)
    cfg.print_raw = args.print_raw

    # resolve jsonl paths relative to cwd, repo root, then script dir
    script_dir = Path(__file__).resolve().parent
    grounding_jsonl = Path(args.grounding_jsonl)
    if not grounding_jsonl.exists():
        grounding_jsonl = CODE_ROOT / args.grounding_jsonl
    if not grounding_jsonl.exists():
        grounding_jsonl = script_dir / args.grounding_jsonl
    choice_jsonl = Path(args.choice_jsonl)
    if not choice_jsonl.exists():
        choice_jsonl = CODE_ROOT / args.choice_jsonl
    if not choice_jsonl.exists():
        choice_jsonl = script_dir / args.choice_jsonl

    model_dir = sanitize_model_name(cfg.openai_model)
    output_root = Path(cfg.output_root)
    if not output_root.is_absolute():
        output_root = CODE_ROOT / output_root
    out_root = ensure_dir(output_root / model_dir)
    resume = not args.no_resume

    print(f"[config] model={cfg.openai_model}  base_url={cfg.openai_base_url}")
    print(f"[config] max_search_rounds={cfg.max_search_rounds}  "
          f"saliency_top_k={cfg.saliency_top_k}  "
          f"max_workers={cfg.max_workers}")

    if task in ("grounding", "all"):
        entries = load_grounding_entries(
            str(grounding_jsonl),
            sample_id=args.sample_id,
            qa_id=args.qa_id,
        )
        if args.limit:
            entries = entries[:args.limit]

        g_json = out_root / "grounding.json"
        g_jsonl = out_root / "grounding.jsonl"
        g_vis = ensure_dir(out_root / "grounding_vis")

        if not resume:
            g_jsonl.unlink(missing_ok=True)

        summary = run_grounding_batch(
            cfg, entries, g_json, g_jsonl, g_vis, resume
        )
        print(f"\n[GROUNDING] acc@iou50 = {summary['accuracy_iou50']:.4f}  "
              f"({summary['num_correct_iou50']}/{summary['num_questions']})")

    if task in ("choice", "all"):
        entries = load_choice_entries(
            str(choice_jsonl),
            sample_id=args.sample_id,
            qa_id=args.qa_id,
        )
        if args.limit:
            entries = entries[:args.limit]

        c_json = out_root / "choice.json"
        c_jsonl = out_root / "choice.jsonl"
        c_vis = ensure_dir(out_root / "choice_vis")

        if not resume:
            c_jsonl.unlink(missing_ok=True)

        summary = run_choice_batch(
            cfg, entries, c_json, c_jsonl, c_vis, resume
        )
        print(f"\n[CHOICE] acc = {summary['accuracy']:.4f}  "
              f"({summary['num_correct']}/{summary['num_questions']})")

    print("\n[done]")


if __name__ == "__main__":
    main()
