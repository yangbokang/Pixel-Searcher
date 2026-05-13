#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline evaluation script for grounding and choice outputs.

Usage:
    python evaluate.py --grounding <path-to-grounding.jsonl>
    python evaluate.py --choice <path-to-choice.jsonl>
    python evaluate.py --grounding <path> --choice <path> --detail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grounding and choice results.")
    parser.add_argument("--grounding", help="Path to grounding result JSONL.")
    parser.add_argument("--choice", help="Path to choice result JSONL.")
    parser.add_argument("--detail", action="store_true", help="Print per-sample detail.")
    return parser.parse_args()


def load_results(jsonl_path: str) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        print(f"[warn] File not found: {jsonl_path}")
        return []

    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


def evaluate_grounding(results: List[Dict[str, Any]], detail: bool = False) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "with_prediction": 0, "correct_iou50": 0, "accuracy_iou50": 0.0, "mean_iou": 0.0, "errors": 0}

    correct = 0
    has_prediction = 0
    errors = 0
    ious: List[float] = []

    for row in results:
        iou = float(row.get("best_iou_to_target", 0.0))
        ious.append(iou)
        is_correct = bool(row.get("is_correct_iou50"))
        if is_correct:
            correct += 1
        if row.get("predicted_bbox") is not None:
            has_prediction += 1
        if row.get("error"):
            errors += 1
        if detail:
            status = "OK" if is_correct else "FAIL"
            print(
                f"[{status}] {row.get('qa_id', '?')} "
                f"iou={iou:.3f} rounds={row.get('rounds_used', 0)} "
                f"pred={row.get('predicted_bbox')}"
            )

    return {
        "total": total,
        "with_prediction": has_prediction,
        "correct_iou50": correct,
        "accuracy_iou50": round(correct / total, 4),
        "mean_iou": round(sum(ious) / total, 4),
        "errors": errors,
    }


def evaluate_choice(results: List[Dict[str, Any]], detail: bool = False) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0, "answered": 0, "correct": 0, "accuracy": 0.0, "errors": 0}

    correct = 0
    answered = 0
    errors = 0

    for row in results:
        is_correct = bool(row.get("is_correct"))
        if is_correct:
            correct += 1
        if row.get("selected_index") is not None:
            answered += 1
        if row.get("error"):
            errors += 1
        if detail:
            status = "OK" if is_correct else "FAIL"
            print(
                f"[{status}] {row.get('qa_id', '?')} "
                f"sel={row.get('selected_index')} ans={row.get('answer_index')}"
            )

    return {
        "total": total,
        "answered": answered,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "errors": errors,
    }


def print_summary(grounding_metrics: Optional[Dict[str, Any]], choice_metrics: Optional[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if grounding_metrics:
        metrics = grounding_metrics
        print("\nGROUNDING")
        print(f"{'Total':20s} {metrics['total']}")
        print(f"{'With prediction':20s} {metrics['with_prediction']}")
        print(f"{'Correct (IoU>=0.5)':20s} {metrics['correct_iou50']}")
        print(f"{'acc@iou50':20s} {metrics['accuracy_iou50']:.4f}")
        print(f"{'Mean IoU':20s} {metrics['mean_iou']:.4f}")
        print(f"{'Errors':20s} {metrics['errors']}")

    if choice_metrics:
        metrics = choice_metrics
        print("\nCHOICE")
        print(f"{'Total':20s} {metrics['total']}")
        print(f"{'Answered':20s} {metrics['answered']}")
        print(f"{'Correct':20s} {metrics['correct']}")
        print(f"{'Accuracy':20s} {metrics['accuracy']:.4f}")
        print(f"{'Errors':20s} {metrics['errors']}")

    print("=" * 60)


def main() -> None:
    args = parse_args()
    if not args.grounding and not args.choice:
        print("Please provide at least one of --grounding or --choice.")
        return

    grounding_metrics: Optional[Dict[str, Any]] = None
    choice_metrics: Optional[Dict[str, Any]] = None

    if args.grounding:
        grounding_results = load_results(args.grounding)
        grounding_metrics = evaluate_grounding(grounding_results, detail=args.detail) if grounding_results else None

    if args.choice:
        choice_results = load_results(args.choice)
        choice_metrics = evaluate_choice(choice_results, detail=args.detail) if choice_results else None

    print_summary(grounding_metrics, choice_metrics)


if __name__ == "__main__":
    main()
