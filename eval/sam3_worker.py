#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 stdin/stdout JSONL worker.

The main eval runner starts this script in an environment that can import
`sam3`. Each request is a JSON object with `cmd: "predict"`, `image_path`, and
an optional bbox prompt in `box: [x1, y1, x2, y2]`.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import traceback
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def normalize_best_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    masks_arr = np.asarray(masks)
    scores_arr = np.asarray(scores).reshape(-1)
    if scores_arr.size == 0:
        raise ValueError("SAM3 returned empty scores")

    best_flat = int(np.argmax(scores_arr))
    best_score = float(scores_arr[best_flat])
    if masks_arr.ndim == 2:
        best_mask = masks_arr
    elif masks_arr.ndim == 3:
        best_mask = masks_arr[best_flat]
    elif masks_arr.ndim == 4:
        best_mask = masks_arr.reshape(-1, masks_arr.shape[-2], masks_arr.shape[-1])[best_flat]
    else:
        raise ValueError(f"Unsupported masks shape: {masks_arr.shape}")

    if best_mask.dtype != np.bool_:
        best_mask = best_mask > 0.5
    return best_mask.astype(np.uint8), best_score


def mask_bbox_xywh(mask01: np.ndarray) -> list[int]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def mask_to_png_b64(mask01: np.ndarray) -> str:
    mask255 = (mask01 * 255).astype(np.uint8)
    image = Image.fromarray(mask255, mode="L")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict(processor: Sam3Processor, model: Any, req: dict[str, Any]) -> dict[str, Any]:
    image = None
    raw_path = str(req.get("image_path", "")).strip()
    if raw_path:
        image_path = Path(raw_path).resolve()
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
    if image is None and req.get("image_b64"):
        image = Image.open(BytesIO(base64.b64decode(str(req["image_b64"])))).convert("RGB")
    if image is None:
        raise FileNotFoundError(f"Image path not found: {raw_path}")

    inference_state = processor.set_image(image)
    raw_box = req.get("box")
    box = None
    if raw_box is not None:
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise ValueError("box must be [x1, y1, x2, y2]")
        box = np.array(raw_box, dtype=np.float32)[None, :]

    masks, scores, _ = model.predict_inst(
        inference_state,
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True,
    )
    best_mask, best_score = normalize_best_mask(masks, scores)
    return {
        "ok": True,
        "mask_png_b64": mask_to_png_b64(best_mask),
        "score": best_score,
        "area": int(best_mask.sum()),
        "bbox_xywh": mask_bbox_xywh(best_mask),
        "height": int(best_mask.shape[0]),
        "width": int(best_mask.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3 worker")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = args.checkpoint.strip() or None
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        device=args.device,
        enable_inst_interactivity=True,
    )
    processor = Sam3Processor(model, device=args.device)
    emit({"ok": True, "event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                emit({"ok": True, "event": "pong"})
            elif cmd == "predict":
                emit(predict(processor, model, req))
            else:
                emit({"ok": False, "error": f"Unsupported cmd: {cmd}"})
        except Exception as exc:
            emit({"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=3)})


if __name__ == "__main__":
    main()
