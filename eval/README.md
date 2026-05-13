# 📊 WebEyes Evaluation

This directory contains the evaluation tools for **WebEyes**. It supports two
common workflows:

1. **Run inference and evaluate** OpenAI-compatible vision models with
   [`infer.py`](infer.py).
2. **Evaluate existing predictions** from any model or method with
   [`evaluate.py`](evaluate.py).


## Setup

Install the Python packages used by the evaluator and inference runner:

```bash
pip install openai pillow numpy
```

For OpenAI-compatible inference, create an environment file:

```bash
cp eval/.env.example eval/.env
```

Then edit `eval/.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

You can also override these values from the command line with `--api-key`,
`--base-url`, and `--model`.

## Run Model Inference

Use `infer.py` to call a vision model through an OpenAI-compatible chat
completion API. It currently supports:

- `search-grounding`
- `search-vqa`
- `all` for both grounding and VQA

Run a quick smoke test:

```bash
python eval/infer.py --env-file eval/.env --dataset-root ./dataset --task all --limit 10
```

Run one task:

```bash
python eval/infer.py --env-file eval/.env --dataset-root ./dataset --task search-grounding
python eval/infer.py --env-file eval/.env --dataset-root ./dataset --task search-vqa
```

By default, outputs are written to:

```text
outputs/eval/<model>/
```

The inference runner writes prediction files and, unless `--no-evaluate` is
set, immediately scores them:

```text
grounding_predictions.jsonl
grounding_summary.json
grounding_scored.jsonl
vqa_predictions.jsonl
vqa_summary.json
vqa_scored.jsonl
run_summary.json
```

For each task, these files have different roles:

| File pattern | Meaning |
| --- | --- |
| `*_predictions.jsonl` | Raw model predictions keyed by `qa_id`, such as predicted boxes, selected choices, raw responses, and model-side errors. |
| `*_scored.jsonl` | Per-item evaluation results, including ground-truth targets, item-level metrics, correctness flags, and evaluation errors. |
| `*_summary.json` | Aggregate metrics for the task, such as `mean_iou`, `recall_iou50`, `accuracy`, item counts, and error counts. |
| `run_summary.json` | One run-level summary collecting all evaluated tasks in the output directory. |

For example, grounding writes `grounding_predictions.jsonl`,
`grounding_scored.jsonl`, and `grounding_summary.json` so prediction reuse,
error analysis, and final reporting stay separate.

Useful options:

| Option | Description |
| --- | --- |
| `--dataset-root PATH` | Dataset root containing `data/search_*.jsonl`. |
| `--output-dir PATH` | Write results to a custom directory. |
| `--limit N` | Process only the first `N` records after filtering. |
| `--sample-id ID` | Run one sample only. |
| `--qa-id ID` | Run one QA item only. |
| `--workers N` | Number of concurrent API calls. |
| `--retry-times N` | Retry failed API calls. |
| `--coord-mode auto\|abs\|rel1000` | Requested bbox coordinate format for grounding. |
| `--max-boxes N` | Maximum parsed grounding boxes before selecting the best one. |
| `--min-box-area-ratio FLOAT` | Drop grounding boxes smaller than this image-area ratio. |
| `--max-box-area-ratio FLOAT` | Drop grounding boxes larger than this image-area ratio. |
| `--nms-iou-threshold FLOAT` | Label-wise NMS IoU threshold for grounding boxes. |
| `--no-nms` | Disable grounding NMS. |
| `--print-raw` | Print raw model responses. |
| `--no-evaluate` | Write predictions without computing metrics. |
| `--grounding-jsonl PATH` | Override the grounding dataset file and ignore `--dataset-root` for grounding. |
| `--vqa-jsonl PATH` | Override the VQA dataset file and ignore `--dataset-root` for VQA. |
| `--seg-jsonl PATH` | Override the segmentation dataset file used with SAM3 and ignore `--dataset-root` for segmentation. |

## Grounding to Segmentation with SAM3

For grounding models, `infer.py` can pass predicted boxes to SAM3 and evaluate
the resulting masks on WebEyes-Seg:

```cmd
python eval/infer.py ^
  --env-file eval/.env ^
  --dataset-root ./dataset ^
  --task search-grounding ^
  --sam3 ^
  --sam-python C:\path\to\python.exe ^
  --sam-checkpoint C:\path\to\sam3.pt ^
  --sam-device cuda
```

The example above uses Windows Command Prompt line continuation. On Linux,
macOS, or Git Bash, use backslashes instead:

```bash
python eval/infer.py \
  --env-file eval/.env \
  --dataset-root ./dataset \
  --task search-grounding \
  --sam3 \
  --sam-python /path/to/python \
  --sam-checkpoint /path/to/sam3.pt \
  --sam-device cuda
```

This adds:

```text
sam3_masks/
sam3_seg_predictions.jsonl
sam3_seg_summary.json
sam3_seg_scored.jsonl
```

`--sam-python` must point to a Python environment that can import SAM3.

## Evaluate Existing Predictions

Use `evaluate.py` when predictions already exist. This script does not call any
model.

### Grounding

Prediction JSONL:

```jsonl
{"qa_id":"sample_0001__qa_004","predicted_bbox":[73,443,472,1106]}
```

Evaluate:

```bash
python eval/evaluate.py \
  --task search-grounding \
  --prediction-jsonl ./outputs/pred_grounding.jsonl \
  --model-name MODEL
```

Accepted bbox keys are `predicted_bbox`, `bbox_xyxy`, and `bbox`.

### Segmentation

If masks are stored as one PNG per `qa_id`:

```bash
python eval/evaluate.py \
  --task search-seg \
  --prediction-mask-dir ./outputs/masks \
  --mask-name-template "{qa_id}.png" \
  --model-name MODEL
```

If masks are listed in a JSONL file:

```jsonl
{"qa_id":"sample_0001__qa_004","predicted_mask_path":"./outputs/masks/sample_0001__qa_004.png"}
```

```bash
python eval/evaluate.py \
  --task search-seg \
  --prediction-jsonl ./outputs/pred_seg.jsonl \
  --model-name MODEL
```

The evaluator also accepts `mask_path`, `prediction_mask_path`, or an inline
`mask_png_b64` field. Predicted masks are resized to the ground-truth mask
shape with nearest-neighbor interpolation before scoring.

### VQA

Prediction JSONL:

```jsonl
{"qa_id":"sample_0001__qa_004","selected_index":3}
```

Evaluate:

```bash
python eval/evaluate.py \
  --task search-vqa \
  --prediction-jsonl ./outputs/pred_vqa.jsonl \
  --model-name MODEL
```

Accepted answer keys are `selected_index`, `choice_index`, `answer_index`,
`predicted_id`, and `prediction`.

## Evaluation Options

| Option | Description |
| --- | --- |
| `--dataset-jsonl PATH` | Override the default dataset file for the selected task. Defaults to `./dataset/data/*.jsonl`. |
| `--output-json PATH` | Save summary plus per-item results. |
| `--output-jsonl PATH` | Save per-item scored rows. |
| `--sample-id ID` | Evaluate one sample only. |
| `--qa-id ID` | Evaluate one QA item only. |
| `--limit N` | Evaluate only the first `N` records after filtering. |
| `--mask-threshold FLOAT` | Foreground threshold for mask images in `[0, 255]`. |

## Metrics

| Task | Main metrics |
| --- | --- |
| WebEyes-Ground | `mean_iou`, `recall_iou50`, `accuracy_iou50` |
| WebEyes-Seg | `mask_g_iou`, `mask_c_iou`, `mask_iou_ge_50_rate` |
| WebEyes-VQA | `accuracy` |

All summaries also include item counts and error counts.
