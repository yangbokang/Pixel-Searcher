# Pixel-Searcher Method

`method/` contains the search-based Pixel-Searcher pipeline for WebEyes-Ground
and WebEyes-VQA. It runs model inference, optional
external search, candidate scoring, visual grounding, and answer selection.

Use `eval/evaluate.py` for final benchmark metrics.

## Files

- `run_benchmark.py`: batch runner with resume and concurrency.
- `multi_round_agent.py`: search, entity resolution, grounding, and VQA logic.
- `saliency_filter.py`: candidate filtering, ranking, NMS, crops, and visualizations.
- `config.py`: config loading, API client setup, dataset parsing, paths, and helpers.
- `.env.example`: runtime configuration template.

## Setup

Run from the repository root:

```bash
cp method/.env.example method/.env
```

Edit `method/.env`:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
SERPER_API_KEY=your_serper_api_key_here

DATASET_ROOT=./dataset
OUTPUT_ROOT=./outputs
```

`SERPER_API_KEY` is used by the search-based reasoning pipeline. Relative
`DATASET_ROOT` and `OUTPUT_ROOT` paths are resolved from the repository root.

Expected data files:

```text
dataset/data/search_grounding.jsonl
dataset/data/search_vqa.jsonl
dataset/annotations/dataset.jsonl
```

Ground-truth labels, boxes, masks, and answer indices should not be used during
model inference except where the task definition provides them as input. VQA
uses the provided target bbox and option list.

## Run

```bash
python method/run_benchmark.py --env-file method/.env --task grounding --limit 10
python method/run_benchmark.py --env-file method/.env --task choice --limit 10
python method/run_benchmark.py --env-file method/.env --task all
```

Task aliases:

- `grounding` or `search-grounding`
- `choice` or `search-vqa`
- `all`

Useful options:

- `--model`: override `OPENAI_MODEL`.
- `--limit`: process a subset.
- `--sample-id`, `--qa-id`: filter examples.
- `--workers`: override concurrency.
- `--print-raw`: print raw model outputs.
- `--no-resume`: reprocess completed examples.
- `--grounding-jsonl`, `--choice-jsonl`: override task JSONL paths.

## Outputs

With the default config, outputs are written to:

```text
outputs/<model-name>/
```

Typical files:

```text
grounding.json
grounding.jsonl
grounding_vis/
choice.json
choice.jsonl
choice_vis/
```

Grounding rows include `predicted_bbox` and `best_iou_to_target`.
VQA rows include `selected_index` and `is_correct`.

## Final Evaluation

```bash
python eval/evaluate.py \
  --task search-grounding \
  --prediction-jsonl ./outputs/MODEL/grounding.jsonl \
  --model-name MODEL

python eval/evaluate.py \
  --task search-vqa \
  --prediction-jsonl ./outputs/MODEL/choice.jsonl \
  --model-name MODEL
```
