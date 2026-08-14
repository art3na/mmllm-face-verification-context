# Multimodal Large Language Models for Face Verification: The Impact of Adversarial and Cooperative Context

This repository contains the reproduction code for the paper:

**Multimodal Large Language Models for Face Verification: The Impact of Adversarial and Cooperative Context**
*Arturas Nakvosas*
Institute of Data Science and Digital Technologies, Vilnius University
Corresponding author: arturas.nakvosas@mif.vu.lt

## Overview

The project evaluates Multimodal Large Language Models (MM-LLMs) on face verification tasks under different context modes:
- **Base**: Standard verification question ("Are these the same person?").
- **Attack**: Adversarial context implying they are different when they are same, or vice versa.
- **Assist**: Cooperative context reinforcing the ground truth.

## Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

You will need an OpenAI-compatible API endpoint (e.g., vLLM or standard OpenAI API) serving your MM-LLM.

## Dataset Structure

Ensure your datasets are structured as follows:

```
dtb/
  AgeDB/          # Images
  AgeDB_30_genuine.csv
  AgeDB_30_impostor.csv
  calfw/          # Images inside 'aligned images'
  cplfw/
  lfw/
  archive/
    lfw-deepfunneled/
```

Adjust paths in `run_experiments.py` `DATASETS` config if necessary.

## Data Preparation

If you need to generate the pair CSV files (e.g., if you only have the raw images or original text pair lists), you can use the scripts provided in `dtb/`:

### AgeDB
```bash
cd dtb/AgeDB
python generate.py --root_img_dir . --output_dir ..
```

### CALFW
```bash
cd dtb/calfw
# Ensure pairs_CALFW.txt is present in dtb/calfw/
python generate_pairs.py --source_file pairs_CALFW.txt --output_dir .
```

### CPLFW
```bash
cd dtb/cplfw
# Ensure pairs_CPLFW.txt is present in dtb/cplfw/
python generate_pairs.py --source_file pairs_CPLFW.txt --output_dir .
```

### LFW (Labeled Faces in the Wild)
The `run_experiments.py` script supports LFW using the "View 1" (DevTest) protocol by default, as the necessary CSV files (`matchpairsDevTest.csv`, `mismatchpairsDevTest.csv`) are typically included in `dtb/lfw`.
- **Default path**: `dtb/lfw`
- **Images**: Expected in `dtb/lfw/lfw-deepfunneled` (configurable via `--lfw_dir`).
- **Usage**: LFW is now included in the default datasets list. To run only LFW:
  ```bash
  python run_experiments.py --model qwen3-vl-8b --base_url ... --datasets lfw
  ```

## Usage

### 1. Run Experiments

Use `run_experiments.py` to evaluate a model. This script sends image pairs to the MM-LLM and records the responses and log-probabilities.

```bash
python run_experiments.py \
  --base_url "http://localhost:8000/v1" \
  --api_key "your-api-key" \ # Optional if using vLLM
  --model "qwen3-vl-8b" \
  --datasets age_db_30 calfw \
  --max_workers 64
```

Arguments:
- `--base_url`: URL of the LLM API.
- `--model`: Model name to query.
- `--datasets`: List of datasets to evaluate (default: `age_db_30`, `calfw`, `cplfw`).
- `--max_workers`: Number of parallel requests.
- `--context_format`: `text` (default) runs the full natural-language context ensemble (`TEXT_PAIRS`); `json` runs the structured-context probe (see below).

Results will be saved in `results/<dataset>/<model>/`.

### Structured-context (JSON) probe

To test whether contextual sensitivity persists when the context is delivered as
structured metadata instead of a natural-language statement, run the JSON probe:

```bash
python run_experiments.py \
  --base_url "http://localhost:8000/v1" \
  --model "qwen3-vl-8b" \
  --datasets lfw \
  --context_format json
```

The context channel then carries a single machine-style JSON record with a boolean
claim and no natural-language surface:

```json
{"task": "face_verification", "same_person": true}
```

with `same_person` set consistently with the ground truth in Assist mode and
flipped in Attack mode. All 15 prompts are used, so results are comparable to the
natural-language ensemble. The Base (no-context) condition is skipped because it
is identical to the Base runs of `--context_format text` — run the text ensemble
first (or at least its base condition) on the same dataset/model. Result files are
named `<model>_P<p>_J0_logprobs_{attack,assist}.csv` alongside the `_T*` files.

### 2. Plot Results

Use `plot_aggregated_det_from_csv_with_metrics.py` to analyze the results, calculate metrics (EER, AUC, etc.), and generate DET curves.

```bash
python plot_aggregated_det_from_csv_with_metrics.py \
  --results_dir "results" \
  --plots_dir "plots" \
  --datasets age_db_30 calfw \
  --models qwen3-vl-8b facellm-8b
```

Arguments:
- `--results_dir`: Root directory containing results (default: `results`).
- `--plots_dir`: Directory to save generated plots (default: `plots`).
- `--datasets`: Datasets to process.
- `--models`: List of models to include in comparison plots.
- `--context_format`: `text` (default) analyzes the natural-language ensemble (`_T*` files); `json` analyzes the structured-context probe (`_J*` files, reusing Base runs from the `_T*` files). All JSON-mode outputs (metrics CSVs, plots, significance rows) carry a `_json` suffix so they never overwrite the text-ensemble outputs.

Example for the JSON probe:

```bash
python plot_aggregated_det_from_csv_with_metrics.py \
  --results_dir "results" \
  --datasets lfw \
  --models qwen3-vl-8b \
  --context_format json
```

## License

[License Information]
