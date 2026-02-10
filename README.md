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

Results will be saved in `results/<dataset>/<model>/`.

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

## License

[License Information]
