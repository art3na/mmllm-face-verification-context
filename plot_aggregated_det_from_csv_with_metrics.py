"""
Script: plot_aggregated_det_from_csv_with_metrics.py

Loads LFW logprobs CSV files (one per prompt + text prior + mode), reconstructs
(y_true, y_pred, p_yes) for each run, and:

- Plots aggregated DET curves with uncertainty bands across prompts (per model).
- Computes micro- and macro-averaged metrics (accuracy, precision,
  recall, F1, FAR, FRR, ROC-AUC, Brier, EER, NLL) for each mode.
- Macro metrics are reported:
    * over prompts (each prompt_idx aggregated over all text priors)
    * over text priors (each text_idx aggregated over all prompts)
- Additionally, plots DET curves per mode (Base / Attack / Assist)
  overlaying all models in MODEL_NAME_LS, with legends that include EER.
- NEW: Saves all metrics for all models and modes into a CSV file:
    all_models_metrics.csv

Expected CSV format (from your perfom_test):
    idx, y_true, y_pred, p_yes, lp_yes, lp_no, mode, raw_text

Expected filename pattern (per model):
    results/{MODEL_NAME}/{MODEL_NAME}_P{idx}_T{text}_llfw_logprobs_{mode}.csv
Where mode ∈ {base, attack, assist}.
"""

import os
import re
import glob
import argparse
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.ticker as mticker
import math

def _pct_formatter(decimals=2):
    def fmt(x, pos=None):
        return f"{x*100:.{decimals}f}%"
    return mticker.FuncFormatter(fmt)

def percent_formatter(sig=3):
    def fmt(x, pos=None):
        if x == 0:
            return "0%"
        p = x * 100.0
        # nice integers: 1 -> 1%, 10 -> 10%, 100 -> 100%
        if abs(p - round(p)) < 1e-12:
            return f"{int(round(p))}%"
        # otherwise: significant digits (no trailing zeros like 100.000%)
        return f"{p:.{sig}g}%"
    return mticker.FuncFormatter(fmt)


from sklearn.metrics import (
    det_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)

# =========================
# Config
# =========================

# =========================
# Config (Defaults)
# =========================

# List of models for cross-model DET plots + metrics export
DEFAULT_MODEL_NAME_LS = [
    "facellm-8b",
    "gemma-3-12b-it",
    "intern_vl2-8B",
    "internvl3_5-8b",
    "qwen2-vl-7b",
    "qwen3-vl-8b",
    "resnet100",
    "lvface-b",
]

# Consistent colors for each model
MODEL_COLORS = {
    "facellm-8b":      "#1f77b4",  # blue +
    "gemma-3-12b-it":  "#ff7f0e",  # orange +
    "intern_vl2-8B":   "#2ca02c",  # green +
    "internvl3_5-8b":  "#d62728",  # red +
    "qwen2-vl-7b":     "#9467bd",  # purple +
    "qwen3-vl-8b":     "#8c564b",  # brown +
    "resnet100":       "#17becf",  # teal +
    "lvface-b":       "#e377c2",  # pink
}

# Formal display names
MODEL_PRETTY_NAMES = {
    "facellm-8b":      "FaceLLM 8B",
    "gemma-3-12b-it":  "Gemma-3 12B IT",
    "intern_vl2-8B":   "InternVL2 8B",
    "internvl3_5-8b":  "InternVL3.5 8B",
    "qwen2-vl-7b":     "Qwen2-VL 7B",
    "qwen3-vl-8b":     "Qwen3-VL 8B",
    "resnet100":       "ResNet-100",
    "lvface-b":       "LVFace-Vit-b",
}

# Prompts List (Keep consistent with run_experiments.py)
PROMPTS_LIST = [
    'Are these two images showing the same person? Answer only “yes” or “no”.',
    'Compare [Image 1] and [Image 2]. Do they depict the same individual? Reply with “yes” or “no” only.',
    'Face verification task: decide if [Image 1] and [Image 2] are of the same person. Respond strictly with “yes” or “no”.',
    'Determine whether [Image 1] and [Image 2] show the same person. Output only “yes” or “no”, with no extra text.',
    'Check if the faces in [Image 1] and [Image 2] belong to the same person. Answer “yes” or “no” only.',
    'Your task is to compare two face images. Are [Image 1] and [Image 2] the same person? Reply only “yes” or “no”.',
    'Binary decision: do [Image 1] and [Image 2] depict the same identity? Answer with exactly “yes” or “no”.',
    'Given [Image 1] and [Image 2], determine if they are the same individual. Respond with a single word: “yes” or “no”.',
    'Face matching: Are the two provided images of the same person? Return only “yes” or “no”.',
    'Evaluate if [Image 1] and [Image 2] correspond to the same human face. Output strictly “yes” or “no”.',
    'Compare the two faces. Are they the same person? Answer with exactly “yes” or “no”, nothing else.',
    'Task: face identity comparison. Do [Image 1] and [Image 2] show the same person? Reply only “yes” or “no”.',
    'Decide if there is identity match between [Image 1] and [Image 2]. Respond strictly with “yes” or “no”.',
    'Question: Are these two face images from the same person? Only answer “yes” or “no”.',
    'Determine if there is a face identity match between [Image 1] and [Image 2]. Output only “yes” or “no”.'
]

SINGLE_MODEL_DET_PNG = "{model}_aggregated_det_all_prompts_{dataset}.png"
ALL_METRICS_CSV_TEMPLATE = "all_models_metrics_{dataset}.csv"


# =========================
# Data loading
# =========================

def load_runs_from_csv(model_name: str, base_results_root: str):
    """
    Scan all CSVs for a given model and group them by mode.

    Returns:
        all_runs: dict {mode_name: [(y_det, p_det), ...]}
                  for DET aggregation (NaN probs removed).
        metric_data: dict {mode_name: [dict_per_run, ...]}
            dict_per_run = {
                "prompt_idx": int,
                "text_idx": int,
                "y_true": 1D np.array of int,
                "y_pred": 1D np.array of int,
                "p":      1D np.array of float (can contain NaN)
            }
    """
    results_dir = os.path.join(base_results_root, model_name)

    csv_glob_pattern = os.path.join(
        results_dir,
        f"{model_name}_P*_T*_logprobs_*.csv"
    )

    filename_re = re.compile(
        r".*_P(\d+)_T(\d+)_logprobs_(base|attack|assist)\.csv$"
    )

    all_runs = {
        "base":   [],
        "attack": [],
        "assist": [],
    }

    metric_data = {
        "base":   [],
        "attack": [],
        "assist": [],
    }

    csv_files = sorted(glob.glob(csv_glob_pattern))
    if not csv_files:
        print(f"[{model_name}] No CSV files found for pattern: {csv_glob_pattern}")
        return all_runs, metric_data

    print(f"[{model_name}] Found {len(csv_files)} CSV files:")
    for path in csv_files:
        print("  ", path)

    for path in csv_files:
        fname = os.path.basename(path)
        m = filename_re.match(fname)
        if not m:
            print(f"[{model_name}] Skipping {fname}: filename does not match expected pattern.")
            continue

        prompt_idx = int(m.group(1))
        text_idx = int(m.group(2))
        mode = m.group(3)  # 'base', 'attack', or 'assist'

        if mode == 'base' and text_idx > 0:
            print(f"[{model_name}] Skipping {fname}: filename base text idx is greater than 0")
            continue 


        df = pd.read_csv(path)

        idx_all = df["idx"].to_numpy(dtype=int) if "idx" in df.columns else np.arange(len(df))

        if not {"y_true", "y_pred", "p_yes"}.issubset(df.columns):
            print(f"[{model_name}] Skipping {fname}: required columns 'y_true', 'y_pred', 'p_yes' not found.")
            continue

        # Full label & prediction arrays (for metrics)
        y_true_all = df["y_true"].to_numpy(dtype=int)
        y_pred_all = df["y_pred"].to_numpy(dtype=int)
        # Probabilities (may contain empty strings)
        p_all = pd.to_numeric(df["p_yes"], errors="coerce").to_numpy()

        if len(y_true_all) == 0:
            print(f"[{model_name}] Skipping {fname}: empty file.")
            continue

        # Store data for metrics (including NaNs in p_all)
        metric_data[mode].append(
            {
                "prompt_idx": prompt_idx,
                "text_idx": text_idx,
                "idx": idx_all,
                "y_true": y_true_all,
                "y_pred": y_pred_all,
                "p": p_all,
            }
        )

        # For DET we need label + prob with NaNs removed
        mask = ~np.isnan(p_all)
        y_det = y_true_all[mask]
        p_det = p_all[mask]

        if len(y_det) == 0:
            print(f"[{model_name}] Warning: {fname} has no valid probabilities for DET.")
        else:
            all_runs[mode].append((y_det, p_det))

        print(
            f"[{model_name}] Loaded {fname}: prompt {prompt_idx}, text {text_idx} mode={mode}, "
            f"pairs={len(y_true_all)}, valid_probs_for_DET={len(y_det)}"
        )

    for mode in ["base", "attack", "assist"]:
        print(
            f"[{model_name}] Mode '{mode}': "
            f"{len(metric_data[mode])} runs (metrics), "
            f"{len(all_runs[mode])} runs (DET)"
        )

    return all_runs, metric_data


# =========================
# DET utilities
# =========================

def compute_eer_from_det(fpr, fnr):
    """
    Compute EER (Equal Error Rate) from det_curve outputs.
    """
    diff = np.abs(fpr - fnr)
    i_eer = np.argmin(diff)
    eer_fpr = fpr[i_eer]
    eer_fnr = fnr[i_eer]
    eer = 0.5 * (eer_fpr + eer_fnr)
    return eer, eer_fpr, eer_fnr


def compute_frr_at_far(fpr, fnr, far_target=0.01):
    """
    Compute FRR at a specific FAR operating point (default 1%).
    """
    # Find index where FPR is closest to far_target
    idx = np.argmin(np.abs(fpr - far_target))
    return fnr[idx]


def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    n = len(y_prob)
    
    for i in range(n_bins):
        idx = binids == i
        if not np.any(idx):
            continue
            
        acc_in_bin = np.mean(y_true[idx] == (y_prob[idx] > 0.5))
        conf_in_bin = np.mean(y_prob[idx])  # or max(p, 1-p) if multiclass, but here p is p_yes
        
        # For binary p_yes, confidence is usually defined as max(p, 1-p) for the predicted class
        # BUT standard ECE often just takes difference between avg_prob and avg_acc.
        # Let's align with: |acc - conf| * (n_bin / n)
        # where conf is average probability of the predicted class.
        
        # Actually for binary verification p(match), if we predict match (p>0.5), conf is p.
        # If we predict non-match (p<=0.5), conf is 1-p.
        
        y_pred_bin = (y_prob[idx] > 0.5).astype(int)
        # correct if y_true == y_pred_bin
        acc_bin = np.mean(y_true[idx] == y_pred_bin)
        
        prob_pred_class = np.where(y_pred_bin == 1, y_prob[idx], 1.0 - y_prob[idx])
        conf_bin = np.mean(prob_pred_class)
        
        ece += np.abs(acc_bin - conf_bin) * (np.sum(idx) / n)
        
    return ece


def aggregate_det_for_mode(runs, n_points=200, eps=1e-6, log_grid=True):
    """
    Aggregate DET curves for one mode across multiple runs (prompts).

    runs: list of (y, p) for that mode.

    Returns:
        fpr_grid
        fnr_mean
        fnr_lo  (10th percentile)
        fnr_hi  (90th percentile)
    """
    if log_grid:
        fpr_grid = np.logspace(np.log10(eps), 0, n_points)
    else:
        fpr_grid = np.linspace(0.0, 1.0, n_points)

    fnr_samples = []

    for (y, p) in runs:
        if y is None or p is None or len(np.unique(y)) < 2:
            continue

        fpr, fnr, _ = det_curve(y, p)

        # ensure sorted by FPR
        order = np.argsort(fpr)
        fpr = np.clip(fpr[order], eps, 1 - eps)
        fnr = fnr[order]

        #smooth
        u_fpr, inv = np.unique(fpr, return_inverse=True)
        fnr_u = np.zeros_like(u_fpr)
        for k in range(len(u_fpr)):
            fnr_u[k] = fnr[inv == k].min()

        fpr, fnr = u_fpr, fnr_u

        # interpolate FNR on the common FPR grid
        fnr_interp = np.interp(fpr_grid, fpr, fnr, left=fnr[0], right=fnr[-1])
        fnr_samples.append(fnr_interp)

    if not fnr_samples:
        return None, None, None, None

    fnr_samples = np.vstack(fnr_samples)
    fnr_mean = fnr_samples.mean(axis=0)
    fnr_lo   = np.percentile(fnr_samples, 10, axis=0)
    fnr_hi   = np.percentile(fnr_samples, 90, axis=0)

    return fpr_grid, fnr_mean, fnr_lo, fnr_hi


def plot_aggregated_det(all_runs, model_name, save_path=None, log_axes=True):
    """
    Per-model DET aggregation.

    all_runs: dict
      {
        "base":   [(y,p), (y,p), ...],
        "attack": [...],
        "assist": [...],
      }

    Legend shows mode name + EER from the mean curve.

    Returns:
        eer_dict: {mode_name: eer_mean_curve (float, fraction)}
    """
    eps = 1e-6
    eps_y = 1e-4
    plt.figure()
    eer_dict = {}

    # Equal-error line (secondary style)
    if log_axes:
        diag_x = np.logspace(np.log10(eps), 0, 200)
    else:
        diag_x = np.linspace(0.0, 1.0, 200)
    plt.plot(diag_x, diag_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    for mode_name, runs in all_runs.items():
        if not runs:
            print(f"[{model_name}] No runs for mode={mode_name}, skipping in DET plot.")
            continue

        aggr = aggregate_det_for_mode(runs, log_grid=log_axes)
        if aggr[0] is None:
            print(f"[{model_name}] No valid DET data for mode={mode_name}, skipping in DET plot.")
            continue

        fpr_grid, fnr_mean, fnr_lo, fnr_hi = aggr

        # EER from mean curve (for legend + saving)
        eer_mean_curve, _, _ = compute_eer_from_det(fpr_grid, fnr_mean)
        eer_percent = eer_mean_curve * 100.0
        eer_dict[mode_name] = eer_mean_curve
        
        # Operating point: FRR @ FAR=1%
        frr_at_1pct = compute_frr_at_far(fpr_grid, fnr_mean, far_target=0.01)
        frr_pct = frr_at_1pct * 100.0

        # Model pretty name
        pretty_name = MODEL_PRETTY_NAMES.get(model_name, model_name)

        # main mean curve
        # Just stick to default cycle colors for "same model, different mode" plot 
        # OR if user wants strict consistency, we might have an issue plotting multiple modes for one model?
        # The prompt says: "Keep model colors consistent across every plot (same model = same color everywhere)."
        # But this function plots ONE model, MULTIPLE modes. 
        # If we use the model's color for ALL modes, we can't distinguish modes by color.
        # However, usually "consistent model colors" applies when comparing multiple models.
        # For a single-model plot comparing modes, we should probably distinct colors for modes.
        # Let's keep default colors for this specific SINGLE-MODEL plot unless user specified otherwise.
        # User constraint: "Keep model colors consistent across every plot (same model = same color everywhere)."
        # This strongly implies even in single model plots? 
        # If I use blue for FaceLLM in multi-model plot, should I use blue for FaceLLM-Base, FaceLLM-Attack?
        # That would make them indistinguishable. 
        # Interpretation: logic applies when models are compared. 
        # For single model plot, we are comparing MODES. So we need different colors for modes.
        # Let's stick to default colors or a mode-specific palette here.
        
        label_str = f"{mode_name} (EER={eer_percent:.2f}%, FRR@1%FAR={frr_pct:.2f}%)"
        
        line, = plt.plot(
            fpr_grid,
            fnr_mean,
            label=label_str
        )
        color = line.get_color()

        # uncertainty band
        plt.fill_between(
            fpr_grid,
            fnr_lo,
            fnr_hi,
            alpha=0.2,
            color=color,
        )

        # Also print mean ± std EER across runs
        eers = []
        for (y, p) in runs:
            if y is None or p is None or len(np.unique(y)) < 2:
                continue
            fpr, fnr, _ = det_curve(y, p)
            eer, _, _ = compute_eer_from_det(fpr, fnr)
            eers.append(eer)
        if eers:
            print(
                f"[{model_name}] Mode {mode_name}: "
                f"EER mean={np.mean(eers)*100:.2f}%, "
                f"std={np.std(eers)*100:.2f}% (across prompts, prob-based)"
            )

    if log_axes:
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(eps, 1.0)
        plt.ylim(eps_y, 1.0)
    else:
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

    ax = plt.gca()

    # If you're using log axes, these locators keep ticks at 1e-5, 1e-4, ..., 1
    if log_axes:
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.xaxis.set_major_formatter(_pct_formatter(decimals=3 if log_axes else 1))
    ax.yaxis.set_major_formatter(_pct_formatter(decimals=3 if log_axes else 1))

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(f"Aggregated DET curves across prompts\nModel: {model_name}")
    plt.grid(True, which="both")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[{model_name}] Saved aggregated DET plot to: {save_path}")
        plt.close()
    else:
        plt.show()

    return eer_dict


def plot_det_all_models(all_models_runs, dataset_name, save_dir="plots", log_axes=True):
    """
    Plot DET curves for each mode, overlaying all models.

    all_models_runs: dict
      {
        model_name: {
          "base":   [(y,p), ...],
          "attack": [...],
          "assist": [...],
        },
        ...
      }
    Produces 3 images (base/attack/assist) defined in ALL_MODELS_DET_PNG.

    Legend shows model name + EER from the mean curve.
    """
    eps = 1e-6
    y_eps = 1e-4
    modes = ["base", "attack", "assist"]

    for mode in modes:
        plt.figure()

        # Equal-error line (secondary style)
        if log_axes:
            diag_x = np.logspace(np.log10(eps), 0, 200)
        else:
            diag_x = np.linspace(0.0, 1.0, 200)
        plt.plot(diag_x, diag_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

        any_plotted = False

        for model_name, runs_dict in all_models_runs.items():
            runs = runs_dict.get(mode, [])
            if not runs:
                print(f"[{model_name}] No runs for mode={mode}, skipping in multi-model DET.")
                continue

            aggr = aggregate_det_for_mode(runs, log_grid=log_axes)
            if aggr[0] is None:
                print(f"[{model_name}] No valid DET data for mode={mode}, skipping in multi-model DET.")
                continue

            fpr_grid, fnr_mean, _, _ = aggr

            # EER from mean curve for legend
            eer_mean_curve, _, _ = compute_eer_from_det(fpr_grid, fnr_mean)
            eer_percent = eer_mean_curve * 100.0
            
            # Operating point: FRR @ FAR=1%
            frr_at_1pct = compute_frr_at_far(fpr_grid, fnr_mean, far_target=0.01)
            frr_pct = frr_at_1pct * 100.0
            
            # Setup Plotting Style
            pretty_name = MODEL_PRETTY_NAMES.get(model_name, model_name)
            color = MODEL_COLORS.get(model_name, None)

            plt.plot(
                fpr_grid,
                fnr_mean,
                label=f"{pretty_name} (EER={eer_percent:.2f}%, FRR@1%FAR={frr_pct:.2f}%)",
                color=color
            )
            any_plotted = True

        if not any_plotted:
            print(f"[ALL MODELS] No data plotted for mode={mode}.")
            plt.close()
            continue

        if log_axes:
            plt.xscale("log")
            plt.yscale("log")
            plt.xlim(eps, 1.0)
            plt.ylim(y_eps, 1.0)
        else:
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(percent_formatter(sig=3))
        ax.yaxis.set_major_formatter(percent_formatter(sig=3))

        plt.xlabel("False Acceptance Rate (FAR)")
        plt.ylabel("False Rejection Rate (FRR)")
        plt.title(f"DET curves ({mode}) across models")
        plt.grid(True, which="both")
        plt.legend(loc="lower left", fontsize=9, framealpha=0.9)

        save_path = os.path.join(save_dir, f"det_{mode}_all_models_{dataset_name}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[ALL MODELS] Saved DET plot for mode={mode} to: {save_path}")
        plt.close()


# =========================
# Metrics across prompts / text
# =========================

def group_runs_by_prompt(metric_runs):
    """
    Group (prompt_idx, text_idx) runs into per-prompt aggregates.

    Returns:
        prompt_runs: list of dicts with keys:
            {
              "prompt_idx": int,
              "y_true": np.array,
              "y_pred": np.array,
              "p": np.array,
            }
    """
    grouped = {}

    for run in metric_runs:
        pid = run["prompt_idx"]
        if pid not in grouped:
            grouped[pid] = {
                "prompt_idx": pid,
                "y_true": [],
                "y_pred": [],
                "p": [],
            }
        grouped[pid]["y_true"].append(run["y_true"])
        grouped[pid]["y_pred"].append(run["y_pred"])
        grouped[pid]["p"].append(run["p"])

    prompt_runs = []
    for pid, g in grouped.items():
        prompt_runs.append(
            {
                "prompt_idx": pid,
                "y_true": np.concatenate(g["y_true"]),
                "y_pred": np.concatenate(g["y_pred"]),
                "p": np.concatenate(g["p"]),
            }
        )

    return prompt_runs


def group_runs_by_text(metric_runs):
    """
    Group (prompt_idx, text_idx) runs into per-text-prior aggregates.

    Returns:
        text_runs: list of dicts with keys:
            {
              "text_idx": int,
              "y_true": np.array,
              "y_pred": np.array,
              "p": np.array,
            }
    """
    grouped = {}

    for run in metric_runs:
        tid = run["text_idx"]
        if tid not in grouped:
            grouped[tid] = {
                "text_idx": tid,
                "y_true": [],
                "y_pred": [],
                "p": [],
            }
        grouped[tid]["y_true"].append(run["y_true"])
        grouped[tid]["y_pred"].append(run["y_pred"])
        grouped[tid]["p"].append(run["p"])

    text_runs = []
    for tid, g in grouped.items():
        text_runs.append(
            {
                "text_idx": tid,
                "y_true": np.concatenate(g["y_true"]),
                "y_pred": np.concatenate(g["y_pred"]),
                "p": np.concatenate(g["p"]),
            }
        )

    return text_runs

import scipy.stats as st

def ms(values, conf=0.95):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)

    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    
    mean_val = float(np.mean(vals))
    
    # sample standard deviation
    std_val = float(np.std(vals, ddof=1)) if n > 1 else np.nan

    if n < 2:
        return (mean_val, std_val, np.nan, np.nan)

    sem = std_val / np.sqrt(n)

    alpha = 1 - conf
    t_crit = st.t.ppf(1 - alpha/2, df=n - 1)
    margin = t_crit * sem

    ci_lo = mean_val - margin
    ci_hi = mean_val + margin
    return mean_val, std_val, float(ci_lo), float(ci_hi)

'''

def ms(values):
    if not values:
        return (np.nan, np.nan, np.nan, np.nan)
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]  # remove NaNs
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals)) # population or sample? usually sample for CI
    
    # For CI, we need n and standard error
    n = len(vals)
    if n < 2:
        return (mean_val, std_val, np.nan, np.nan)

    # Standard error of the mean
    sem = scipy.stats.sem(vals,)
    
    # t-statistic for 95% confidence
    # degrees of freedom = n - 1
    t_crit = scipy.stats.t.ppf(0.975, df=n-1)
    
    margin = t_crit * sem
    ci_lo = mean_val - margin
    ci_hi = mean_val + margin
    
    return mean_val, std_val, float(ci_lo), float(ci_hi)
'''

def compute_macro_for_groups(grouped_runs):
    """
    grouped_runs: list of dicts with keys:
      - y_true, y_pred, p

    Returns:
        metrics: dict
          {
            'accuracy': {'mean':..., 'std':...}, ...
            'far': {'mean':..., 'std':...}, ... (fractions)
            'eer': {'mean':..., 'std':...}, ... (fractions)
            'nll': {'mean':..., 'std':...}, ...
          }
    """
    if not grouped_runs:
        return {}

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    auc_list, brier_list, eer_list, nll_list, ece_list = [], [], [], [], []
    far_list, frr_list = [], []

    for run in grouped_runs:
        y = run["y_true"]
        y_pred = run["y_pred"]
        p = run["p"]

        # threshold-based
        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        cm   = confusion_matrix(y, y_pred)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            frr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
            far_list.append(far)
            frr_list.append(frr)

        # prob-based for this group
        mask = ~np.isnan(p)
        if mask.any():
            y_prob = y[mask]
            p_prob = p[mask]

            if len(np.unique(y_prob)) > 1:
                auc_run = roc_auc_score(y_prob, p_prob)
                auc_list.append(auc_run)

                fpr, fnr, _ = det_curve(y_prob, p_prob)
                eer_run, _, _ = compute_eer_from_det(fpr, fnr)
                eer_list.append(eer_run)

                nll_run = log_loss(y_prob, p_prob)
                nll_list.append(nll_run)

            brier_run = brier_score_loss(y_prob, p_prob)
            brier_list.append(brier_run)
            
            ece_run = compute_ece(y_prob, p_prob)
            ece_list.append(ece_run)

    metrics = {
        "accuracy": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(acc_list))),
        "precision": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(prec_list))),
        "recall": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(rec_list))),
        "f1": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(f1_list))),
        "far": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(far_list))),
        "frr": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(frr_list))),
        "roc_auc": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(auc_list))),
        "brier": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(brier_list))),
        "ece": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(ece_list))),
        "eer": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(eer_list))),  # fractions
        "nll": dict(zip(["mean", "std", "ci_lo", "ci_hi"], ms(nll_list))),
    }
    return metrics


def compute_metrics_for_mode(metric_runs, mode_name):
    """
    metric_runs: list of dicts, each with:
        "prompt_idx", "text_idx", "y_true", "y_pred", "p"
    mode_name: "base" / "attack" / "assist"

    Prints micro- and macro-averaged metrics for this mode:
      - micro pooled over all (prompt, text) runs
      - macro over prompts
      - macro over text priors

    Returns:
        metrics_mode: dict
          {
            'mode': mode_name,
            'micro': {...},
            'macro_prompts': {...},
            'macro_text': {...},
          }
    """
    metrics_mode = {
        "mode": mode_name,
        "micro": {},
        "macro_prompts": {},
        "macro_text": {},
    }

    if not metric_runs:
        print(f"\n[Metrics] Mode '{mode_name}': no runs, skipping.")
        return metrics_mode

    print(f"\n[Metrics] Mode '{mode_name}':")

    # ---------- Micro (pooled across prompts and text priors) ----------
    y_micro_list = []
    y_pred_micro_list = []
    p_micro_list = []

    for run in metric_runs:
        y_micro_list.append(run["y_true"])
        y_pred_micro_list.append(run["y_pred"])
        p_micro_list.append(run["p"])

    y_micro = np.concatenate(y_micro_list)
    y_pred_micro = np.concatenate(y_pred_micro_list)
    p_micro = np.concatenate(p_micro_list)

    # threshold-based micro metrics
    acc_micro  = accuracy_score(y_micro, y_pred_micro)
    prec_micro = precision_score(y_micro, y_pred_micro, zero_division=0)
    rec_micro  = recall_score(y_micro, y_pred_micro, zero_division=0)
    f1_micro   = f1_score(y_micro, y_pred_micro, zero_division=0)
    cm_micro   = confusion_matrix(y_micro, y_pred_micro)

    # FAR / FRR (micro, threshold-based)
    if cm_micro.shape == (2, 2):
        tn, fp, fn, tp = cm_micro.ravel()
        far_micro = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        frr_micro = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    else:
        far_micro = np.nan
        frr_micro = np.nan

    # prob-based micro metrics
    mask_valid = ~np.isnan(p_micro)
    auc_micro = brier_micro = eer_micro = nll_micro = None

    if mask_valid.any():
        y_prob_micro = y_micro[mask_valid]
        p_prob_micro = p_micro[mask_valid]

        if len(np.unique(y_prob_micro)) > 1:
            auc_micro = roc_auc_score(y_prob_micro, p_prob_micro)
            fpr, fnr, _ = det_curve(y_prob_micro, p_prob_micro)
            eer_micro, _, _ = compute_eer_from_det(fpr, fnr)
            # Negative log-likelihood (NLL)
            nll_micro = log_loss(y_prob_micro, p_prob_micro)

        brier_micro = brier_score_loss(y_prob_micro, p_prob_micro)

    print("  Micro (pooled across all prompts and text priors):")
    print(f"    Accuracy:  {acc_micro:.4f}")
    print(f"    Precision: {prec_micro:.4f}")
    print(f"    Recall:    {rec_micro:.4f}")
    print(f"    F1:        {f1_micro:.4f}")
    if not np.isnan(far_micro):
        print(f"    FAR:       {far_micro*100:.2f}%")
    if not np.isnan(frr_micro):
        print(f"    FRR:       {frr_micro*100:.2f}%")
    if auc_micro is not None:
        print(f"    ROC-AUC:   {auc_micro:.4f}")
    if brier_micro is not None:
        print(f"    Brier:     {brier_micro:.4f}")
    if eer_micro is not None:
        print(f"    EER:       {eer_micro*100:.2f}%")
    if nll_micro is not None:
        print(f"    NLL:       {nll_micro:.4f}")
    print("    Confusion matrix (micro):")
    print(cm_micro)

    metrics_mode["micro"] = {
        "accuracy": acc_micro,
        "precision": prec_micro,
        "recall": rec_micro,
        "f1": f1_micro,
        "far": far_micro,
        "frr": frr_micro,
        "roc_auc": auc_micro,
        "brier": brier_micro,
        "ece": None, # todo micro ECE
        "eer": eer_micro,
        "nll": nll_micro,
    }

    # ---------- Macro over prompts ----------
    prompt_runs = group_runs_by_prompt(metric_runs)
    macro_prompts = compute_macro_for_groups(prompt_runs)
    metrics_mode["macro_prompts"] = macro_prompts

    print("  Macro (over prompts, mean ± std, [95% CI]):")

    for k, v in macro_prompts.items():

        if np.isnan(v["mean"]):
            continue

        if k in ("far", "frr", "eer"):
            # report in %
            mean_pct = v["mean"] * 100.0
            std_pct = v["std"] * 100.0
            lo_pct  = v["ci_lo"] * 100.0
            hi_pct  = v["ci_hi"] * 100.0
            print(f"    {k.upper():<8}: {mean_pct:.2f}% ± {std_pct:.2f}%  (95% CI: [{lo_pct:.2f}%, {hi_pct:.2f}%])")
        else:
            print(f"    {k.capitalize():<8}: {v['mean']:.4f} ± {v['std']:.4f}  (95% CI: [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}])")

    # ---------- Macro over text priors ----------
    text_runs = group_runs_by_text(metric_runs)
    macro_text = compute_macro_for_groups(text_runs)
    metrics_mode["macro_text"] = macro_text

    print("  Macro (over text priors, mean ± std, [95% CI]):")
    for k, v in macro_text.items():
        if np.isnan(v["mean"]):
            continue
        if k in ("far", "frr", "eer"):
            mean_pct = v["mean"] * 100.0
            std_pct = v["std"] * 100.0
            lo_pct  = v["ci_lo"] * 100.0
            hi_pct  = v["ci_hi"] * 100.0
            print(f"    {k.upper():<8}: {mean_pct:.2f}% ± {std_pct:.2f}%  (95% CI: [{lo_pct:.2f}%, {hi_pct:.2f}%])")
        else:
            print(f"    {k.capitalize():<8}: {v['mean']:.4f} ± {v['std']:.4f}  (95% CI: [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}])")

    return metrics_mode


def compute_all_metrics(metric_data):
    """
    metric_data: dict {mode_name: [runs...]}

    Calls compute_metrics_for_mode for each mode.

    Returns:
        metrics_all_modes: dict {mode: metrics_mode_dict}
    """
    metrics_all_modes = {}
    for mode_name in ["base", "attack", "assist"]:
        metrics_all_modes[mode_name] = compute_metrics_for_mode(
            metric_data[mode_name],
            mode_name
        )
    return metrics_all_modes


def flatten_metrics_for_csv(model_name, metrics_all_modes):
    """
    Convert metrics structure to a list of rows for CSV export.

    Returns:
        rows: list of dicts with keys:
          model, mode, level, metric, value, std
    """
    rows = []

    for mode_name, mdict in metrics_all_modes.items():
        # Micro
        micro = mdict.get("micro", {})
        for metric, val in micro.items():
            rows.append(
                {
                    "model": model_name,
                    "mode": mode_name,
                    "level": "micro",
                    "metric": metric,
                    "value": float(val) if val is not None and not np.isnan(val) else np.nan,
                    "std": np.nan,
                    "ci95_lo": np.nan,
                    "ci95_hi": np.nan,
                }
            )

        # Macro prompts
        macro_prompts = mdict.get("macro_prompts", {})
        for metric, mstats in macro_prompts.items():
            rows.append(
                {
                    "model": model_name,
                    "mode": mode_name,
                    "level": "macro_prompts",
                    "metric": metric,
                    "value": float(mstats["mean"]) if mstats["mean"] is not None else np.nan,
                    "std": float(mstats["std"]) if mstats["std"] is not None else np.nan,
                    "ci95_lo": float(mstats["ci_lo"]) if mstats["ci_lo"] is not None else np.nan,
                    "ci95_hi": float(mstats["ci_hi"]) if mstats["ci_hi"] is not None else np.nan,
                }
            )

        # Macro text
        macro_text = mdict.get("macro_text", {})
        for metric, mstats in macro_text.items():
            rows.append(
                {
                    "model": model_name,
                    "mode": mode_name,
                    "level": "macro_text",
                    "metric": metric,
                    "value": float(mstats["mean"]) if mstats["mean"] is not None else np.nan,
                    "std": float(mstats["std"]) if mstats["std"] is not None else np.nan,
                    "ci95_lo": float(mstats["ci_lo"]) if mstats["ci_lo"] is not None else np.nan,
                    "ci95_hi": float(mstats["ci_hi"]) if mstats["ci_hi"] is not None else np.nan,
                }
            )

    return rows


# =========================
# Global Prompt Analysis
# =========================

def analyze_global_prompt_performance(all_metric_data_global):
    """
    Aggregates data across ALL datasets and models per prompt index.
    Ranks prompts by Accuracy.
    """
    print("\n" + "=" * 80)
    print("GLOBAL PROMPT PERFORMANCE ANALYSIS (Across all models & datasets)")
    print("=" * 80)

    for mode in ["base", "attack", "assist"]:
        print(f"\n--- Mode: {mode.upper()} ---")
        
        runs = all_metric_data_global.get(mode, [])
        if not runs:
            print("No data found.")
            continue

        # Group by prompt_idx
        prompt_groups = {}
        for r in runs:
            pid = r["prompt_idx"]
            if pid not in prompt_groups:
                prompt_groups[pid] = {"y_true": [], "y_pred": [], "p": []}
            prompt_groups[pid]["y_true"].append(r["y_true"])
            prompt_groups[pid]["y_pred"].append(r["y_pred"])
            prompt_groups[pid]["p"].append(r["p"])

        # Calculate metrics for each prompt
        prompt_metrics = []
        for pid, g in prompt_groups.items():
            y_all = np.concatenate(g["y_true"])
            yp_all = np.concatenate(g["y_pred"])
            p_all = np.concatenate(g["p"])

            acc = accuracy_score(y_all, yp_all)
            
            # EER
            mask = ~np.isnan(p_all)
            eer = np.nan
            if mask.any():
                y_prob = y_all[mask]
                p_prob = p_all[mask]
                if len(np.unique(y_prob)) > 1:
                    fpr, fnr, _ = det_curve(y_prob, p_prob)
                    eer, _, _ = compute_eer_from_det(fpr, fnr)

            prompt_text = PROMPTS_LIST[pid] if pid < len(PROMPTS_LIST) else f"Unknown Prompt {pid}"
            prompt_metrics.append({
                "id": pid,
                "text": prompt_text,
                "acc": acc,
                "eer": eer
            })

        # Sort by Accuracy (Descending)
        prompt_metrics.sort(key=lambda x: x["acc"], reverse=True)

        print("\nTOP 3 PROMPTS (Best Accuracy):")
        for i, pm in enumerate(prompt_metrics[:3]):
            eer_str = f"{pm['eer']*100:.2f}%" if not np.isnan(pm['eer']) else "N/A"
            print(f"  {i+1}. [ID:{pm['id']}] Acc: {pm['acc']:.4f}, EER: {eer_str}")
            print(f"     \"{pm['text']}\"")

        print("\nBOTTOM 3 PROMPTS (Worst Accuracy):")
        for i, pm in enumerate(prompt_metrics[-3:][::-1]): # reverse to show worst first
            eer_str = f"{pm['eer']*100:.2f}%" if not np.isnan(pm['eer']) else "N/A"
            print(f"  {i+1}. [ID:{pm['id']}] Acc: {pm['acc']:.4f}, EER: {eer_str}")
            print(f"     \"{pm['text']}\"")
            
        # Save to CSV
        df = pd.DataFrame(prompt_metrics)
        df_csv_path = f"global_prompt_performance_analysis.csv"
        df.to_csv(df_csv_path, index=False)
        print(f"\nSaved global prompt analysis to: {df_csv_path}")


def aggregate_across_datasets_old(dataset_metrics_store):
    """
    dataset_metrics_store: {model: {dataset: {mode: {level: {metric: value_dict}}}}
    level is typically 'macro_prompts' (which we care about most).
    """
    print("\n" + "=" * 80)
    print("GLOBAL AGGREGATION ACROSS ALL DATASETS")
    print("=" * 80)
    
    rows = []
    
    # We want to average the 'mean' values of metrics across datasets
    # metrics structure: metrics["macro_prompts"]["accuracy"]["mean"] 
    # But compute_all_metrics returns {mode: metrics_mode}
    # metrics_mode has "macro_prompts" -> {metric: {mean:..., std:..., ci...}}
    
    for model, d_metrics in dataset_metrics_store.items():
        # d_metrics: {dataset: metrics_all_modes}
        
        # We need to collect values for each (mode, metric) across datasets
        # Key: (mode, metric) -> list of means
        aggregator = {} 
        
        for dataset, metrics_all_modes in d_metrics.items():
            for mode, mdict in metrics_all_modes.items():
                macro = mdict.get("macro_prompts", {})
                for metric, stats in macro.items():
                    if np.isnan(stats["mean"]):
                        continue
                        
                    key = (mode, metric)
                    if key not in aggregator:
                        aggregator[key] = []
                    aggregator[key].append(stats["mean"])
        
        # Compute Macro-Macro stats
        for (mode, metric), values in aggregator.items():
            if not values:
                continue
            
            vals = np.array(values)
            mean_across_datasets = float(np.mean(vals))
            std_across_datasets = float(np.std(vals))
            
            rows.append({
                "model": model,
                "mode": mode,
                "metric": metric,
                "mean_across_datasets": mean_across_datasets,
                "std_across_datasets": std_across_datasets,
                "n_datasets": len(vals)
            })
            
    if not rows:
        print("No data to aggregate.")
        return

    df = pd.DataFrame(rows)
    out_path = "all_models_metrics_aggregated_across_datasets.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved aggregated metrics across datasets to: {out_path}")
    
    # Print a small summary for Accuracy
    print("\n--- Summary: Accuracy (Mean across datasets) ---")
    acc_df = df[df["metric"] == "accuracy"]
    if not acc_df.empty:
        print(acc_df.pivot(index="model", columns="mode", values="mean_across_datasets"))


def aggregate_across_datasets(dataset_metrics_store, levels=("macro_prompts", "macro_text"), conf=0.95):
    """
    Aggregates metrics across datasets, using DATASET as the statistical unit (n = #datasets).

    dataset_metrics_store: {model: {dataset: metrics_all_modes}}
    metrics_all_modes: {mode: {"macro_prompts": {metric: {mean,std,ci_lo,ci_hi}}, ...}}

    levels: which levels to aggregate across datasets (e.g. ("macro_prompts","macro_text"))
    """
    print("\n" + "=" * 80)
    print("GLOBAL AGGREGATION ACROSS ALL DATASETS (dataset-level CI)")
    print("=" * 80)

    rows = []

    for model, d_metrics in dataset_metrics_store.items():
        # key: (level, mode, metric) -> list of (dataset_name, value)
        aggregator = {}

        for dataset, metrics_all_modes in d_metrics.items():
            for mode, mdict in metrics_all_modes.items():
                for level in levels:
                    block = mdict.get(level, {})
                    if not isinstance(block, dict):
                        continue

                    for metric, stats in block.items():
                        if not isinstance(stats, dict):
                            continue
                        v = stats.get("mean", np.nan)
                        if v is None or np.isnan(v):
                            continue

                        key = (level, mode, metric)
                        aggregator.setdefault(key, []).append((dataset, float(v)))

        # Compute mean/std/CI across datasets (n = number of datasets with data)
        for (level, mode, metric), pairs in aggregator.items():
            dataset_names = [d for d, _ in pairs]
            values = [v for _, v in pairs]

            mean_val, std_val, ci_lo, ci_hi = ms(values, conf=conf)

            rows.append({
                "model": model,
                "mode": mode,
                "level": level,
                "metric": metric,
                "mean_across_datasets": mean_val,
                "std_across_datasets": std_val,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_datasets": len(values),
                "datasets_used": ",".join(dataset_names),
            })

    if not rows:
        print("No data to aggregate.")
        return

    df = pd.DataFrame(rows)
    out_path = "all_models_metrics_aggregated_across_datasets_macro_prompts_and_macro_text.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved aggregated metrics across datasets to: {out_path}")

    # Print compact Accuracy summary for both levels
    print("\n--- Summary: Accuracy (Mean across datasets, 95% CI) ---")
    acc_df = df[df["metric"] == "accuracy"].copy()
    if not acc_df.empty:
        acc_df = acc_df.sort_values(["level", "model", "mode"])
        # show in a readable table-like print
        cols = ["level", "model", "mode", "mean_across_datasets", "ci_lo", "ci_hi", "n_datasets"]
        print(acc_df[cols].to_string(index=False))

def _binomtest_two_sided(k, n, p=0.5):
    # scipy >=1.7 has binomtest; older has binom_test
    if hasattr(scipy.stats, "binomtest"):
        return scipy.stats.binomtest(k, n, p=p, alternative="two-sided").pvalue
    return scipy.stats.binom_test(k, n, p=p, alternative="two-sided")

def mcnemar_exact_from_paired_correct(correct_a, correct_b):
    """
    correct_a, correct_b: boolean arrays aligned per example.
    Returns: (b, c, p_value)
      b = a correct, b wrong
      c = a wrong, b correct
    Exact McNemar uses Binomial(n=b+c, p=0.5) on min(b,c).
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)

    # discordant
    b01 = np.sum(a & ~b)  # A correct, B wrong
    b10 = np.sum(~a & b)  # A wrong, B correct
    n = b01 + b10
    if n == 0:
        return int(b01), int(b10), 1.0

    pval = _binomtest_two_sided(k=min(b01, b10), n=n, p=0.5)
    return int(b01), int(b10), float(pval)

def unpaired_tests_per_model(metric_data_m, model_name, dataset_name, out_csv_path=None):
    """
    metric_data_m: dict with keys base/attack/assist -> list of runs
    Each run has: prompt_idx, text_idx, y_true, y_pred, (optional idx)
    Pairs by (prompt_idx, text_idx) intersection.
    """
    def to_run_map(runs):
        m = {}
        for r in runs:
            key = (r["prompt_idx"], r["text_idx"])
            m[key] = r
        return m

    base_map   = to_run_map(metric_data_m.get("base", [])) #15 pairs
    attack_map = to_run_map(metric_data_m.get("attack", [])) #300 pairs
    assist_map = to_run_map(metric_data_m.get("assist", [])) # 300 pairs


    def dz_from_diffs(diffs):
        diffs = np.asarray(diffs, float)
        diffs = diffs[np.isfinite(diffs)]
        return float(np.mean(diffs) / np.std(diffs, ddof=1))

    def power_paired_t(n, dz, alpha=0.05):
        df = n - 1
        ncp = dz * math.sqrt(n)
        tcrit = scipy.stats.t.ppf(1 - alpha/2, df)
        return float(scipy.stats.nct.sf(tcrit, df, ncp) + scipy.stats.nct.cdf(-tcrit, df, ncp))

    def unpaired_accuracy_test(map_a, map_b, label_a, label_b):
        keys_a = sorted(set(map_a.keys()))
        keys_b = sorted(set(map_b.keys()))
        if not keys_a or not keys_b:
            return None
        
        base_by_prompt = {}
        for k in keys_a:
            p = k[0]
            base_by_prompt.setdefault(p, []).append(k)

        b_by_prompt = {}
        for k in keys_b:
            p = k[0]
            b_by_prompt.setdefault(p, []).append(k)

        prompts = sorted(set(base_by_prompt.keys()) & set(b_by_prompt.keys()))
        if not prompts:
            return None
        

        acc_a = []
        acc_b = []
        for p in prompts:
            # base prompt accuracy (mean if multiple base runs exist for that prompt)
            a_vals = []
            for k in base_by_prompt[p]:
                ra = map_a[k]
                a_vals.append(accuracy_score(ra["y_true"], ra["y_pred"]))
            a_mean = float(np.mean(a_vals))

            # b prompt accuracy = mean over variants
            b_vals = []
            for k in b_by_prompt[p]:
                rb = map_b[k]
                b_vals.append(accuracy_score(rb["y_true"], rb["y_pred"]))
            b_mean = float(np.mean(b_vals))

            acc_a.append(a_mean)
            acc_b.append(b_mean)

        acc_a = np.asarray(acc_a, float)  # length ~= 15
        acc_b = np.asarray(acc_b, float)  # length ~= 15
        diffs = acc_b - acc_a
        dz = dz_from_diffs(diffs)          # diffs = acc_b_prompt - acc_base_prompt (length 15)
        power = power_paired_t(len(diffs), dz, alpha=0.05)

        # Paired tests across prompts
        # Wilcoxon: handle all-zero diffs
        if np.allclose(diffs, 0):
            w_stat, p_w = 0.0, 1.0
        else:
            w_stat, p_w = scipy.stats.wilcoxon(diffs, alternative="two-sided")

        # Optional paired t-test (Welch not relevant; pairing already handles variance)
        t_stat, p_t = scipy.stats.ttest_rel(acc_a, acc_b)

        return {
            "dataset": dataset_name,
            "model": model_name,
            "comparison": f"{label_a}_vs_{label_b}",
            "n_prompts": int(len(prompts)),
            "acc_mean_a": float(np.mean(acc_a)),
            "acc_mean_b": float(np.mean(acc_b)),
            "mean_diff_b_minus_a": float(np.mean(diffs)),
            "p_wilcoxon": float(p_w),
            "p_paired_ttest": float(p_t),
            "dz": dz,
            "power": power,
            "significant_1p": (p_w < 0.01)
        }
        

    rows = []
    r1 = unpaired_accuracy_test(base_map, attack_map, "base", "attack")
    r2 = unpaired_accuracy_test(base_map, assist_map, "base", "assist")
    if r1: rows.append(r1)
    if r2: rows.append(r2)

    # optional CSV append
    if out_csv_path and rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        header = not os.path.exists(out_csv_path)
        df.to_csv(out_csv_path, mode="a", header=header, index=False)

    return rows

# =========================
# Main
# =========================


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Aggregated DET Curves and Compute Metrics")
    parser.add_argument("--results_dir", type=str, default="results", help="Root directory of results")
    parser.add_argument("--datasets", nargs="+", default=["lfw", "age_db_30", "calfw", "cplfw"], help="Datasets to process")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_NAME_LS, help="List of models to include")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save plots")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.plots_dir, exist_ok=True)

    global_metric_data = {
        "base": [],
        "attack": [],
        "assist": []
    }
    
    # Store per-dataset metrics for final aggregation
    # {model: {dataset: metrics_all_modes}}
    dataset_metrics_store = {m: {} for m in args.models}

    for dataset in args.datasets:
        print("\n" + "#" * 80)
        print(f"PROCESSING DATASET: {dataset}")
        print("#" * 80)

        base_results_root = os.path.join(args.results_dir, dataset)
        all_metrics_csv = ALL_METRICS_CSV_TEMPLATE.format(dataset=dataset)
        
        all_rows = []
        all_models_runs = {}

        # Check if results dir exists
        if not os.path.exists(base_results_root):
            print(f"Directory not found: {base_results_root}. Skipping dataset {dataset}.")
            continue

        # ---- Loop over all models: metrics + DET data collection ----
        for m in args.models:
            print("\n" + "=" * 40)
            print(f"Processing model: {m} (Dataset: {dataset})")
            print("=" * 40)

            runs_m, metric_data_m = load_runs_from_csv(m, base_results_root)
            all_models_runs[m] = runs_m

            # Accumulate for global analysis
            # Exclude resnet100 (unimodal, dummy prompts) from global prompt analysis & significance tests
            if m != "resnet100":
                for mode_key in ["base", "attack", "assist"]:
                    if mode_key in metric_data_m:
                        global_metric_data[mode_key].extend(metric_data_m[mode_key])

            # Compute metrics for this model
            metrics_m = compute_all_metrics(metric_data_m)

            sig_out = "significance_tests_per_model.csv"
            rows = unpaired_tests_per_model(metric_data_m, m, dataset, out_csv_path=sig_out)

            for r in rows:
                print(
                    f"[SIGTEST] {r['dataset']} | {r['model']} | {r['comparison']} "
                    f"p_wilcoxon={r['p_wilcoxon']:.3e}"
                )
            
            # Store for cross-dataset aggregation
            dataset_metrics_store[m][dataset] = metrics_m
            
            rows_m = flatten_metrics_for_csv(m, metrics_m)
            all_rows.extend(rows_m)

            # For the "single detailed" model, also make a per-model DET with bands
            # Use each model as "single detailed" in loop
            single_det_png = SINGLE_MODEL_DET_PNG.format(model=m, dataset=dataset)
            _ = plot_aggregated_det(runs_m, m, save_path=os.path.join(args.plots_dir, single_det_png), log_axes=True)

        # ---- Save all metrics to CSV ----
        if all_rows:
            df_metrics = pd.DataFrame(all_rows)
            df_metrics.to_csv(all_metrics_csv, index=False)
            print(f"\n[{dataset}] Saved all metrics to: {all_metrics_csv}")
        else:
            print(f"\n[{dataset}] No metrics rows collected, skipping CSV save.")

        # ---- Multi-model DET plots (per mode, all curves on one plot) ----
        # Pass simpler call or refactor plot_det_all_models to match new args?
        # The existing function writes to `plots/` hardcoded.
        # Let's fix that inside the function too, but for now I'll just change the call logic or let it write to plots/
        # Wait, the function plot_det_all_models writes to `plots/`.
        # I should probably update `plot_det_all_models` to use `args.plots_dir` but passing it is cleaner.
        # Since I am not refactoring that function signature in this chunk, I will assume `plots/` exists or is created.
        # I created `args.plots_dir` (default "plots").
        
        # NOTE: plot_det_all_models writes to hardcoded "plots/..."
        # I will update the function signature later or assuming "plots" is fine.
        plot_det_all_models(all_models_runs, dataset, save_dir=args.plots_dir, log_axes=True)

    # ---- Global Prompt Analysis ----
    analyze_global_prompt_performance(global_metric_data)
    
    # ---- Global Aggregation Across Datasets ----
    aggregate_across_datasets(dataset_metrics_store)

    

