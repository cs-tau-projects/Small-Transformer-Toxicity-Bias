"""
Aggregate fixed OOD results across multiple seed CSVs into mean ± std tables.

The input CSVs (ood_seed_*.csv and ood_llama_seed_*.csv) use a different
column schema than the original experiment outputs, so this script handles
that format directly.

Produces 4 tables:
  Table 1: Overall AUC per model
  Table 2: Per-subgroup bias metrics (Subgroup AUC, BPSN AUC, BNSP AUC)
  Table 3: Per-subgroup FNR / FPR comparison across models
  Table 4: Overall FNR / FPR per model — Jigsaw (ID) vs ToxiGen (OOD)

Usage:
    python src/aggregate_ood_results.py \
        --csv_dir "results/Fixed OOD Results" \
        --run_results_dir "results/Run Results" \
        --llama_results_dir "results/Fixed LLama Results" \
        --output_dir results/fixed_ood_summary
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Column mapping: new CSV name -> internal name
COL_MAP = {
    "Model": "model_name",
    "Identity": "subgroup",
    "Total Examples": "total_examples",
    "1. Overall AUC": "overall_auc",
    "2. Overall FNR": "overall_fnr",
    "3. Overall FPR": "overall_fpr",
    "4. Subgroup AUC": "subgroup_auc",
    "5. BPSN AUC": "bpsn_auc",
    "6. BNSP AUC": "bnsp_auc",
    "7. Subgroup FNR": "subgroup_fnr",
    "8. Subgroup FPR": "subgroup_fpr",
}


def _fmt(mean, std):
    """Format a mean ± std cell. Returns '—' for NaN."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_all_seeds(csv_dir):
    """Load all ood_*seed_*.csv files, normalise columns, return one
    DataFrame with an added ``seed`` column."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "ood_*seed_*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No ood_*seed_*.csv files found in {csv_dir}")

    frames = []
    for p in paths:
        # Extract seed number from filenames like ood_seed_42.csv or
        # ood_llama_seed_42.csv
        match = re.search(r"seed_(\d+)", os.path.basename(p))
        seed = match.group(1) if match else os.path.basename(p)

        df = pd.read_csv(p)
        df = df.rename(columns=COL_MAP)
        df["seed"] = seed
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    console.print(
        f"[green]Loaded {len(paths)} files "
        f"({', '.join(os.path.basename(p) for p in paths)})[/green]")
    return combined


# Mapping from Jigsaw metric filenames to model display names.
JIGSAW_METRIC_FILES = {
    "baseline_metrics.csv": "Baseline",
    "naive_baseline_metrics.csv": "Naive",
    "distilbert-base-uncased_finetuned_metrics.csv": "distilbert-base-uncased",
    "distilroberta-base_finetuned_metrics.csv": "distilroberta-base",
    "google_bert_uncased_L-4_H-512_A-8_finetuned_metrics.csv":
        "google/bert_uncased_L-4_H-512_A-8",
}

# Model name alignment between Jigsaw (ID) and OOD display names.
# Keys = Jigsaw name, Values = OOD name.
MODEL_NAME_MAP = {
    "Baseline": "Baseline (TF-IDF + LR)",
    "Naive": "Naive (Majority Vote)",
    "meta-llama/Llama-3.2-1B": "meta-llama/Llama-3.2-1B (Zero-shot)",
}


def load_jigsaw_overall(run_results_dir, llama_results_dir):
    """Load overall FNR / FPR from individual *_metrics.csv files across
    seed runs (Jigsaw in-distribution evaluation).

    Returns a DataFrame with columns:
        model_name, seed, overall_fnr, overall_fpr
    """
    rows = []

    # --- standard models from Run Results ---
    seed_dirs = sorted(glob.glob(os.path.join(run_results_dir, "*_run",
                                              "results")))
    # Also handle run_67 / run_89 style names
    seed_dirs += sorted(glob.glob(os.path.join(run_results_dir, "run_*",
                                               "results")))

    for results_dir in seed_dirs:
        parent = os.path.basename(os.path.dirname(results_dir))
        match = re.search(r"(\d+)", parent)
        if not match:
            continue
        seed = match.group(1)

        for fname, model_name in JIGSAW_METRIC_FILES.items():
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, nrows=1)  # overall values identical per row
            df = df.rename(columns=COL_MAP)
            rows.append({
                "model_name": model_name,
                "seed": seed,
                "overall_fnr": df["overall_fnr"].iloc[0],
                "overall_fpr": df["overall_fpr"].iloc[0],
            })

        # Llama raw in Run Results (only present in some seeds)
        llama_path = os.path.join(results_dir,
                                  "meta-llama_Llama-3.2-1B_raw_metrics.csv")
        if os.path.exists(llama_path):
            df = pd.read_csv(llama_path, nrows=1)
            df = df.rename(columns=COL_MAP)
            rows.append({
                "model_name": "meta-llama/Llama-3.2-1B",
                "seed": seed,
                "overall_fnr": df["overall_fnr"].iloc[0],
                "overall_fpr": df["overall_fpr"].iloc[0],
            })

    # --- Fixed Llama results (may provide additional / replacement seeds) ---
    if llama_results_dir and os.path.isdir(llama_results_dir):
        for seed_dir in sorted(glob.glob(os.path.join(llama_results_dir,
                                                       "llama_seed_*",
                                                       "results"))):
            match = re.search(r"seed_(\d+)", seed_dir)
            if not match:
                continue
            seed = match.group(1)
            path = os.path.join(seed_dir,
                                "meta-llama_Llama-3.2-1B_raw_metrics.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, nrows=1)
            df = df.rename(columns=COL_MAP)
            rows.append({
                "model_name": "meta-llama/Llama-3.2-1B",
                "seed": seed,
                "overall_fnr": df["overall_fnr"].iloc[0],
                "overall_fpr": df["overall_fpr"].iloc[0],
            })

    combined = pd.DataFrame(rows)
    # Deduplicate: if a seed appears from both Run Results and Fixed Llama,
    # keep the Fixed Llama version (last seen wins after sort).
    combined = combined.drop_duplicates(subset=["model_name", "seed"],
                                        keep="last")
    console.print(f"[green]Loaded Jigsaw (ID) overall FNR/FPR for "
                  f"{combined['model_name'].nunique()} models across "
                  f"{combined['seed'].nunique()} seeds[/green]")
    return combined


# ------------------------------------------------------------------ #
#  Table 1 – Overall AUC per model
# ------------------------------------------------------------------ #

def build_table1(df):
    """Overall AUC per model, mean ± std across seeds."""
    unique = df.drop_duplicates(subset=["model_name", "seed"])
    agg = (unique.groupby("model_name")["overall_auc"]
           .agg(["mean", "std"]).reset_index())
    agg["std"] = agg["std"].fillna(0)
    agg["Overall AUC"] = agg.apply(
        lambda r: _fmt(r["mean"], r["std"]), axis=1)
    return agg[["model_name", "Overall AUC"]].rename(
        columns={"model_name": "Model"})


# ------------------------------------------------------------------ #
#  Table 2 – Per-subgroup bias metrics
# ------------------------------------------------------------------ #

def build_table2(df):
    """Per-subgroup Subgroup AUC, BPSN AUC, BNSP AUC, mean ± std."""
    metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    agg = (df.groupby(["model_name", "subgroup"])[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Model": r["model_name"],
            "Subgroup": r["subgroup"],
            "Subgroup AUC": _fmt(r["subgroup_auc_mean"],
                                 r["subgroup_auc_std"]),
            "BPSN AUC": _fmt(r["bpsn_auc_mean"], r["bpsn_auc_std"]),
            "BNSP AUC": _fmt(r["bnsp_auc_mean"], r["bnsp_auc_std"]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Table 3 – FNR / FPR comparison
# ------------------------------------------------------------------ #

def build_table3(df):
    """Per-subgroup FNR and FPR, mean ± std across seeds."""
    metrics = ["subgroup_fnr", "subgroup_fpr"]
    agg = (df.groupby(["model_name", "subgroup"])[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Model": r["model_name"],
            "Subgroup": r["subgroup"],
            "FNR": _fmt(r["subgroup_fnr_mean"], r["subgroup_fnr_std"]),
            "FPR": _fmt(r["subgroup_fpr_mean"], r["subgroup_fpr_std"]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Table 4 – Overall FNR / FPR: Jigsaw (ID) vs ToxiGen (OOD)
# ------------------------------------------------------------------ #

def build_table4(ood_df, jigsaw_df):
    """Overall FNR and FPR per model for both evaluation settings."""

    # --- OOD side ---
    ood_unique = ood_df.drop_duplicates(subset=["model_name", "seed"])
    ood_agg = (ood_unique.groupby("model_name")[["overall_fnr", "overall_fpr"]]
               .agg(["mean", "std"]).reset_index())
    ood_agg.columns = ["_".join(c).rstrip("_") for c in ood_agg.columns]

    # --- Jigsaw (ID) side ---
    jig_agg = (jigsaw_df.groupby("model_name")[["overall_fnr", "overall_fpr"]]
               .agg(["mean", "std"]).reset_index())
    jig_agg.columns = ["_".join(c).rstrip("_") for c in jig_agg.columns]

    # Build a unified model list.  Use OOD names as canonical and map Jigsaw
    # names where they differ.
    jig_agg["canonical"] = jig_agg["model_name"].map(
        lambda n: MODEL_NAME_MAP.get(n, n))

    rows = []
    all_models = sorted(
        set(ood_agg["model_name"].tolist())
        | set(jig_agg["canonical"].tolist()))

    for model in all_models:
        row = {"Model": model}

        # Jigsaw
        jig_row = jig_agg[jig_agg["canonical"] == model]
        if not jig_row.empty:
            r = jig_row.iloc[0]
            row["Jigsaw FNR"] = _fmt(r["overall_fnr_mean"],
                                     r["overall_fnr_std"])
            row["Jigsaw FPR"] = _fmt(r["overall_fpr_mean"],
                                     r["overall_fpr_std"])
        else:
            row["Jigsaw FNR"] = "—"
            row["Jigsaw FPR"] = "—"

        # OOD
        ood_row = ood_agg[ood_agg["model_name"] == model]
        if not ood_row.empty:
            r = ood_row.iloc[0]
            row["ToxiGen FNR"] = _fmt(r["overall_fnr_mean"],
                                      r["overall_fnr_std"])
            row["ToxiGen FPR"] = _fmt(r["overall_fpr_mean"],
                                      r["overall_fpr_std"])
        else:
            row["ToxiGen FNR"] = "—"
            row["ToxiGen FPR"] = "—"

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Display & Save
# ------------------------------------------------------------------ #

def _rich_print(table_df, title):
    t = Table(title=title, show_header=True, header_style="bold magenta",
              show_lines=True)
    for col in table_df.columns:
        t.add_column(col)
    for _, row in table_df.iterrows():
        t.add_row(*[str(v) for v in row])
    console.print(t)


def _save(table_df, path, title):
    table_df.to_csv(path, index=False)
    console.print(f"[green]Saved {title} → {path}[/green]")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate fixed OOD results into summary tables.")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing ood_*seed_*.csv files")
    parser.add_argument("--run_results_dir", type=str, required=True,
                        help="Directory containing per-seed run folders "
                             "(e.g. results/Run Results)")
    parser.add_argument("--llama_results_dir", type=str, default=None,
                        help="Directory containing fixed Llama seed folders "
                             "(e.g. results/Fixed LLama Results)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ood_df = load_all_seeds(args.csv_dir)
    jigsaw_df = load_jigsaw_overall(args.run_results_dir,
                                    args.llama_results_dir)

    t1 = build_table1(ood_df)
    _rich_print(t1, "Table 1: Overall AUC — OOD (mean ± std across seeds)")
    _save(t1, os.path.join(args.output_dir, "table1_overall_auc_ood.csv"),
          "Table 1")

    t2 = build_table2(ood_df)
    _rich_print(t2, "Table 2: Per-Subgroup Bias Metrics — OOD")
    _save(t2, os.path.join(args.output_dir, "table2_subgroup_ood.csv"),
          "Table 2")

    t3 = build_table3(ood_df)
    _rich_print(t3, "Table 3: FNR / FPR Comparison — OOD")
    _save(t3, os.path.join(args.output_dir, "table3_fnr_fpr_ood.csv"),
          "Table 3")

    t4 = build_table4(ood_df, jigsaw_df)
    _rich_print(t4, "Table 4: Overall FNR / FPR — Jigsaw (ID) vs ToxiGen (OOD)")
    _save(t4, os.path.join(args.output_dir,
                           "table4_overall_fnr_fpr.csv"), "Table 4")

    console.print("\n[bold cyan]All 4 summary tables generated.[/bold cyan]")


if __name__ == "__main__":
    main()
