"""
Aggregate LLaMA results across multiple seeds into mean ± std summary tables.

Reads from results/llama_results/seed_*/ and produces 4 tables mirroring
the structure of aggregate_results.py:
  Table 1: Overall AUC — in-distribution (Jigsaw) and OOD (ToxiGen)
  Table 2: Per-subgroup bias metrics (in-distribution, Jigsaw)
  Table 3: Per-subgroup bias metrics (OOD, ToxiGen)
  Table 4: FNR / FPR comparison (ID + OOD)

Usage:
    python src/aggregate_llama_results.py \
        --llama_dir results/llama_results \
        --output_dir results/llama_summary_tables
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_NAME = "meta-llama/Llama-3.2-1B (Zero-shot)"

# Column mapping: LLaMA CSV headers → canonical names
_ID_COL_MAP = {
    "Identity": "subgroup",
    "1. Overall AUC": "overall_auc",
    "2. Overall FNR": "overall_fnr",
    "3. Overall FPR": "overall_fpr",
    "4. Subgroup AUC": "subgroup_auc",
    "5. BPSN AUC": "bpsn_auc",
    "6. BNSP AUC": "bnsp_auc",
    "7. Subgroup FNR": "subgroup_fnr",
    "8. Subgroup FPR": "subgroup_fpr",
}

_OOD_COL_MAP = {
    "Identity": "subgroup",
    "1. Overall AUC": "overall_auc",
    "2. Overall FNR": "overall_fnr",
    "3. Overall FPR": "overall_fpr",
    "4. Subgroup AUC": "subgroup_auc",
    "5. BPSN AUC": "bpsn_auc",
    "6. BNSP AUC": "bnsp_auc",
    "7. Subgroup FNR": "subgroup_fnr",
    "8. Subgroup FPR": "subgroup_fpr",
}

EXCLUDED_SUBGROUPS_ID = {"other_disability", "other_gender"}


def _fmt(mean, std):
    """Format a mean ± std cell.  Returns '—' for NaN."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_all_seeds(llama_dir):
    """Load ID and OOD metrics for every seed_* directory and return a
    single DataFrame with columns matching the canonical schema."""
    seed_dirs = sorted(glob.glob(os.path.join(llama_dir, "seed_*")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found in {llama_dir}")

    frames = []
    for sd in seed_dirs:
        seed = os.path.basename(sd).replace("seed_", "")

        # --- In-distribution (Jigsaw) ---
        id_path = os.path.join(sd, "meta-llama_Llama-3.2-1B_raw_metrics.csv")
        if os.path.exists(id_path):
            df_id = pd.read_csv(id_path).rename(columns=_ID_COL_MAP)
            df_id["evaluation_type"] = "id"
            df_id["seed"] = seed
            df_id["model_name"] = MODEL_NAME
            frames.append(df_id)

        # --- OOD (ToxiGen) ---
        ood_path = os.path.join(sd, "ood_toxigen_metrics.csv")
        if os.path.exists(ood_path):
            df_ood = pd.read_csv(ood_path).rename(columns=_OOD_COL_MAP)
            df_ood["evaluation_type"] = "ood"
            df_ood["seed"] = seed
            df_ood["model_name"] = MODEL_NAME
            frames.append(df_ood)

    combined = pd.concat(frames, ignore_index=True)

    # Ensure numeric types for metric columns
    metric_cols = ["overall_auc", "overall_fnr", "overall_fpr",
                   "subgroup_auc", "bpsn_auc", "bnsp_auc",
                   "subgroup_fnr", "subgroup_fpr"]
    for col in metric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    console.print(f"[green]Loaded {len(seed_dirs)} seed directories "
                  f"({', '.join(os.path.basename(d) for d in seed_dirs)})[/green]")
    return combined


# --------------------------------------------------------------------------- #
#  Table 1 – Overall AUC + FNR + FPR
# --------------------------------------------------------------------------- #

def build_table1(df):
    """Overall AUC, FNR, FPR per evaluation type, mean ± std across seeds."""
    unique = df.drop_duplicates(subset=["evaluation_type", "seed"])

    metrics = ["overall_auc", "overall_fnr", "overall_fpr"]
    agg = (unique.groupby("evaluation_type")[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Evaluation": r["evaluation_type"],
            "Overall AUC": _fmt(r["overall_auc_mean"], r["overall_auc_std"]),
            "Overall FNR": _fmt(r["overall_fnr_mean"], r["overall_fnr_std"]),
            "Overall FPR": _fmt(r["overall_fpr_mean"], r["overall_fpr_std"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Table 2 – Per-subgroup bias metrics (ID)
# --------------------------------------------------------------------------- #

def build_table2(df):
    """Per-subgroup bias metrics for in-distribution (Jigsaw)."""
    sub = df[(df["evaluation_type"] == "id") &
             (~df["subgroup"].isin(EXCLUDED_SUBGROUPS_ID))].copy()

    metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    agg = (sub.groupby("subgroup")[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Subgroup": r["subgroup"],
            "Subgroup AUC": _fmt(r["subgroup_auc_mean"], r["subgroup_auc_std"]),
            "BPSN AUC": _fmt(r["bpsn_auc_mean"], r["bpsn_auc_std"]),
            "BNSP AUC": _fmt(r["bnsp_auc_mean"], r["bnsp_auc_std"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Table 3 – Per-subgroup bias metrics (OOD)
# --------------------------------------------------------------------------- #

def build_table3(df):
    """Per-subgroup bias metrics for OOD (ToxiGen)."""
    sub = df[df["evaluation_type"] == "ood"].copy()

    metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    agg = (sub.groupby("subgroup")[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Subgroup": r["subgroup"],
            "Subgroup AUC": _fmt(r["subgroup_auc_mean"], r["subgroup_auc_std"]),
            "BPSN AUC": _fmt(r["bpsn_auc_mean"], r["bpsn_auc_std"]),
            "BNSP AUC": _fmt(r["bnsp_auc_mean"], r["bnsp_auc_std"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Table 4 – FNR / FPR comparison
# --------------------------------------------------------------------------- #

def build_table4(df):
    """Per-subgroup FNR and FPR, aggregated across seeds, for both ID & OOD."""
    metrics = ["subgroup_fnr", "subgroup_fpr"]
    agg = (df.groupby(["evaluation_type", "subgroup"])[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Eval Type": r["evaluation_type"],
            "Subgroup": r["subgroup"],
            "FNR": _fmt(r["subgroup_fnr_mean"], r["subgroup_fnr_std"]),
            "FPR": _fmt(r["subgroup_fpr_mean"], r["subgroup_fpr_std"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Display & Save helpers
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LLaMA multi-seed results into summary tables.")
    parser.add_argument("--llama_dir", type=str, required=True,
                        help="Directory containing seed_*/ subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_all_seeds(args.llama_dir)

    # Table 1
    t1 = build_table1(df)
    _rich_print(t1, f"Table 1: {MODEL_NAME} — Overall Metrics (mean ± std)")
    _save(t1, os.path.join(args.output_dir, "llama_table1_overall.csv"),
          "Table 1")

    # Table 2
    t2 = build_table2(df)
    _rich_print(t2, f"Table 2: {MODEL_NAME} — Per-Subgroup Bias (ID / Jigsaw)")
    _save(t2, os.path.join(args.output_dir, "llama_table2_subgroup_id.csv"),
          "Table 2")

    # Table 3
    t3 = build_table3(df)
    _rich_print(t3, f"Table 3: {MODEL_NAME} — Per-Subgroup Bias (OOD / ToxiGen)")
    _save(t3, os.path.join(args.output_dir, "llama_table3_subgroup_ood.csv"),
          "Table 3")

    # Table 4
    t4 = build_table4(df)
    _rich_print(t4, f"Table 4: {MODEL_NAME} — FNR / FPR Comparison")
    _save(t4, os.path.join(args.output_dir, "llama_table4_fnr_fpr.csv"),
          "Table 4")

    console.print(f"\n[bold cyan]All 4 LLaMA summary tables generated.[/bold cyan]")


if __name__ == "__main__":
    main()
