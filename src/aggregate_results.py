"""
Aggregate results across multiple seed CSVs into mean ± std summary tables.

Produces 4 tables (as described in Action_Items.md §3):
  Table 1: Overall AUC — all models, in-distribution and OOD
  Table 2: Per-subgroup bias metrics (in-distribution)
  Table 3: Per-subgroup bias metrics (OOD)
  Table 4: FNR / FPR comparison across model types

Usage:
    python src/aggregate_results.py --csv_dir results/results_csv --output_dir results/summary_tables
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

METRIC_COLS = ["overall_auc", "subgroup_auc", "bpsn_auc", "bnsp_auc",
               "subgroup_fnr", "subgroup_fpr"]

# Subgroups to exclude from main tables (too few examples for reliable metrics).
EXCLUDED_SUBGROUPS_ID = {"other_disability", "other_gender"}

# Subgroups to flag with a caveat footnote.
CAVEAT_SUBGROUPS_ID = {
    "other_sexual_orientation", "physical_disability",
    "other_religion", "intellectual_or_learning_disability", "buddhist",
}


def _fmt(mean, std):
    """Format a mean ± std cell. Returns '—' for NaN."""
    if np.isnan(mean):
        return "—"
    if np.isnan(std) or std == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def load_all_seeds(csv_dir):
    """Load all results_*.csv files from *csv_dir* and return a single
    DataFrame with an added ``seed`` column."""
    paths = sorted(glob.glob(os.path.join(csv_dir, "results_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No results_*.csv files found in {csv_dir}")

    frames = []
    for p in paths:
        seed = os.path.basename(p).replace("results_", "").replace(".csv", "")
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    console.print(f"[green]Loaded {len(paths)} seed files "
                  f"({', '.join(os.path.basename(p) for p in paths)})[/green]")
    return combined


# --------------------------------------------------------------------------- #
#  Table 1 – Overall AUC
# --------------------------------------------------------------------------- #

def build_table1(df):
    """Overall AUC per model per evaluation type, mean ± std across seeds."""
    # Each (model, eval_type, seed) has a single overall_auc (same for every
    # subgroup row), so deduplicate first.
    unique = df.drop_duplicates(subset=["model_name", "evaluation_type", "seed"])

    agg = (unique.groupby(["model_name", "evaluation_type"])["overall_auc"]
           .agg(["mean", "std"]).reset_index())
    agg["std"] = agg["std"].fillna(0)
    agg["auc_str"] = agg.apply(lambda r: _fmt(r["mean"], r["std"]), axis=1)

    # Pivot so evaluation types become columns
    pivot = agg.pivot(index="model_name", columns="evaluation_type",
                      values="auc_str").reset_index()
    pivot = pivot.rename(columns={"model_name": "Model"})

    # Friendly column order
    col_order = ["Model"]
    for c in ["baseline", "finetuned", "raw", "ood"]:
        if c in pivot.columns:
            pivot = pivot.rename(columns={c: f"AUC ({c})"})
            col_order.append(f"AUC ({c})")
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    return pivot


# --------------------------------------------------------------------------- #
#  Table 2 & 3 – Per-subgroup bias metrics
# --------------------------------------------------------------------------- #

def _build_subgroup_table(df, eval_types, excluded, caveat):
    """Per-subgroup bias metrics (subgroup_auc, bpsn_auc, bnsp_auc),
    aggregated as mean ± std across seeds."""
    sub = df[df["evaluation_type"].isin(eval_types)].copy()
    sub = sub[~sub["subgroup"].isin(excluded)]

    metrics = ["subgroup_auc", "bpsn_auc", "bnsp_auc"]
    agg = (sub.groupby(["model_name", "subgroup"])[metrics]
           .agg(["mean", "std"]).reset_index())

    # Flatten multi-level columns
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        subgroup = r["subgroup"]
        flag = " *" if subgroup in caveat else ""
        rows.append({
            "Model": r["model_name"],
            "Subgroup": subgroup + flag,
            "Subgroup AUC": _fmt(r["subgroup_auc_mean"], r["subgroup_auc_std"]),
            "BPSN AUC": _fmt(r["bpsn_auc_mean"], r["bpsn_auc_std"]),
            "BNSP AUC": _fmt(r["bnsp_auc_mean"], r["bnsp_auc_std"]),
        })
    return pd.DataFrame(rows)


def build_table2(df):
    return _build_subgroup_table(
        df, ["baseline", "finetuned"],
        EXCLUDED_SUBGROUPS_ID, CAVEAT_SUBGROUPS_ID)


def build_table3(df):
    return _build_subgroup_table(
        df, ["ood"],
        set(), set())  # OOD has different subgroup names; include all


# --------------------------------------------------------------------------- #
#  Table 4 – FNR / FPR comparison
# --------------------------------------------------------------------------- #

def build_table4(df):
    """Per-subgroup FNR and FPR, aggregated across seeds, for comparing
    failure mode differences across model types."""
    metrics = ["subgroup_fnr", "subgroup_fpr"]
    agg = (df.groupby(["model_name", "evaluation_type", "subgroup"])[metrics]
           .agg(["mean", "std"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    rows = []
    for _, r in agg.iterrows():
        rows.append({
            "Model": r["model_name"],
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
        description="Aggregate multi-seed results into summary tables.")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing results_*.csv files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for output summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_all_seeds(args.csv_dir)

    # Table 1
    t1 = build_table1(df)
    _rich_print(t1, "Table 1: Overall AUC (mean ± std across seeds)")
    _save(t1, os.path.join(args.output_dir, "table1_overall_auc.csv"),
          "Table 1")

    # Table 2
    t2 = build_table2(df)
    _rich_print(t2, "Table 2: Per-Subgroup Bias Metrics — In-Distribution "
                    "(* = caveat: small sample)")
    _save(t2, os.path.join(args.output_dir, "table2_subgroup_id.csv"),
          "Table 2")

    # Table 3
    t3 = build_table3(df)
    _rich_print(t3, "Table 3: Per-Subgroup Bias Metrics — OOD (ToxiGen)")
    _save(t3, os.path.join(args.output_dir, "table3_subgroup_ood.csv"),
          "Table 3")

    # Table 4
    t4 = build_table4(df)
    _rich_print(t4, "Table 4: FNR / FPR Comparison")
    _save(t4, os.path.join(args.output_dir, "table4_fnr_fpr.csv"),
          "Table 4")

    console.print("\n[bold cyan]All 4 summary tables generated.[/bold cyan]")


if __name__ == "__main__":
    main()
