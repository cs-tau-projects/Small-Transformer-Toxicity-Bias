import logging
import os
import json
import ast
import pandas as pd
import numpy as np
from datasets import load_from_disk
from rich.console import Console
from rich.table import Table
from src.steps.eval_ood_step import TOXIGEN_GROUP_NORMALIZATION

logger = logging.getLogger("pipeline")
console = Console()


def _display_stats_table(stats_df, title):
    """Displays a stats DataFrame as a rich table and logs it."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for col in stats_df.columns:
        table.add_column(col)
    for _, row in stats_df.iterrows():
        table.add_row(*[str(val) for val in row])
    console.print(table)


def _compute_jigsaw_subgroup_stats(test_ds, identity_columns):
    """Computes per-subgroup sample counts for the Jigsaw test set."""
    stats = []

    total_samples = len(test_ds)
    total_toxic = sum(test_ds["is_toxic"])
    stats.append({
        "Identity": "Overall",
        "Total": total_samples,
        "Toxic": total_toxic,
        "Non-Toxic": total_samples - total_toxic,
        "Toxicity Rate": f"{(total_toxic / total_samples):.2%}" if total_samples > 0 else "0%"
    })

    for col in identity_columns:
        subgroup_mask = [val >= 0.5 for val in test_ds[col]]
        is_toxic = test_ds["is_toxic"]

        subgroup_indices = [i for i, x in enumerate(subgroup_mask) if x]
        subgroup_labels = [is_toxic[i] for i in subgroup_indices]
        n_subgroup = len(subgroup_labels)
        n_toxic = sum(subgroup_labels)

        stats.append({
            "Identity": col,
            "Total": n_subgroup,
            "Toxic": n_toxic,
            "Non-Toxic": n_subgroup - n_toxic,
            "Toxicity Rate": f"{(n_toxic / n_subgroup):.2%}" if n_subgroup > 0 else "N/A"
        })

    return pd.DataFrame(stats)


def _compute_toxigen_subgroup_stats(toxigen_df):
    """Computes per-subgroup sample counts for the ToxiGen OOD dataset."""
    # Find the group column
    possible_group_cols = ["target_groups", "target_group", "group"]
    found_group_col = next((c for c in possible_group_cols if c in toxigen_df.columns), None)

    if not found_group_col:
        logger.warning("Could not find identity group column in ToxiGen data.")
        return None

    # Parse identity groups per row (same logic as extract_toxigen_identities_and_evaluate)
    group_membership = {}
    for i, row in toxigen_df.iterrows():
        group_val = row[found_group_col]
        if isinstance(group_val, str):
            if group_val.startswith("[") and group_val.endswith("]"):
                try:
                    groups = ast.literal_eval(group_val)
                except (ValueError, SyntaxError):
                    groups = [group_val]
            else:
                groups = [g.strip() for g in group_val.split(",")]
        elif isinstance(group_val, list):
            groups = group_val
        else:
            groups = [str(group_val)]

        for g in groups:
            g = TOXIGEN_GROUP_NORMALIZATION.get(g, g)
            if g and g.lower() not in ["none", "nan", "null", "unknown"]:
                if g not in group_membership:
                    group_membership[g] = []
                group_membership[g].append(i)

    stats = []
    total = len(toxigen_df)
    total_toxic = int(toxigen_df["label"].sum())
    stats.append({
        "Identity": "Overall",
        "Total": total,
        "Toxic": total_toxic,
        "Non-Toxic": total - total_toxic,
        "Toxicity Rate": f"{(total_toxic / total):.2%}" if total > 0 else "0%"
    })

    for group_name in sorted(group_membership.keys()):
        indices = group_membership[group_name]
        subgroup_labels = toxigen_df.loc[indices, "label"]
        n_subgroup = len(indices)
        n_toxic = int(subgroup_labels.sum())

        stats.append({
            "Identity": group_name,
            "Total": n_subgroup,
            "Toxic": n_toxic,
            "Non-Toxic": n_subgroup - n_toxic,
            "Toxicity Rate": f"{(n_toxic / n_subgroup):.2%}" if n_subgroup > 0 else "N/A"
        })

    return pd.DataFrame(stats)


def run_analysis_step(data_dir, results_dir, model_name_for_errors=None):
    """
    Performs dataset statistics (Jigsaw + ToxiGen) and error analysis.
    """
    logger.info("Running Dataset Analysis & Error Sampling...")
    console.print("\n[bold cyan]--- Running Dataset Analysis & Error Sampling ---[/bold cyan]")

    # 1. Load Data
    test_path = os.path.join(data_dir, "test")
    if not os.path.exists(test_path):
        logger.error(f"Test dataset not found at {test_path}")
        console.print(f"[red]Error: Test dataset not found at {test_path}[/red]")
        return

    test_ds = load_from_disk(test_path)
    with open(os.path.join(data_dir, "identity_columns.json"), "r") as f:
        identity_columns = json.load(f)

    # 2. Jigsaw Test Set Statistics
    console.print("Calculating Jigsaw subgroup distributions...")
    jigsaw_stats_df = _compute_jigsaw_subgroup_stats(test_ds, identity_columns)
    jigsaw_stats_path = os.path.join(results_dir, "dataset_stats.csv")
    jigsaw_stats_df.to_csv(jigsaw_stats_path, index=False)
    _display_stats_table(jigsaw_stats_df, "Jigsaw Test Set — Subgroup Sample Sizes")
    logger.info(f"Saved Jigsaw statistics to {jigsaw_stats_path}")
    console.print(f"[green]Saved Jigsaw statistics to {jigsaw_stats_path}[/green]")

    # 3. ToxiGen OOD Statistics
    toxigen_path = os.path.join(data_dir, "toxigen_standardized.parquet")
    if os.path.exists(toxigen_path):
        console.print("Calculating ToxiGen (OOD) subgroup distributions...")
        toxigen_df = pd.read_parquet(toxigen_path)
        toxigen_stats_df = _compute_toxigen_subgroup_stats(toxigen_df)

        if toxigen_stats_df is not None:
            toxigen_stats_path = os.path.join(results_dir, "dataset_stats_toxigen.csv")
            toxigen_stats_df.to_csv(toxigen_stats_path, index=False)
            _display_stats_table(toxigen_stats_df, "ToxiGen (OOD) — Subgroup Sample Sizes")
            logger.info(f"Saved ToxiGen statistics to {toxigen_stats_path}")
            console.print(f"[green]Saved ToxiGen statistics to {toxigen_stats_path}[/green]")
    else:
        console.print("[yellow]ToxiGen parquet not found — run eval-ood step first to generate OOD statistics.[/yellow]")
        logger.warning(f"ToxiGen parquet not found at {toxigen_path}. Skipping OOD statistics.")
    
    # 4. Error Analysis
    # We look for a prediction file to sample errors from
    pred_files = [f for f in os.listdir(results_dir) if f.startswith("preds_") and "finetuned" in f and not "ood" in f]
    
    target_pred_file = None
    if model_name_for_errors:
        safe_name = model_name_for_errors.replace("/", "_")
        target_pred_file = f"preds_{safe_name}_finetuned.csv"
        
    if not target_pred_file or not os.path.exists(os.path.join(results_dir, target_pred_file)):
        if pred_files:
            target_pred_file = pred_files[0]
        else:
            console.print("[yellow]Warning: No fine-tuned prediction files found for error analysis.[/yellow]")
            return
            
    console.print(f"Sampling errors from [bold]{target_pred_file}[/bold]...")
    preds_df = pd.read_csv(os.path.join(results_dir, target_pred_file))
    
    # Ensure indices align (assuming preds_df was saved in same order as test_ds)
    if len(preds_df) != len(test_ds):
        console.print("[red]Error: Prediction file length does not match test dataset length. Skipping error sampling.[/red]")
        return
        
    analysis_df = pd.DataFrame({
        "comment_text": test_ds["comment_text"],
        "true_label": test_ds["is_toxic"],
        "pred_score": preds_df["toxicity_score"]
    })
    analysis_df["pred_label"] = (analysis_df["pred_score"] >= 0.5).astype(int)
    
    # False Positives (Non-toxic predicted as toxic)
    fps = analysis_df[(analysis_df["true_label"] == 0) & (analysis_df["pred_label"] == 1)].copy()
    fps = fps.sort_values(by="pred_score", ascending=False).head(10)
    fps["error_type"] = "False Positive"
    
    # False Negatives (Toxic predicted as non-toxic)
    fns = analysis_df[(analysis_df["true_label"] == 1) & (analysis_df["pred_label"] == 0)].copy()
    fns = fns.sort_values(by="pred_score", ascending=True).head(10)
    fns["error_type"] = "False Negative"
    
    errors_df = pd.concat([fps, fns])
    errors_path = os.path.join(results_dir, "error_analysis.csv")
    errors_df.to_csv(errors_path, index=False)
    
    console.print("\n[bold]Sampled False Positives (Non-toxic labeled as Toxic):[/bold]")
    for i, row in fps.head(5).iterrows():
        console.print(f"  - [red]Pred {row['pred_score']:.4f}:[/red] {row['comment_text'][:200]}...")
        
    console.print("\n[bold]Sampled False Negatives (Toxic labeled as Non-toxic):[/bold]")
    for i, row in fns.head(5).iterrows():
        console.print(f"  - [blue]Pred {row['pred_score']:.4f}:[/blue] {row['comment_text'][:200]}...")
        
    logger.info(f"Saved error analysis to {errors_path}")
    console.print(f"\n[green]Saved error analysis to {errors_path}[/green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_name_for_errors", type=str, default=None)
    args = parser.parse_args()
    
    run_analysis_step(args.data_dir, args.results_dir, args.model_name_for_errors)
