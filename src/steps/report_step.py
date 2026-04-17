import os
import pandas as pd
from src.steps.utils import load_saved_data, setup_logging
from datetime import datetime
from rich.console import Console
from rich.table import Table

console = Console()

# Define the headers for the new structured log file
CSV_HEADERS = [
    "timestamp",
    "experiment_name",
    "model_name",
    "evaluation_type",
    "overall_auc",
    "subgroup",
    "subgroup_auc",
    "subgroup_fnr",
    "subgroup_fpr",
]

def log_results_to_csv(results_dir, all_results_dict, experiment_name="default_experiment"):
    """
    Appends results from a dictionary of dataframes to a master CSV log.
    Transforms wide-format metrics into a long-format for easier analysis.
    """
    log_path = os.path.join(results_dir, "results.csv")
    
    # Prepare a list to hold all the new rows to be appended
    new_rows = []
    timestamp = datetime.now().isoformat()
    
    for model_display_name, df in all_results_dict.items():
        # Deconstruct the display name to get model and type
        if " Finetuned" in model_display_name:
            model_name = model_display_name.replace(" Finetuned", "")
            eval_type = "finetuned"
        elif " Raw" in model_display_name:
            model_name = model_display_name.replace(" Raw", "")
            eval_type = "raw"
        elif " OOD" in model_display_name:
            model_name = model_display_name.replace(" OOD", "")
            eval_type = "ood"
        else:
            model_name = model_display_name
            eval_type = "baseline"
            
        # The overall AUC is the same for all rows in a given dataframe
        if "1. Overall AUC" in df.columns:
            overall_auc = df["1. Overall AUC"].iloc[0]
        else:
            overall_auc = pd.NA

        # Transform each row (subgroup) of the input dataframe into a dictionary
        for _, row in df.iterrows():
            new_row = {
                "timestamp": timestamp,
                "experiment_name": experiment_name,
                "model_name": model_name,
                "evaluation_type": eval_type,
                "overall_auc": overall_auc,
                "subgroup": row["Identity"] if "Identity" in row else row.get("subgroup", "Overall"),
                "subgroup_auc": row.get("4. Subgroup AUC"),
                "subgroup_fnr": row.get("5. Subgroup FNR"),
                "subgroup_fpr": row.get("6. Subgroup FPR"),
            }
            new_rows.append(new_row)

    if not new_rows:
        return

    # Create DataFrame from the new rows
    new_df = pd.DataFrame(new_rows, columns=CSV_HEADERS)

    # Append to the master CSV file
    if not os.path.exists(log_path):
        # File doesn't exist, write with header
        new_df.to_csv(log_path, index=False, header=True)
        console.print(f"[green]Created new structured log at {log_path}[/green]")
    else:
        # File exists, append without header
        new_df.to_csv(log_path, mode='a', index=False, header=False)
        console.print(f"[green]Appended {len(new_rows)} rows to structured log at {log_path}[/green]")

def display_rich_table(df, title):
    """Utility to display a pandas DataFrame as a Rich Table."""
    table = Table(title=title, show_header=True, header_style="bold magenta", box=None)

    for column in df.columns:
        table.add_column(str(column))

    for _, row in df.iterrows():
        # Convert all values to string for display, format floats to 4 decimal places
        formatted_row = []
        for val in row:
            if isinstance(val, float):
                formatted_row.append(f"{val:.4f}")
            else:
                formatted_row.append(str(val))
        table.add_row(*formatted_row)

    console.print(table)
    console.print("\n")

def format_final_report(all_results_dict):
    """Combines metrics from all models into a comparative table suitable for an ACL report."""
    if not all_results_dict:
        console.print("[yellow]No results to report. Are there CSV files in the results directory?[/yellow]")
        return

    console.print("\n" + "="*80)
    console.print("[bold cyan]FINAL COMPARISON REPORT (ACL Format)[/bold cyan]")
    console.print("="*80 + "\n")
    
    def extract_summary(df, model_name):
        df_copy = df.copy()
        # Some legacy columns might not be there cleanly, so we check carefully
        cols = ['Identity']
        if 'Total Examples' in df.columns:
            cols.append('Total Examples')
        metric_cols = [col for col in df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.'))]
        
        # We rename the metric columns to have the model name prefixed
        rename_dict = {col: f"{model_name} {col.split('. ')[1]}" for col in metric_cols}
        df_copy = df_copy[cols + metric_cols].rename(columns=rename_dict)
        return df_copy
    
    # Try to use Baseline as the base dataframe if it exists, otherwise pick the first one
    base_key = "Baseline" if "Baseline" in all_results_dict else list(all_results_dict.keys())[0]
    
    if 'Identity' not in all_results_dict[base_key].columns:
        # Fallback if Identity column is missing (e.g. from some OOD formats)
        all_results_dict[base_key]['Identity'] = 'Overall'

    if 'Total Examples' in all_results_dict[base_key].columns:
        final_df = all_results_dict[base_key][['Identity', 'Total Examples']].copy()
    else:
        final_df = all_results_dict[base_key][['Identity']].copy()

    for model_name, df in all_results_dict.items():
        if 'Identity' not in df.columns:
            df['Identity'] = 'Overall'
        sum_df = extract_summary(df, model_name)
        if 'Total Examples' in sum_df.columns and model_name != base_key:
            sum_df = sum_df.drop(columns=['Total Examples'])
        final_df = final_df.merge(sum_df, on='Identity', how='left')
    
    # 1. Overall AUC Comparison
    auc_cols = ['Identity'] + [c for c in final_df.columns if 'Overall AUC' in c]
    if auc_cols[1:]:
        overall_auc = final_df[auc_cols].head(1).copy()
        overall_auc.loc[0, 'Identity'] = 'Overall Dataset'
        display_rich_table(overall_auc, "1. Overall AUC Comparison")
    
    # 2. Subgroup AUC Comparison
    subgroup_cols = ['Identity'] + [c for c in final_df.columns if 'Subgroup AUC' in c]
    if subgroup_cols[1:]:
        display_rich_table(final_df[subgroup_cols], "2. Subgroup AUC Comparison")
    
    # 3. FNR Comparison
    fnr_cols = ['Identity'] + [c for c in final_df.columns if 'FNR' in c]
    if fnr_cols[1:]:
        display_rich_table(final_df[fnr_cols], "3. FNR Comparison (Subgroup and Overall)")

    # 4. FPR Comparison
    fpr_cols = ['Identity'] + [c for c in final_df.columns if 'FPR' in c]
    if fpr_cols[1:]:
        display_rich_table(final_df[fpr_cols], "4. FPR Comparison (Subgroup and Overall)")

    return final_df

def run_report_step(data_dir, results_dir, cache_dir, llama_model, models, eval_samples, seed=42):
    console.print(f"\n[bold]Generating Report from {results_dir}...[/bold]")
    all_results_dict = {}
    
    # Map filenames back to nice display names
    reverse_map = {m.replace("/", "_"): m for m in models + [llama_model]}
    
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if not fname.endswith(".csv") or fname == "final_report.csv" or fname == "results.csv":
                continue
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            
            # Reverse-engineer the display name from the filename
            if fname == "baseline_metrics.csv":
                all_results_dict["Baseline"] = df
            elif fname == "naive_baseline_metrics.csv":
                all_results_dict["Naive"] = df
            elif fname.endswith("_raw_metrics.csv"):
                safe_name = fname.replace("_raw_metrics.csv", "")
                real_name = reverse_map.get(safe_name, safe_name)
                all_results_dict[f"{real_name} Raw"] = df
            elif fname.endswith("_finetuned_metrics.csv"):
                safe_name = fname.replace("_finetuned_metrics.csv", "")
                real_name = reverse_map.get(safe_name, safe_name)
                all_results_dict[f"{real_name} Finetuned"] = df
            elif fname == "ood_toxigen_metrics.csv":
                # OOD CSV contains multiple models, split them out
                for model_display_name in df['Model'].unique():
                    model_df = df[df['Model'] == model_display_name].copy()
                    # Keep only the standard metric columns
                    all_results_dict[f"{model_display_name} OOD"] = model_df

    final_df = format_final_report(all_results_dict)
    if final_df is not None:
        out_path = os.path.join(results_dir, "final_report.csv")
        final_df.to_csv(out_path, index=False)
        console.print(f"[green]Saved final report to {out_path}[/green]")
        
        # Log to the master CSV
        log_results_to_csv(results_dir, all_results_dict)
        
    console.print("\nCompiling single aggregated [bold]final_predictions.csv[/bold]...")
    try:
        from src.steps.eval_ood_step import load_toxigen_dataset
        _, test_ds, _ = load_saved_data(data_dir)
        jigsaw_df = pd.DataFrame({
            'sentence': test_ds['comment_text'],
            'dataset': 'Jigsaw',
            'true_label': test_ds['is_toxic']
        })
        
        try:
            toxigen = load_toxigen_dataset(cache_dir, eval_samples, seed=seed)
            if 'text' in toxigen.columns:
                t_text = toxigen['text']
            elif 'generation' in toxigen.columns:
                t_text = toxigen['generation']
            elif 'comment_text' in toxigen.columns:
                t_text = toxigen['comment_text']
            if 'label' in toxigen.columns:
                t_label = toxigen['label']
            else:
                t_label = [pd.NA] * len(toxigen)
            toxigen_df = pd.DataFrame({
                'sentence': t_text,
                'dataset': 'ToxiGen',
                'true_label': t_label
            })
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load toxigen dataset for final predictions compilation: {e}[/yellow]")
            toxigen_df = pd.DataFrame(columns=['sentence', 'dataset', 'true_label'])
            
        def try_merge(base_df, expected_name, pred_file):
            pred_path = os.path.join(results_dir, pred_file)
            if os.path.exists(pred_path):
                preds = pd.read_csv(pred_path)
                if len(preds) == len(base_df):
                    base_df[expected_name] = (preds['toxicity_score'].values >= 0.5).astype(int)
                else:
                    base_df[expected_name] = pd.NA
            else:
                base_df[expected_name] = pd.NA
                
        for model_name in models:
            safe = model_name.replace("/", "_")
            try_merge(jigsaw_df, f"Raw {model_name}", f"preds_{safe}_raw.csv")
            try_merge(jigsaw_df, f"Fine-tuned {model_name}", f"preds_{safe}_finetuned.csv")
            try_merge(toxigen_df, f"Fine-tuned {model_name}", f"preds_{safe}_finetuned_ood.csv")
            
        try_merge(jigsaw_df, "ML Baseline (LR)", "preds_Baseline.csv")
        try_merge(toxigen_df, "ML Baseline (LR)", "preds_Baseline_ood.csv")
        
        try_merge(jigsaw_df, "Naive Baseline (Majority)", "preds_Naive.csv")
        try_merge(toxigen_df, "Naive Baseline (Majority)", "preds_Naive_ood.csv")
        
        safe_llama = llama_model.replace("/", "_")
        try_merge(jigsaw_df, f"LLaMA ({llama_model})", f"preds_{safe_llama}_llama.csv")
        try_merge(toxigen_df, f"LLaMA ({llama_model})", f"preds_{safe_llama}_llama_ood.csv")

        final_preds = pd.concat([jigsaw_df, toxigen_df], ignore_index=True)
        out_path = os.path.join(results_dir, "final_predictions.csv")
        final_preds.to_csv(out_path, index=False)
        console.print(f"[green]Saved compiled predictions to {out_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error compiling final predictions csv: {e}[/red]")
