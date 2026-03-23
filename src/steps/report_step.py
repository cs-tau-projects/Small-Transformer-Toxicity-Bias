import os
import pandas as pd
from datetime import datetime

# Define the headers for the new structured log file
CSV_HEADERS = [
    "timestamp", "experiment_name", "model_name", "evaluation_type", 
    "overall_auc", "subgroup", "subgroup_auc", "subgroup_fnr", "subgroup_fpr"
]

def log_results_to_csv(results_dir, all_results_dict, full_config):
    """
    Appends results from a dictionary of dataframes to a master CSV log.
    Transforms wide-format metrics into a long-format for easier analysis.
    """
    log_path = os.path.join(results_dir, "results.csv")
    
    # Prepare a list to hold all the new rows to be appended
    new_rows = []
    
    # Extract details from the config, with defaults
    experiment_name = full_config.get("experiment_name", "default_experiment")
    timestamp = datetime.now().isoformat()
    
    for model_display_name, df in all_results_dict.items():
        # Deconstruct the display name to get model and type
        if " Finetuned" in model_display_name:
            model_name = model_display_name.replace(" Finetuned", "")
            eval_type = "finetuned"
        elif " Raw" in model_display_name:
            model_name = model_display_name.replace(" Raw", "")
            eval_type = "raw"
        else:
            model_name = model_display_name
            eval_type = "baseline"
            
        # The overall AUC is the same for all rows in a given dataframe
        overall_auc = df["1. Overall AUC"].iloc[0]

        # Transform each row (subgroup) of the input dataframe into a dictionary
        for _, row in df.iterrows():
            new_row = {
                "timestamp": timestamp,
                "experiment_name": experiment_name,
                "model_name": model_name,
                "evaluation_type": eval_type,
                "overall_auc": overall_auc,
                "subgroup": row["Identity"],
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
        print(f"Created new structured log at {log_path}")
    else:
        # File exists, append without header
        new_df.to_csv(log_path, mode='a', index=False, header=False)
        print(f"Appended {len(new_rows)} rows to structured log at {log_path}")

def format_final_report(all_results_dict):
    """Combines metrics from all models into a comparative table suitable for an ACL report."""
    if not all_results_dict:
        print("\nNo results to report. Are there CSV files in the results directory?")
        return

    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT (ACL Format)")
    print("="*80)
    
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
    
    if 'Total Examples' in all_results_dict[base_key].columns:
        final_df = all_results_dict[base_key][['Identity', 'Total Examples']].copy()
    else:
        final_df = all_results_dict[base_key][['Identity']].copy()

    for model_name, df in all_results_dict.items():
        sum_df = extract_summary(df, model_name)
        if 'Total Examples' in sum_df.columns and model_name != base_key:
            sum_df = sum_df.drop(columns=['Total Examples'])
        final_df = final_df.merge(sum_df, on='Identity', how='left')
    
    print("\n1. Overall AUC Comparison:")
    auc_cols = ['Identity'] + [c for c in final_df.columns if 'Overall AUC' in c]
    if auc_cols[1:]:  # Ensure matching cols exist
        overall_auc = final_df[auc_cols].head(1).copy()
        overall_auc.loc[0, 'Identity'] = 'Overall Dataset'
        print(overall_auc.to_string(index=False))
    
    print("\n2. Subgroup AUC Comparison:")
    subgroup_cols = ['Identity'] + [c for c in final_df.columns if 'Subgroup AUC' in c]
    subgroup_auc = final_df[subgroup_cols]
    print(subgroup_auc.to_string(index=False))
    
    print("\n3. FNR Comparison (Subgroup and Overall):")
    fnr_cols = ['Identity'] + [c for c in final_df.columns if 'FNR' in c]
    fnr = final_df[fnr_cols]
    print(fnr.to_string(index=False))

    print("\n4. FPR Comparison (Subgroup and Overall):")
    fpr_cols = ['Identity'] + [c for c in final_df.columns if 'FPR' in c]
    fpr = final_df[fpr_cols]
    print(fpr.to_string(index=False))

    return final_df


def run_report_step(results_dir, llama_model, models, full_config={}):
    print(f"\nGenerating Report from {results_dir}...")
    all_results_dict = {}
    
    # Map filenames back to nice display names
    reverse_map = {m.replace("/", "_"): m for m in models + [llama_model]}
    
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if not fname.endswith(".csv") or fname in ["final_report.csv", "results.csv"]:
                continue
            path = os.path.join(results_dir, fname)
            df = pd.read_csv(path)
            
            # Reverse-engineer the display name from the filename
            if fname == "baseline_metrics.csv":
                all_results_dict["Baseline"] = df
            elif fname.endswith("_raw_metrics.csv"):
                safe_name = fname.replace("_raw_metrics.csv", "")
                real_name = reverse_map.get(safe_name, safe_name)
                all_results_dict[f"{real_name} Raw"] = df
            elif fname.endswith("_finetuned_metrics.csv"):
                safe_name = fname.replace("_finetuned_metrics.csv", "")
                real_name = reverse_map.get(safe_name, safe_name)
                all_results_dict[f"{real_name} Finetuned"] = df

    # NEW: Log results to the master CSV file
    log_results_to_csv(results_dir, all_results_dict, full_config)

    # Keep the old functionality for console and final_report.csv
    final_df = format_final_report(all_results_dict)
    if final_df is not None:
        out_path = os.path.join(results_dir, "final_report.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved final comparison report to {out_path}")
