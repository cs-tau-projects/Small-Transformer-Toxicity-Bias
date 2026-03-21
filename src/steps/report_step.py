import os
import pandas as pd

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


def run_report_step(results_dir, llama_model, models):
    print(f"\nGenerating Report from {results_dir}...")
    all_results_dict = {}
    
    # Map filenames back to nice display names
    reverse_map = {m.replace("/", "_"): m for m in models + [llama_model]}
    
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if not fname.endswith(".csv") or fname == "final_report.csv":
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

    final_df = format_final_report(all_results_dict)
    if final_df is not None:
        out_path = os.path.join(results_dir, "final_report.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Saved final report to {out_path}")
