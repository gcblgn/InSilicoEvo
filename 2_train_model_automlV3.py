"""
Enzyme Optimal Temperature (Topt) Prediction - PyCaret AutoML (Final v7)
================================================================================
Date: 2026-01-23

Fixes:
- 'CatBoost Feature Mismatch' error resolved.
- Plot generation moved before 'finalize_model' step.
- All reports and plots are collected in the 'report' folder.
- LightGBM C++ warnings suppressed.
"""

import pandas as pd
import numpy as np
import argparse
import sys
import warnings
import os
import shutil
import base64
import re
from contextlib import contextmanager

# Suppress warnings
warnings.filterwarnings('ignore')

# --- ADDITIONAL SECTION (START) ---
# Environment variable settings to silence LightGBM
# These settings must be applied BEFORE importing PyCaret.
os.environ['LGBM_VERBOSITY'] = '-1'
os.environ['LIGHTGBM_VERBOSE'] = '-1'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
# --- ADDITIONAL SECTION (END) ---

# PyCaret import - MUST BE DONE BEFORE SUPPRESS
from pycaret.regression import (
    setup, compare_models, create_model, predict_model,
    finalize_model, save_model, pull, get_config, plot_model
)

# ============================================================================
# STDOUT/STDERR SUPPRESSION (for LightGBM C++ messages)
# ============================================================================

@contextmanager
def suppress_output():
    """Suppresses stdout and stderr - effective for LightGBM C++ messages"""
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    devnull = os.open(os.devnull, os.O_WRONLY)
    
    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull)

def create_report_dir(dir_name='report'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def load_data(file_path, target='topt'):
    for sep in [';', ',']:
        try:
            df = pd.read_csv(file_path, sep=sep)
            cols_lower = [c.lower() for c in df.columns]
            if target in cols_lower:
                target_col = df.columns[cols_lower.index(target)]
                if target_col != target:
                    df = df.rename(columns={target_col: target})
                print(f"Data loaded: {len(df)} records, {len(df.columns)} columns")
                return df
        except:
            continue
    print(f"Error: Could not read file {file_path}!")
    sys.exit(1)

def prepare_data(df, target='topt'):
    # Base columns to drop (always non-feature columns)
    drop_columns = ['id', 'ec', 'uniprot_id', 'domain', 'organism',
                    'ogt_note', 'topt_note', 'sequence']
    # If target is 'topt', also drop 'ogt' (and vice versa)
    if target == 'topt':
        drop_columns.append('ogt')
    elif target == 'ogt':
        drop_columns.append('topt')
    existing_drops = [col for col in drop_columns if col in df.columns]
    df_clean = df.drop(columns=existing_drops, errors='ignore')

    # Remove NaN rows
    if df_clean.isnull().sum().sum() > 0:
        before = len(df_clean)
        df_clean = df_clean.dropna()
        print(f"   {before - len(df_clean)} rows removed due to NaN.")

    # Remove Inf values
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df_clean[numeric_cols]).any(axis=1)
    if inf_mask.sum() > 0:
        print(f"   {inf_mask.sum()} rows removed due to Inf values.")
        df_clean = df_clean[~inf_mask]

    print(f"Prepared data: {len(df_clean)} records, {len(df_clean.columns)} columns")
    return df_clean

def generate_plots(model, report_folder):
    """Saves plots to report folder and returns markdown image references."""
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    plots = [
        ('residuals', 'Residual Plot'),
        ('error', 'Prediction Error'),
        ('feature', 'Feature Importance')
    ]

    md_lines = []
    for plot_type, plot_name in plots:
        try:
            saved_filename = plot_model(model, plot=plot_type, save=True)

            source_path = saved_filename
            target_path = os.path.join(report_folder, os.path.basename(saved_filename))

            if os.path.exists(target_path):
                os.remove(target_path)

            shutil.move(source_path, target_path)
            print(f"   {plot_name} saved.")

            # Add markdown image reference (relative path inside report folder)
            img_filename = os.path.basename(target_path)
            md_lines.append(f"### {plot_name}\n")
            md_lines.append(f"![{plot_name}]({img_filename})\n")

        except Exception as e:
            print(f"    {plot_name} could not be created. Error: {e}")

    return md_lines

def generate_catboost_learning_curve(report_folder):
    """Generates CatBoost training error curve. Returns markdown image lines."""
    import matplotlib.pyplot as plt

    catboost_dir = 'catboost_info'
    learn_error_path = os.path.join(catboost_dir, 'learn_error.tsv')

    if not os.path.exists(learn_error_path):
        print(f"    CatBoost learn_error.tsv not found.")
        return []

    try:
        error_df = pd.read_csv(learn_error_path, sep='\t')

        plt.figure(figsize=(10, 6))
        plt.plot(error_df['iter'], error_df['RMSE'], 'b-', linewidth=1.5)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('CatBoost Learning Curve', fontsize=14)
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(report_folder, 'CatBoost_Learning_Curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   CatBoost Learning Curve saved.")
        return [
            "### CatBoost Learning Curve\n",
            "![CatBoost Learning Curve](CatBoost_Learning_Curve.png)\n"
        ]

    except Exception as e:
        print(f"   CatBoost Learning Curve could not be created. Error: {e}")
        return []

def build_hyperparameters_md(best_models, comparison_df):
    """Returns markdown lines for model hyperparameters."""
    lines = []
    lines.append("## Model Hyperparameters\n")
    for i, model in enumerate(best_models):
        model_name = comparison_df.index[i]
        lines.append(f"### Model {i+1}: {model_name}\n")
        try:
            params = model.get_params()
            lines.append("| Parameter | Value |")
            lines.append("| --- | --- |")
            for param_name, param_value in sorted(params.items()):
                lines.append(f"| {param_name} | {param_value} |")
        except Exception as e:
            lines.append(f"Could not retrieve parameters: {e}")
        lines.append("")
    return lines

def cleanup_temp_files():
    """Cleans up temporary files and folders."""
    print("\n" + "="*60)
    print("Cleaning Up Temporary Files")
    print("="*60)
    
    if os.path.exists('__pycache__'):
        try:
            shutil.rmtree('__pycache__')
            print("   __pycache__ folder deleted.")
        except Exception as e:
            print(f"   Could not delete __pycache__: {e}")
    
    if os.path.exists('catboost_info'):
        try:
            shutil.rmtree('catboost_info')
            print("   catboost_info folder deleted.")
        except Exception as e:
            print(f"   Could not delete catboost_info: {e}")

def md_to_html(md_path, html_path):
    """Converts a markdown report to a standalone HTML file with embedded images."""
    report_dir = os.path.dirname(os.path.abspath(md_path))

    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # --- Embed images as base64 ---
    def replace_image(match):
        alt_text = match.group(1)
        img_file = match.group(2)
        img_path = os.path.join(report_dir, img_file)
        if os.path.exists(img_path):
            with open(img_path, 'rb') as img_f:
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
            ext = os.path.splitext(img_file)[1].lstrip('.').lower()
            mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
            return f'<img src="data:{mime};base64,{b64}" alt="{alt_text}" style="max-width:100%;">'
        return match.group(0)

    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, md_text)

    # --- Simple markdown to HTML conversion ---
    html_lines = []
    in_table = False
    in_list = False
    lines = md_text.split('\n')

    for line in lines:
        stripped = line.strip()

        # Headings
        if stripped.startswith('### '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h3>{stripped[4:]}</h3>')
            continue
        if stripped.startswith('## '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h2>{stripped[3:]}</h2>')
            continue
        if stripped.startswith('# '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(f'<h1>{stripped[2:]}</h1>')
            continue

        # Already-converted <img> tags
        if stripped.startswith('<img '):
            if in_list:
                html_lines.append('</ol>')
                in_list = False
            html_lines.append(stripped)
            continue

        # Table rows
        if stripped.startswith('|') and stripped.endswith('|'):
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            # Separator row (| --- | --- |)
            if all(re.match(r'^-+$', c) for c in cells):
                continue
            if not in_table:
                in_table = True
                html_lines.append('<table border="1" cellpadding="6" cellspacing="0" '
                                  'style="border-collapse:collapse; margin:10px 0;">')
                html_lines.append('<tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr>')
            else:
                html_lines.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            continue
        else:
            if in_table:
                html_lines.append('</table>')
                in_table = False

        # Ordered list items (1. item)
        list_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if list_match:
            if not in_list:
                in_list = True
                html_lines.append('<ol>')
            html_lines.append(f'<li>{list_match.group(2)}</li>')
            continue
        else:
            if in_list:
                html_lines.append('</ol>')
                in_list = False

        # Bold: **text**
        stripped = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)

        # Bullet list item: - text
        if stripped.startswith('- '):
            html_lines.append(f'<p style="margin:2px 0 2px 20px;">{stripped[2:]}</p>')
            continue

        # Empty line
        if not stripped:
            continue

        # Regular paragraph
        html_lines.append(f'<p>{stripped}</p>')

    if in_table:
        html_lines.append('</table>')
    if in_list:
        html_lines.append('</ol>')

    body = '\n'.join(html_lines)

    html_doc = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>AutoML Results Report</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }}
  h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; }}
  h3 {{ color: #7f8c8d; margin-top: 20px; }}
  table {{ font-size: 13px; }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f2f2f2; }}
  img {{ display: block; margin: 10px 0 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_doc)

    print(f"   HTML report saved: {html_path}")

def df_to_markdown(df):
    """Converts a pandas DataFrame to a markdown table string."""
    cols = list(df.columns)
    # Header row
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    # Data rows
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(row[c]) for c in cols) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows)

def train_and_report_process(df, report_folder, make_plots=True,
                              models=None, fold=3, target='topt'):
    if models is None:
        models = ['catboost', 'lightgbm', 'xgboost', 'rf', 'et']

    md_path = os.path.join(report_folder, 'AutoML_Results_Report.md')
    model_path = os.path.join(report_folder, f'{target}_automl_model')

    # Markdown report content accumulator
    md_lines = []
    md_lines.append(f"# AutoML Results Report\n")
    md_lines.append(f"- **Target:** {target}")
    md_lines.append(f"- **Models:** {', '.join(models)}")
    md_lines.append(f"- **CV Folds:** {fold}")
    md_lines.append("")

    print("\n" + "="*60)
    print(f"PyCaret AutoML - Report Mode (Folder: {report_folder})")
    print(f"Target: {target} | Models: {', '.join(models)} | Folds: {fold}")
    print("="*60)

    # 1. SETUP
    print("\n[1/7] Data preparation and Feature Selection...")

    # Clean string-type numeric columns
    for col in df.columns:
        if df[col].dtype == 'object' and col != target:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    # Remove remaining object-type columns (non-numeric)
    non_numeric = df.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print(f"   Removing non-numeric columns: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # NaN check (may occur after to_numeric conversion)
    if df.isnull().sum().sum() > 0:
        before = len(df)
        df = df.dropna()
        print(f"   {before - len(df)} rows removed due to NaN (after type conversion).")

    print(f"   Data for setup: {len(df)} records, {len(df.columns)} columns")
    print(f"   Data types: {dict(df.dtypes.value_counts())}")

    reg_setup = setup(
        data=df,
        target=target,
        train_size=0.8,
        session_id=42,
        normalize=True,
        feature_selection=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.9,
        fold=fold,
        verbose=False,
        log_experiment=False,
        n_jobs=1
    )

    X_transformed = get_config('X_transformed')
    selected_features = X_transformed.columns.tolist()

    print(f"   Number of selected features: {len(selected_features)}")

    # Add selected features to markdown report
    md_lines.append("## Selected Features\n")
    md_lines.append(f"Total Count: **{len(selected_features)}**\n")
    for i, feat in enumerate(selected_features, 1):
        md_lines.append(f"{i}. {feat}")
    md_lines.append("")

    # 2. MODEL COMPARISON
    print(f"\n[2/7] Comparing models: {', '.join(models)} ({fold}-fold CV)...")

    try:
        best_models = compare_models(
            include=models,
            n_select=min(5, len(models)),
            sort='RMSE',
            turbo=True,
            verbose=True
        )
    except Exception as e:
        print(f"\n   ERROR: compare_models failed: {e}")
        print("   Trying models one by one...")

        # Try models one by one - find which ones work
        working_models = []
        for model_id in models:
            try:
                m = create_model(model_id, verbose=False)
                working_models.append(model_id)
                print(f"     {model_id}: SUCCESS")
            except Exception as e2:
                print(f"     {model_id}: FAILED - {e2}")

        if working_models:
            best_models = compare_models(
                include=working_models,
                n_select=min(5, len(working_models)),
                sort='RMSE',
                turbo=True,
                verbose=True
            )
        else:
            print("\n   CRITICAL ERROR: No model could be trained!")
            # Write error report
            md_lines.append("## ERROR\n")
            md_lines.append("Model training failed. No model could be trained.\n")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
            sys.exit(1)

    # Wrap in list if compare_models returns a single model
    if not isinstance(best_models, list):
        best_models = [best_models]

    comparison_df = pull()

    if best_models is None or len(best_models) == 0 or comparison_df.empty:
        print("\n   ERROR: No model was successfully trained!")
        print("   Possible causes:")
        print("     - Dataset may contain NaN/Inf values")
        print(f"     - Target variable ({target}) may be constant")
        print("     - Too many features / too few records")
        print("\n   Performing data check...")
        print(f"     Row count: {len(df)}")
        print(f"     Column count: {len(df.columns)}")
        print(f"     Total NaN: {df.isnull().sum().sum()}")
        print(f"     Total Inf: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
        if target in df.columns:
            print(f"     {target} min: {df[target].min()}, max: {df[target].max()}, std: {df[target].std():.4f}")
        # Write error report
        md_lines.append("## ERROR\n")
        md_lines.append("Model training failed.\n")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        sys.exit(1)

    # Add comparison to markdown
    md_lines.append("## All Models Summary\n")
    md_lines.append(df_to_markdown(comparison_df.reset_index()))
    md_lines.append("")

    print("   Comparison table added to report.")
    print(comparison_df.iloc[:, :6].to_string())

    # Add hyperparameters to markdown report
    md_lines.extend(build_hyperparameters_md(best_models, comparison_df))

    # 3. DETAILED CV RESULTS
    print(f"\n[3/7] Detailed CV results for top models...")

    for i, model in enumerate(best_models):
        model_name = comparison_df.index[i]
        _ = create_model(model)
        cv_results = pull()
        md_lines.append(f"## CV Detail - {model_name}\n")
        md_lines.append(df_to_markdown(cv_results.reset_index()))
        md_lines.append("")

    print("   CV details added to report.")

    # 4. TEST SET RESULTS
    print(f"\n[4/7] Test set (Hold-out) performance...")
    best_model = best_models[0]

    with suppress_output():
        pred_holdout = predict_model(best_model, verbose=False)

    test_results = pull()
    md_lines.append("## Test Set Holdout Result\n")
    md_lines.append(df_to_markdown(test_results.reset_index()))
    md_lines.append("")

    # 5. PLOTS
    if make_plots:
        print(f"\n[5/7] Generating plots (before Finalize step)...")
        md_lines.append("## Visualizations\n")
        plot_md = generate_plots(best_model, report_folder)
        md_lines.extend(plot_md)
        catboost_md = generate_catboost_learning_curve(report_folder)
        md_lines.extend(catboost_md)
    else:
        print(f"\n[5/7] Plot generation skipped.")

    # 6. REPORT SAVE (Markdown + HTML)
    print(f"\n[6/7] Saving reports...")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"   Markdown report saved: {md_path}")

    html_path = os.path.join(report_folder, 'AutoML_Results_Report.html')
    md_to_html(md_path, html_path)

    # 7. MODEL FINALIZE AND SAVE
    print(f"\n[7/7] Finalizing model (retraining with all data) and saving...")
    with suppress_output():
        final_model = finalize_model(best_model)

    save_model(final_model, model_path)
    print(f"   Model saved: {model_path}.pkl")

    return final_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--models', nargs='+',
                        choices=['catboost', 'lightgbm', 'xgboost', 'rf', 'et'],
                        default=['catboost', 'lightgbm', 'xgboost', 'rf', 'et'],
                        help='ML models to include (default: all)')
    parser.add_argument('--fold', type=int, choices=[2, 3, 4, 5], default=3,
                        help='Number of cross-validation folds (default: 3)')
    parser.add_argument('--target', choices=['topt', 'ogt'], default='topt',
                        help='Target variable for prediction (default: topt)')
    args = parser.parse_args()

    report_folder = create_report_dir('report')
    df = load_data(args.input_file, target=args.target)
    df_clean = prepare_data(df, target=args.target)

    train_and_report_process(df_clean, report_folder,
                             make_plots=not args.no_plots,
                             models=args.models,
                             fold=args.fold,
                             target=args.target)

    cleanup_temp_files()

    print(f"\nPROCESS COMPLETED SUCCESSFULLY.")
    print(f"All files are in '{os.path.abspath(report_folder)}' folder.")

if __name__ == "__main__":
    main()
