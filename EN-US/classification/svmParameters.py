# -*- coding: utf-8 -*-
"""
SVM Parameters Evaluation Script - Gemini v8.0 (Data Leakage Protection)
Features:
- AUTOMATIC Dirty Feature Removal (Anti-Cheating)
- Real Standard Deviation on Confusion Matrices
- Educational Guide for Stability & Data Leakage
- EXACT HTML/CSS preservation (No text changes in legends)
"""

from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import os
from time import process_time, time
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns

#========================================================================
# METRICS & UTILS
#========================================================================

def calculate_detection_metrics(cm):
    """Calculates malware detection metrics."""
    if cm is None or cm.shape != (2, 2): return None
    TN, FP, FN, TP = float(cm[0, 0]), float(cm[0, 1]), float(cm[1, 0]), float(cm[1, 1])
    total = TN + FP + FN + TP
    if total == 0: return None
    
    metrics = {
        'accuracy': (TP + TN) / total,
        'detection_rate': TP / (TP + FN) if (TP + FN) > 0 else 0, # Recall
        'false_positive_rate': FP / (FP + TN) if (FP + TN) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'f1_score': 0,
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
    }
    if metrics['precision'] + metrics['detection_rate'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['detection_rate']) / \
                              (metrics['precision'] + metrics['detection_rate'])
    return metrics

def print_detection_metrics(metrics, title="Metrics"):
    """Prints metrics to terminal."""
    if metrics is None: return
    print(f"\n{'─'*60}\n  📊 {title}\n{'─'*60}")
    print(f"    Detection Rate (Recall):     {metrics['detection_rate']*100:6.2f}%")
    print(f"    False Positive Rate (FPR):   {metrics['false_positive_rate']*100:6.2f}%")
    print(f"    Precision:                   {metrics['precision']*100:6.2f}%")
    print(f"    Confusion Matrix (Avg):      [TN: {metrics['TN']}, FP: {metrics['FP']}]")
    print(f"                                 [FN: {metrics['FN']}, TP: {metrics['TP']}]")
    print(f"{'─'*60}\n")

def print_progress_info(current_iter, total_iter, start_time, detail_str=""):
    percent = 100.0 * current_iter / total_iter if total_iter > 0 else 0
    elapsed = time() - start_time
    if current_iter > 0:
        rem_time = (elapsed / current_iter) * (total_iter - current_iter)
        rem_str = format_time(rem_time)
    else: rem_str = "..."
    
    bar_len = 30
    filled = int(bar_len * current_iter / total_iter) if total_iter > 0 else 0
    bar = '█' * filled + '░' * (bar_len - filled)
    sys.stdout.write(f"\r  ⏱️  [{bar}] {percent:.1f}% | {format_time(elapsed)} < {rem_str} | {detail_str}")
    sys.stdout.flush()

def format_time(seconds):
    if seconds < 60: return f"{int(seconds)}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

#========================================================================
# DIAGNOSIS & ESTIMATION
#========================================================================

def estimate_and_filter_kernels(y, x, kernels_to_test, cost_vector, gamma_vector, k_folds):
    n_samples = len(y)
    max_feature_idx = 0
    if len(x) > 0:
        for row in x[:100]: 
            if row: max_feature_idx = max(max_feature_idx, max(row.keys()))
    n_features = max_feature_idx
    
    total_theo = n_samples * n_features
    total_non_zeros = sum(len(row) for row in x)
    density = total_non_zeros / total_theo if total_theo > 0 else 1.0
    base_factor = 2.0e-7 
    MAX_TIME = 3600 
    
    kernels_kept = []
    skipped_info = {} 
    
    print(f"\n{'='*80}\n⏱️  SMART TIME ESTIMATION\n{'='*80}")
    total_est = 0
    total_iters = 0
    
    for t in kernels_to_test:
        k_name = kernel_str(t)
        gammas = [0] if t == 0 else gamma_vector
        iters_kernel = len(cost_vector) * len(gammas) * k_folds
        comp_mult = 5.0 if t == 1 else 1.2 if t == 2 else 1.0
        est_time = base_factor * (n_samples ** 2) * n_features * density * comp_mult * iters_kernel
        
        if est_time > MAX_TIME:
             skipped_info[t] = {'reason': 'Discarded (Too Slow)', 'detail': f'Est: {format_time(est_time)}'}
             print(f"  ⚠️  Skipping {k_name}: Est. {format_time(est_time)} > Limit")
             continue
             
        total_est += est_time
        total_iters += iters_kernel
        kernels_kept.append(t)
        print(f"  > Accepted {k_name:<10}: ~{format_time(est_time)}")

    print(f"\n  ⏱️  Total Est. Runtime: ~{format_time(total_est)}")
    print("="*80 + "\n")
    return kernels_kept, skipped_info, total_iters

def diagnose_svm_performance(results, kernel_name):
    diagnosis = {'status': 'unknown', 'problem': None, 'explanation': '', 'recommendation': ''}
    if not results: return diagnosis

    acc_test = results.get('accuracy_test', 0)
    acc_train = results.get('accuracy_train', 0)
    std = results.get('std_test', 0)
    cm = results.get('confusion_matrix_test', None)
    
    det_rate, fpr = 0, 0
    if cm is not None and cm.size == 4:
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        pos = TP + FN
        neg = TN + FP
        det_rate = TP / pos if pos > 0 else 0
        fpr = FP / neg if neg > 0 else 0

    if acc_train > 95.0 and (acc_train - acc_test) > 20.0:
        diagnosis['status'] = 'critical'
        diagnosis['problem'] = 'Severe Overfitting (Memorization)'
        diagnosis['explanation'] = (f"The model memorized training data ({acc_train:.1f}%) but failed on new files ({acc_test:.1f}%).")
        diagnosis['recommendation'] = "Reduce Cost (C). If using Radial Basis Function, check scaling."
    elif acc_train < 60.0 and acc_test < 60.0:
        diagnosis['status'] = 'critical'
        diagnosis['problem'] = 'Underfitting (Model too Simple)'
        diagnosis['explanation'] = f"Model failed to learn patterns (Acc: {acc_train:.1f}%)."
        diagnosis['recommendation'] = "Increase Cost (C) or Gamma."
    elif det_rate < 0.05:
        diagnosis['status'] = 'critical'
        diagnosis['problem'] = 'Class Collapse (Zero Detection)'
        diagnosis['explanation'] = "The model predicts everything as Benign (Malware ignored)."
        diagnosis['recommendation'] = "Enable Class Weights (-wi) or increase C."
    elif 'Sigmoid' in kernel_name and acc_test < 70.0:
        diagnosis['status'] = 'warning'
        diagnosis['problem'] = 'Sigmoid Instability'
        diagnosis['explanation'] = "Sigmoid kernels often behave chaotically on tabular data."
        diagnosis['recommendation'] = "Switch to Radial Basis Function."
    elif 'Linear' in kernel_name and acc_test < 75.0:
        diagnosis['status'] = 'poor'
        diagnosis['problem'] = 'Non-Linear Data'
        diagnosis['explanation'] = "Linear kernel cannot separate this complex data."
        diagnosis['recommendation'] = "Switch to Radial Basis Function."
    elif std > 12.0:
        diagnosis['status'] = 'unstable'
        diagnosis['problem'] = 'High Variance'
        diagnosis['explanation'] = f"Results fluctuate wildly (±{std:.1f}%) between folds."
        diagnosis['recommendation'] = "Increase regularization (Lower C)."
    elif fpr > 0.20:
        diagnosis['status'] = 'warning'
        diagnosis['problem'] = 'High False Alarm Rate'
        diagnosis['explanation'] = f"FPR is {fpr*100:.1f}%. Too many safe files flagged."
        diagnosis['recommendation'] = "Tune Gamma or Negative Class Weights."
    elif acc_test > 90.0:
        diagnosis['status'] = 'good'
        diagnosis['problem'] = None
        diagnosis['explanation'] = f"Excellent performance ({acc_test:.1f}%) with balanced detection."
        diagnosis['recommendation'] = "Model suitable for production."
    else:
        diagnosis['status'] = 'good'
        diagnosis['problem'] = None
        diagnosis['explanation'] = f"Stable model with decent accuracy ({acc_test:.1f}%)."
        diagnosis['recommendation'] = "Try Grid Search with finer steps."
    return diagnosis

#========================================================================
# SVM CORE & PLOTTING
#========================================================================

def kernel_str(t):
    return {0:'Linear', 1:'Polynomial', 2:'Radial Basis Function', 3:'Sigmoid'}.get(t, 'Unknown')

def plot_and_save_cm(cm_mean, cm_std, title, filename):
    """Plots CM with Mean and Standard Deviation using custom Labels (Benign/Malign)."""
    if cm_mean is None: return

    # Normalize
    cm_sum = cm_mean.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = cm_mean.astype('float') / cm_sum * 100
    cm_pct = np.nan_to_num(cm_pct)
    
    # Prepare Annotation
    if cm_std is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
             cm_std_pct = cm_std.astype('float') / cm_sum * 100
        cm_std_pct = np.nan_to_num(cm_std_pct)
        annot = (np.asarray(["{0:.1f}%\n±{1:.2f}%".format(m, s)
                            for m, s in zip(cm_pct.flatten(), cm_std_pct.flatten())])
                ).reshape(cm_pct.shape)
    else:
        annot = np.array([[f'{v:.1f}%' for v in r] for r in cm_pct])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap=plt.cm.Blues, 
                cbar_kws={'label': 'Percentage (%)'},
                annot_kws={"size": 12, "weight": "bold"})
    
    # Custom Labels
    plt.yticks([0.5, 1.5], ['Benign', 'Malign'], va='center', fontsize=10)
    plt.xticks([0.5, 1.5], ['Pred Benign', 'Pred Malign'], fontsize=10)

    plt.title(title, fontsize=11, pad=15, fontweight='bold')
    plt.ylabel('True Label', fontsize=10, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_libsvm(y, x, filename):
    with open(filename, 'w') as f:
        for l, feats in zip(y, x):
            f_str = ' '.join(f"{k}:{v}" for k, v in sorted(feats.items()))
            f.write(f"{int(l)} {f_str}\n")

def pruningDataset(y, x, threshold):
    """
    Analyzes correlations.
    1. AUTOMATICALLY removes features with correlation >= 0.999 (Data Leakage).
    2. Applies optional threshold pruning.
    3. Saves selected_features.csv.
    """
    max_f = 0
    for s in x: 
        if s: max_f = max(max_f, max(s.keys()))
    x_arr = np.zeros((len(x), max_f))
    for i, s in enumerate(x):
        for f, v in s.items(): 
            if f > 0: x_arr[i, f-1] = v
            
    print(f"\nAnalyzing Feature Correlations & Checking for Data Leakage...")
    
    dirty_features = [] # Indices of dirty features (1-based)
    corrs = np.zeros(x_arr.shape[1])
    
    with open('selected_features.csv', 'w') as f_csv:
        f_csv.write("Feature_ID,Correlation,Status\n")
        
        for i in range(x_arr.shape[1]):
            if np.std(x_arr[:, i]) > 0: 
                c = np.corrcoef(x_arr[:, i], y)[0, 1]
                corrs[i] = c
                
                # Check for Data Leakage (Dirty Features)
                if abs(c) >= 0.999: 
                    dirty_features.append(i+1)
                    f_csv.write(f"{i+1},{c:.4f},DIRTY (Removed)\n")
                elif threshold and abs(c) >= threshold:
                    f_csv.write(f"{i+1},{c:.4f},Selected\n")
                else:
                    f_csv.write(f"{i+1},{c:.4f},Pruned\n")
            else:
                 f_csv.write(f"{i+1},0.0000,Pruned (Constant)\n")

    dirty_count = len(dirty_features)
    if dirty_count > 0:
        print(f"  🚨 AUTOMATIC CLEANING: Removed {dirty_count} features with correlation ~1.0 (Data Leakage).")
        print(f"     (IDs: {dirty_features[:10]}...)")
    else:
        print("  ✅ No Data Leakage detected (no features with correlation 1.0).")

    # Filter Features: Keep if NOT dirty AND (meets threshold OR no threshold)
    valid_indices = []
    for i in range(x_arr.shape[1]):
        feat_id = i + 1
        is_dirty = feat_id in dirty_features
        meets_thresh = True
        if threshold:
            meets_thresh = abs(corrs[i]) >= threshold
        
        if not is_dirty and meets_thresh:
            valid_indices.append(i)

    # Rebuild Dataset
    mapping = {old: new for new, old in enumerate(valid_indices, 1)}
    x_cleaned = []
    for s in x:
        # Only keep features that are in valid_indices
        # Adjust keys to 0-based for check, then map to new 1-based ID
        new_row = {}
        for k, v in s.items():
            if (k-1) in valid_indices:
                new_row[mapping[k-1]] = v
        x_cleaned.append(new_row)
    
    save_libsvm(y, x_cleaned, 'globalPruned.libsvm')
    return y, x_cleaned, dirty_count

def svmKfold(y, x, t, cost, gamma, k_folds, callback=None):
    np.random.seed(1)
    kf = KFold(n_splits=k_folds, shuffle=True)
    acc_tr, acc_te, t_tr, t_te = [], [], [], []
    cms_tr, cms_te = [], []
    
    param = f'-t {t} -c {cost} -g {gamma} -m 500 -e 0.05 -q'
    
    for tr_idx, te_idx in kf.split(x):
        x_tr, x_te = [x[i] for i in tr_idx], [x[i] for i in te_idx]
        y_tr, y_te = [y[i] for i in tr_idx], [y[i] for i in te_idx]
        
        t0 = process_time()
        m = svm_train(y_tr, x_tr, param)
        t1 = process_time()
        
        _, p_acc_tr, _ = svm_predict(y_tr, x_tr, m, '-q')
        t2 = process_time()
        p_lbl, p_acc_te, _ = svm_predict(y_te, x_te, m, '-q')
        t3 = process_time()
        
        acc_tr.append(p_acc_tr[0]); acc_te.append(p_acc_te[0])
        t_tr.append(t1-t0); t_te.append(t3-t2)
        cms_te.append(confusion_matrix(y_te, p_lbl, labels=sorted(np.unique(y))))
        if callback: callback()
    
    cm_mean = np.mean(cms_te, axis=0)
    cm_std = np.std(cms_te, axis=0) 
        
    return np.mean(acc_tr), np.std(acc_tr), np.mean(acc_te), np.std(acc_te), \
           np.mean(t_tr), np.std(t_tr), np.mean(t_te), np.std(t_te), \
           cm_std, cm_mean

class svmParameters():
    def main(self, dataset, threshold):
        y, x = svm_read_problem(dataset)
        
        # Now returns dirty_count as well
        y, x, dirty_count = pruningDataset(y, x, threshold)

        c_vec = [1, 1000]
        g_vec = [1]
        k_folds = 5
        
        kernels_run, skip_info, tot_iters = estimate_and_filter_kernels(y, x, range(4), c_vec, g_vec, k_folds)
        if not kernels_run: return None, {}, skip_info, dirty_count

        min_acc, max_acc = 101, -1
        max_res, min_res = {}, {}
        res = {}
        
        start_t = time()
        curr_iter = 0
        def cb(): nonlocal curr_iter; curr_iter += 1; print_progress_info(curr_iter, tot_iters, start_t, "Running...")

        for t in kernels_run:
            runs = []
            gs = [0] if t == 0 else g_vec
            for c in c_vec:
                for g in gs:
                    print_progress_info(curr_iter, tot_iters, start_t, f"K:{kernel_str(t)} C:{c}")
                    mtr, str_, mte, ste, mttr, sttr, mtte, stte, stm, mcm = svmKfold(y, x, t, c, g, k_folds, cb)
                    sys.stdout.write("\r" + " "*100 + "\r")
                    dm = calculate_detection_metrics(mcm)
                    print(f"✅ Finished: {kernel_str(t)} | C={c} | G={g} | Acc: {mte:.2f}%")
                    if dm: print_detection_metrics(dm, f"Results {kernel_str(t)}")
                    
                    diag = diagnose_svm_performance({
                        'accuracy_train': mtr, 'accuracy_test': mte, 
                        'std_test': ste, 'confusion_matrix_test': mcm
                    }, kernel_str(t))
                    
                    dat = {
                        "accuracy_train": mtr, "std_train": str_, "accuracy_test": mte, "std_test": ste,
                        "time_train": mttr, "std_time_train": sttr, "time_test": mtte, "std_time_test": stte,
                        "cost": c, "gamma": g, "confusion_matrix_test": mcm, "confusion_matrix_std": stm,
                        "diagnosis": diag
                    }
                    runs.append(dat)
                    if mte > max_acc: max_acc = mte; max_res = {**dat, 'kernel_id': t}
                    if mte < min_acc: min_acc = mte; min_res = {**dat, 'kernel_id': t}
            
            if runs:
                res[t] = { "max_test": max(runs, key=lambda r: r['accuracy_test']), "min_test": min(runs, key=lambda r: r['accuracy_test']) }

        print_progress_info(tot_iters, tot_iters, start_t, "Done!")
        print("\n")
        glob = {"max": max_res, "min": min_res}
        return glob, res, skip_info, dirty_count

#========================================================================
# HTML REPORT GENERATION (PRESERVED LAYOUT + LEAKAGE REPORT)
#========================================================================

def generate_html_report(global_results, kernel_results, skipped_info, dirty_count, output_file='svm_report.html'):
    if not global_results or not global_results.get('max'): return
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(script_dir, 'svm_report_images')
    os.makedirs(img_dir, exist_ok=True)
    def get_img(f): return os.path.join(img_dir, f)

    # Plot Globals
    cm_global_best = None
    if global_results['max'].get('confusion_matrix_test') is not None:
        plot_and_save_cm(
            global_results['max']['confusion_matrix_test'],
            global_results['max'].get('confusion_matrix_std'),
            f"Average CM - Best Global ({kernel_str(global_results['max']['kernel_id'])})", 
            get_img('cm_global_best.png')
        )
        cm_global_best = 'svm_report_images/cm_global_best.png'
        
    cm_global_worst = None
    if global_results['min'].get('confusion_matrix_test') is not None:
        plot_and_save_cm(
            global_results['min']['confusion_matrix_test'], 
            global_results['min'].get('confusion_matrix_std'),
            f"Average CM - Worst Global ({kernel_str(global_results['min']['kernel_id'])})", 
            get_img('cm_global_worst.png')
        )
        cm_global_worst = 'svm_report_images/cm_global_worst.png'

    # Plot Kernels
    act_cm = {}
    diagnostics_dict = {} 
    
    for k_id, data in kernel_results.items():
        k_name = kernel_str(k_id)
        k_key = k_name.replace(" ", "_")
        best_img, worst_img = None, None
        
        if data.get('max_test'):
            plot_and_save_cm(
                data['max_test']['confusion_matrix_test'], 
                data['max_test'].get('confusion_matrix_std'),
                f'Average CM - Best Test ({k_name})', 
                get_img(f'cm_{k_key}_best.png')
            )
            best_img = f'svm_report_images/cm_{k_key}_best.png'
            if 'diagnosis' in data['max_test']: diagnostics_dict[k_name] = data['max_test']['diagnosis']

        if data.get('min_test'):
            plot_and_save_cm(
                data['min_test']['confusion_matrix_test'], 
                data['min_test'].get('confusion_matrix_std'),
                f'Average CM - Worst Test ({k_name})', 
                get_img(f'cm_{k_key}_worst.png')
            )
            worst_img = f'svm_report_images/cm_{k_key}_worst.png'
        act_cm[k_name] = {'best': best_img, 'worst': worst_img}

    for k_id, info in skipped_info.items():
        k_name = kernel_str(k_id)
        diagnostics_dict[k_name] = {
            'status': 'skipped', 'problem': info['reason'], 
            'explanation': f"Skipped: {info['detail']}.", 
            'recommendation': 'Reduce data size or increase timeout.'
        }

    # Template Prep
    global_results['max']['act'] = kernel_str(global_results['max']['kernel_id'])
    global_results['max']['n_hidden'] = f"C={global_results['max']['cost']}, γ={global_results['max']['gamma']}"
    global_results['min']['act'] = kernel_str(global_results['min']['kernel_id'])
    global_results['min']['n_hidden'] = f"C={global_results['min']['cost']}, γ={global_results['min']['gamma']}"

    act_results = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_str(k_id)
        act_results[k_name] = data
        if 'max_test' in data:
            data['max_test']['n_hidden'] = f"C={data['max_test']['cost']}, γ={data['max_test']['gamma']}"
        if 'min_test' in data:
            data['min_test']['n_hidden'] = f"C={data['min_test']['cost']}, γ={data['min_test']['gamma']}"

    # HTML TEMPLATE with DATA LEAKAGE SECTION ADDED
    html_template = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVM Dashboard - Results Report</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif, Arial; background: linear-gradient(135deg, #8B1538 0%, #A91E4A 50%, #6B1429 100%); min-height: 100vh; color: #333; }
            .dashboard-container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
            .header h1 { font-size: 2.5em; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }
            .subtitle { font-size: 1.2em; color: #666; font-weight: 300; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; }
            .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }
            .stat-card.best { border-left: 10px solid #A5D7A7; } .stat-card.worst { border-left: 10px solid #f9a19a; }
            .card-header { display: flex; align-items: center; margin-bottom: 20px; }
            .card-icon { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 1.5em; font-weight: bold; color: white; }
            .card-icon.best { background: linear-gradient(45deg, #4CAF50, #66BB6A); }
            .card-icon.worst { background: linear-gradient(45deg, #f44336, #EF5350); }
            .card-title { font-size: 1.3em; font-weight: 600; }
            .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); }
            .metric-row:last-child { border-bottom: none; }
            .metric-label { font-weight: 500; color: #555; }
            .metric-value { font-weight: 600; padding: 4px 12px; border-radius: 20px; }
            .metric-value.best { background: rgba(76, 175, 80, 0.1); }
            .metric-value.worst { background: rgba(244, 67, 54, 0.1); }
            .cm-container { text-align: center; margin-top: 20px; }
            .cm-image { max-width: 70%; height: auto; border-radius: 5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
            .kernels-section { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
            .section-title { font-size: 2em; margin-bottom: 20px; font-weight: 600; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
            .kernel-group { margin-bottom: 40px; }
            .kernel-title { font-size: 1.5em; font-weight: 600; margin-bottom: 20px; padding: 15px 20px; color: white; border-radius: 10px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); }
            .kernel-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
            .result-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); transition: transform 0.3s ease; }
            .result-card:hover { transform: translateY(-3px); }
            .result-card.best { border: 2px solid #4CAF50; } .result-card.worst { border: 2px solid #f44336; }
            .result-header { display: flex; align-items: center; margin-bottom: 20px; }
            .result-icon { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; color: white; font-weight: bold; }
            .logo-ufpe { height: 150px; width: auto; }
            .result-icon.best { background: #4CAF50; } .result-icon.worst { background: #f44336; }
            .result-title { font-size: 1.2em; font-weight: 600; }
            .metrics-list { list-style: none; margin-bottom: 20px; }
            .metrics-list li { padding: 8px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); display: flex; justify-content: space-between; align-items: center; }
            .metrics-list li:last-child { border-bottom: none; }
            .metric-name { font-weight: 500; color: #555; }
            .metric-val { font-weight: 600; }
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
            .stat-card, .kernels-section { animation: fadeInUp 0.6s ease forwards; }
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <img src="../../src/ufpe_logo.png" alt="University Logo" class="logo-ufpe">
                <h1>SVM - Parameter Evaluation</h1>
                <p class="subtitle">Test Accuracy is the metric of choice for results</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card best">
                    <div class="card-header"><div class="card-icon best">🏆</div><div class="card-title">Best Global Performance</div></div>
                    <div class="metric-row"><span class="metric-label">Config (C, γ)</span><span class="metric-value best">{{ global_results.max.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Best Kernel</span><span class="metric-value best">{{ global_results.max.act }}</span></div>
                    <div class="metric-row"><span class="metric-label">Train Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_train) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_test) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Train Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_train) }}s</span></div>
                    <div class="metric-row"><span class="metric-label">Test Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_test) }}s</span></div>
                    {% if cm_global_best %}<div class="cm-container"><img class="cm-image" src="{{ cm_global_best }}" alt="CM Best Global"></div>{% endif %}
                </div>

                <div class="stat-card worst">
                    <div class="card-header"><div class="card-icon worst">💎</div><div class="card-title">Worst Global Performance</div></div>
                    <div class="metric-row"><span class="metric-label">Config (C, γ)</span><span class="metric-value worst">{{ global_results.min.n_hidden }}</span></div>
                    <div class="metric-row"><span class="metric-label">Worst Kernel</span><span class="metric-value worst">{{ global_results.min.act }}</span></div>
                    <div class="metric-row"><span class="metric-label">Train Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_train) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_test) }}%</span></div>
                    <div class="metric-row"><span class="metric-label">Train Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_train) }}s</span></div>
                    <div class="metric-row"><span class="metric-label">Test Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_test) }}s</span></div>
                    {% if cm_global_worst %}<div class="cm-container"><img class="cm-image" src="{{ cm_global_worst }}" alt="CM Worst Global"></div>{% endif %}
                </div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">Kernel Summary</h2>
                {% for act_name, data in act_results.items() %}
                <div class="kernel-group">
                    <div class="kernel-title">Kernel: {{ act_name }}</div>
                    <div class="kernel-results">
                        {% if data.max_test %}
                        <div class="result-card best">
                            <div class="result-header"><div class="result-icon best">👍</div><div class="result-title">Best Scenario</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Config (C, γ):</span><span class="metric-val">{{ data.max_test.n_hidden }}</span></li>
                                <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}%</span></li>
                                <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}%</span></li>
                                <li><span class="metric-name">Train Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_train) }}s</span></li>
                                <li><span class="metric-name">Test Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_test) }}s</span></li>
                            </ul>
                            {% if act_cm.get(act_name) and act_cm[act_name].best %}<div class="cm-container"><img class="cm-image" src="{{ act_cm[act_name].best }}" alt="CM Best"></div>{% endif %}
                        </div>
                        {% endif %}

                        {% if data.min_test %}
                        <div class="result-card worst">
                            <div class="result-header"><div class="result-icon worst">💎</div><div class="result-title">Worst Scenario</div></div>
                            <ul class="metrics-list">
                                <li><span class="metric-name">Config (C, γ):</span><span class="metric-val">{{ data.min_test.n_hidden }}</span></li>
                                <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}%</span></li>
                                <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}%</span></li>
                                <li><span class="metric-name">Train Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_train) }}s</span></li>
                                <li><span class="metric-name">Test Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_test) }}s</span></li>
                            </ul>
                            {% if act_cm.get(act_name) and act_cm[act_name].worst %}<div class="cm-container"><img class="cm-image" src="{{ act_cm[act_name].worst }}" alt="CM Worst"></div>{% endif %}
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
            
            {% if diagnostics %}
            <div class="kernels-section" style="margin-top: 40px;">
                <h2 class="section-title">🔍 Automatic Diagnostics</h2>
                {% for kernel_name, diagnosis in diagnostics.items() %}
                <div class="diagnosis-card status-{{ diagnosis.status }}" style="background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 25px; margin-bottom: 20px; border-left: 5px solid {% if diagnosis.status == 'skipped' %}#dc3545{% elif diagnosis.status == 'critical' or diagnosis.status == 'unstable' or diagnosis.status == 'warning' or diagnosis.status == 'poor' %}#ffc107{% elif diagnosis.status == 'good' %}#28a745{% else %}#6c757d{% endif %};">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 2em; margin-right: 15px;">
                            {% if diagnosis.status == 'skipped' %}❌
                            {% elif diagnosis.status == 'critical' or diagnosis.status == 'unstable' or diagnosis.status == 'warning' or diagnosis.status == 'poor' %}⚠️
                            {% elif diagnosis.status == 'good' %}✅{% else %}❓{% endif %}
                        </div>
                        <div>
                            <h3 style="margin: 0; font-size: 1.5em; color: #333;">{{ kernel_name }}</h3>
                            {% if diagnosis.problem %}<p style="margin: 5px 0 0 0; color: #666; font-weight: 500;">{{ diagnosis.problem }}</p>{% endif %}
                        </div>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                        <strong style="color: #555;">📝 Explanation:</strong>
                        <p style="margin: 10px 0 0 0; line-height: 1.6; color: #666;">{{ diagnosis.explanation }}</p>
                    </div>
                    <div style="background: {% if diagnosis.status == 'good' %}#d4edda{% else %}#fff3cd{% endif %}; padding: 15px; border-radius: 8px; border-left: 3px solid {% if diagnosis.status == 'good' %}#28a745{% else %}#ffc107{% endif %};">
                        <strong style="color: #555;">💡 Recommendation:</strong>
                        <p style="margin: 10px 0 0 0; line-height: 1.6; color: #666;">{{ diagnosis.recommendation }}</p>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="kernels-section" style="margin-top: 40px;">
                <h2 class="section-title">📊 Guide: Understanding Deviation (±)</h2>
                <div class="stats-grid">
                    <div class="stat-card" style="border-left: 10px solid #28a745;">
                         <div class="card-header"><div class="card-icon" style="background: #28a745;">✓</div><div class="card-title">Low Deviation (< 2%)</div></div>
                         <p style="font-size: 1em; color: #555; line-height: 1.5;">
                            <b>Highly Stable.</b> The model yields consistent results regardless of how data is split. Reliable for production.
                         </p>
                    </div>
                    <div class="stat-card" style="border-left: 10px solid #ffc107;">
                         <div class="card-header"><div class="card-icon" style="background: #ffc107; color: #333;">!</div><div class="card-title">Moderate (2% - 10%)</div></div>
                         <p style="font-size: 1em; color: #555; line-height: 1.5;">
                            <b>Acceptable Variance.</b> Common in smaller datasets. Shows slight sensitivity to specific data samples.
                         </p>
                    </div>
                    <div class="stat-card" style="border-left: 10px solid #dc3545;">
                         <div class="card-header"><div class="card-icon" style="background: #dc3545;">✕</div><div class="card-title">High Deviation (> 10%)</div></div>
                         <p style="font-size: 1em; color: #555; line-height: 1.5;">
                            <b>Unstable.</b> The model's performance fluctuates wildly. It may be overfitting to specific chunks of data.
                         </p>
                    </div>
                </div>
            </div>

            <div class="kernels-section" style="margin-top: 40px; border-top: 5px solid #d63384;">
                <h2 class="section-title" style="color: #d63384;">🛡️ Data Leakage Report</h2>
                <p class="subtitle" style="margin-bottom: 20px;">Automatic analysis of suspicious features (Correlation ~1.0).</p>
                
                <div class="stats-grid">
                    <div class="stat-card" style="border-left: 10px solid {% if dirty_count > 0 %}#dc3545{% else %}#28a745{% endif %};">
                         <div class="card-header">
                            <div class="card-icon" style="background: {% if dirty_count > 0 %}#dc3545{% else %}#28a745{% endif %};">{% if dirty_count > 0 %}🗑️{% else %}🛡️{% endif %}</div>
                            <div class="card-title">{{ dirty_count }} Features Removed</div>
                         </div>
                         <p style="font-size: 1em; color: #555; line-height: 1.5;">
                            {% if dirty_count > 0 %}
                            Features with perfect correlation to the target were identified and <b>automatically removed</b> from the dataset before training.
                            {% else %}
                            No suspicious features were found. The dataset appears free of obvious data leakage.
                            {% endif %}
                         </p>
                    </div>

                    <div class="stat-card" style="border-left: 10px solid #6c757d;">
                         <div class="card-header"><div class="card-icon" style="background: #6c757d;">ℹ️</div><div class="card-title">Why this matters?</div></div>
                         <p style="font-size: 1em; color: #555; line-height: 1.5;">
                            If a feature has correlation 1.0, the model "cheats" by using only that feature, ignoring the rest. 
                            By removing it (the "Dirty List"), we force the SVM to learn real patterns, providing a realistic accuracy score instead of a fake 100%.
                         </p>
                    </div>
                </div>
            </div>

        </div>
    </body>
    </html>
    """

    template = Template(html_template)
    rendered = template.render(
        global_results=global_results, act_results=act_results,
        cm_global_best=cm_global_best, cm_global_worst=cm_global_worst,
        act_cm=act_cm, diagnostics=diagnostics_dict, dirty_count=dirty_count
    )

    output_path = os.path.join(script_dir, output_file)
    with open(output_path, "w", encoding="utf-8") as f: f.write(rendered)
    print(f"HTML report generated: {output_path}")

#========================================================================
# CLI
#========================================================================

def setOpts(argv):
    parser = argparse.ArgumentParser(description='SVM Parameter Tester (v8.0 Leakage Protected)')
    parser.add_argument('-dataset', dest='dataset', default='heart_scale', help='Dataset file (LIBSVM).')
    parser.add_argument('-threshold', dest='threshold', type=float, default=None, help="Pruning threshold.")
    args = parser.parse_args(argv)
    return args.dataset, args.threshold

if __name__ == "__main__":
    dataset, threshold = setOpts(sys.argv[1:])
    experiment = svmParameters()
    glob, res, skip, dirty_cnt = experiment.main(dataset, threshold)
    generate_html_report(glob, res, skip, dirty_cnt)