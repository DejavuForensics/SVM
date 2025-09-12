# -*- coding: utf-8 -*-
"""
This script performs an evaluation of SVM parameters for different kernels
and generates a complete HTML report with global and per-kernel results,
including the average confusion matrices for each scenario.
"""

# Dependencies for SVM calculation
from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import string
from time import process_time

# Dependencies for HTML report and graph generation
from jinja2 import Template
import os
import matplotlib.pyplot as plt
import seaborn as sns

#========================================================================
# FUNCTIONS AND CLASSES FOR THE SVM EXPERIMENT
#========================================================================

def kernel_str(t):
    """Converts the kernel ID to its string name."""
    if (t == 0):
        return 'Linear'
    elif (t == 1):
        return 'Polynomial'
    elif (t == 2):
        return 'Radial Basis Function'
    elif (t == 3):
        return 'Sigmoid'
    return 'Unknown'

def plot_and_save_cm(cm, title, filename):
    """Generates a confusion matrix plot and saves it as an image."""
    if cm is None or cm.sum() == 0:
        return

    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm_sum * 100
    cm_percent = np.nan_to_num(cm_percent)

    plt.figure(figsize=(6, 5))
    annot_labels = np.array([[f'{val:.1f}%' for val in row] for row in cm_percent])
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap=plt.cm.Blues,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_libsvm(y, x, filename):
    """Saves data in LIBSVM format."""
    with open(filename, 'w') as f:
        for label, features in zip(y, x):
            sorted_features = sorted(features.items(), key=lambda item: item[0])
            feature_str = ' '.join(f"{k}:{v}" for k, v in sorted_features)
            f.write(f"{int(label)} {feature_str}\n")

def pruningDataset(y, x, threshold, save_files=True):
    """Filters features based on a correlation threshold."""
    if not x or not y:
        return y, x

    max_feature = 0
    for sample in x:
        if sample:
            max_feature = max(max_feature, max(sample.keys()))

    x_array = np.zeros((len(x), max_feature))
    for i, sample in enumerate(x):
        for feature, value in sample.items():
            if feature > 0: # Ensure feature index is valid
                x_array[i, feature-1] = value

    correlations = np.zeros(x_array.shape[1])
    for i in range(x_array.shape[1]):
        if np.std(x_array[:, i]) > 0:
            correlations[i] = np.corrcoef(x_array[:, i], y)[0, 1]
        else:
            correlations[i] = 0

    selected_indices = np.where(np.abs(correlations) >= threshold)[0]

    if save_files:
        with open('selected_features.csv', 'w') as ff:
            ff.write("Original_Index,New_Index,Correlation\n")
            for new_idx, old_idx in enumerate(selected_indices, 1):
                ff.write(f"{old_idx+1},{new_idx},{correlations[old_idx]:.6f}\n")

    feature_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices, 1)}
    x_pruned = []
    for sample in x:
        pruned_sample = {feature_mapping.get(k-1): v for k, v in sample.items() if (k-1) in selected_indices}
        # Filter out None keys that may result from mapping
        pruned_sample = {k: v for k, v in pruned_sample.items() if k is not None}
        x_pruned.append(pruned_sample)


    if save_files:
        save_libsvm(y, x_pruned, 'globalPruned.csv')

    return (y, x_pruned)

def svmKfold(y, x, t, cost, gamma):
    """Performs k-fold cross-validation and returns metrics."""
    k = 10
    np.random.seed(1)
    kf = KFold(n_splits=k, shuffle=True)
    accuracies_train, accuracies_test, timing_train, timing_test = [], [], [], []
    confusion_matrices_train, confusion_matrices_test = [], []
    unique_labels = sorted(np.unique(y))
    kcount = 0
    for train_index, test_index in kf.split(x):
        kcount += 1
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        param_str = f'-t {t} -c {cost} -g {gamma} -q'
        print(f'\nRunning fold {kcount}/{k}...')
        start_time_train = process_time()
        m = svm_train(y_train, x_train, param_str)
        end_time_train = process_time()
        p_label_train, p_acc_train, _ = svm_predict(y_train, x_train, m)
        start_time_test = process_time()
        p_label_test, p_acc_test, _ = svm_predict(y_test, x_test, m)
        end_time_test = process_time()
        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])
        timing_train.append(end_time_train - start_time_train)
        timing_test.append(end_time_test - start_time_test)
        cm_train = confusion_matrix(y_train, p_label_train, labels=unique_labels)
        cm_test = confusion_matrix(y_test, p_label_test, labels=unique_labels)
        confusion_matrices_train.append(cm_train)
        confusion_matrices_test.append(cm_test)
    avg_cm_train = np.mean(confusion_matrices_train, axis=0)
    avg_cm_test = np.mean(confusion_matrices_test, axis=0)
    return np.mean(accuracies_train), np.std(accuracies_train), \
           np.mean(accuracies_test), np.std(accuracies_test), \
           np.mean(timing_train), np.std(timing_train), \
           np.mean(timing_test), np.std(timing_test), \
           avg_cm_train, avg_cm_test

class svmParameters():
    def main(self, dataset, threshold):
        """Main method that runs the parameter search."""
        y, x = svm_read_problem(dataset)
        if threshold is not None:
            print(f"Applying feature pruning with threshold: {threshold}")
            y, x = pruningDataset(y, x, threshold, save_files=True)

        cost_vector = [1, 1000]
        gamma_vector = [1]

        # ... (variable initializations)
        min_acc, max_acc = 101, -1; max_cm, min_cm = None, None; min_kernel, max_kernel = -1, -1
        min_cost, max_cost, min_gamma, max_gamma = 0, 0, 0, 0
        min_mean_accuracy_train, min_std_accuracy_train, min_mean_accuracy_test, min_std_accuracy_test = 101, 101, 101, 101
        max_mean_accuracy_train, max_std_accuracy_train, max_mean_accuracy_test, max_std_accuracy_test = -1, -1, -1, -1
        min_mean_time_train, min_std_time_train, min_mean_time_test, min_std_time_test = 101, 101, 101, 101
        max_mean_time_train, max_std_time_train, max_mean_time_test, max_std_time_test = -1, -1, -1, -1
        kernel_results = {}

        for t in range(4):
            all_runs_for_kernel = []
            gammas_to_test = [0] if t == 0 else gamma_vector
            for c in cost_vector:
                for g in gammas_to_test:
                    print(f'\n========== Testing Kernel: {kernel_str(t)}, Cost: {c}, Gamma: {g} ==========')
                    mean_train, std_train, mean_test, std_test, mean_time_train, std_time_train, mean_time_test, std_time_test, avg_cm_train, avg_cm_test = \
                        svmKfold(y, x, t, c, g)
                    current_result_data = {
                        "accuracy_train": mean_train, "std_train": std_train, "accuracy_test": mean_test, "std_test": std_test,
                        "time_train": mean_time_train, "std_time_train": std_time_train, "time_test": mean_time_test, "std_time_test": std_time_test,
                        "cost": c, "gamma": g, "confusion_matrix_train": avg_cm_train, "confusion_matrix_test": avg_cm_test
                    }
                    all_runs_for_kernel.append(current_result_data)
                    if mean_test > max_acc:
                        max_acc = mean_test; max_kernel, max_cost, max_gamma = t, c, g
                        max_mean_accuracy_train, max_std_accuracy_train = mean_train, std_train
                        max_mean_accuracy_test, max_std_accuracy_test = mean_test, std_test
                        max_mean_time_train, max_std_time_train = mean_time_train, std_time_train
                        max_mean_time_test, max_std_time_test = mean_time_test, std_time_test
                        max_cm = avg_cm_test
                    if mean_test < min_acc:
                        min_acc = mean_test; min_kernel, min_cost, min_gamma = t, c, g
                        min_mean_accuracy_train, min_std_accuracy_train = mean_train, std_train
                        min_mean_accuracy_test, min_std_accuracy_test = mean_test, std_test
                        min_mean_time_train, min_std_time_train = mean_time_train, std_time_train
                        min_mean_time_test, min_std_time_test = mean_time_test, std_time_test
                        min_cm = avg_cm_test

            if all_runs_for_kernel:
                kernel_results[t] = {
                    "max_test": max(all_runs_for_kernel, key=lambda r: r['accuracy_test']),
                    "min_test": min(all_runs_for_kernel, key=lambda r: r['accuracy_test'])
                }
            else:
                kernel_results[t] = {}

        global_results = {
            "max": {"accuracy_train": max_mean_accuracy_train, "std_train": max_std_accuracy_train, "accuracy_test": max_mean_accuracy_test, "std_test": max_std_accuracy_test, "time_train": max_mean_time_train, "std_time_train": max_std_time_train, "time_test": max_mean_time_test, "std_time_test": max_std_time_test, "kernel_id": max_kernel, "cost": max_cost, "gamma": max_gamma, "confusion_matrix": max_cm},
            "min": {"accuracy_train": min_mean_accuracy_train, "std_train": min_std_accuracy_train, "accuracy_test": min_mean_accuracy_test, "std_test": min_std_accuracy_test, "time_train": min_mean_time_train, "std_time_train": min_std_time_train, "time_test": min_mean_time_test, "std_time_test": min_std_time_test, "kernel_id": min_kernel, "cost": min_cost, "gamma": min_gamma, "confusion_matrix": min_cm}
        }

        print(f'\n==========================================\nOVERALL RESULTS\n==========================================')
        print(f"Best Test Accuracy: {max_mean_accuracy_test:.2f}% (Kernel: {kernel_str(max_kernel)})")
        print(f"Worst Test Accuracy: {min_mean_accuracy_test:.2f}% (Kernel: {kernel_str(min_kernel)})")
        return global_results, kernel_results

def generate_html_report(global_results, kernel_results, output_file='svm_report.html'):
    """Generates an HTML report from the SVM parameter evaluation results."""
    img_dir = 'svm_report_images'
    os.makedirs(img_dir, exist_ok=True)

    plot_and_save_cm(global_results['max']['confusion_matrix'], 'Average CM - Best Overall Performance (Test)', f'{img_dir}/cm_global_best.png')
    plot_and_save_cm(global_results['min']['confusion_matrix'], 'Average CM - Worst Overall Performance (Test)', f'{img_dir}/cm_global_worst.png')
    for kernel_id, data in kernel_results.items():
        k_name = kernel_str(kernel_id).replace(" ", "_")
        if data.get('max_test'):
            plot_and_save_cm(data['max_test']['confusion_matrix_test'], f'Average CM - Best Test ({kernel_str(kernel_id)})', f'{img_dir}/cm_kernel_{k_name}_best_test.png')
        if data.get('min_test'):
            plot_and_save_cm(data['min_test']['confusion_matrix_test'], f'Average CM - Worst Test ({kernel_str(kernel_id)})', f'{img_dir}/cm_kernel_{k_name}_worst_test.png')

    html_template = """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Dashboard - Results Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif, Arial;
            background: linear-gradient(135deg, #8B1538 0%, #A91E4A 50%, #6B1429 100%);
            min-height: 100vh;
            color: #333;
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header-content {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .logo-ufpe {
            height: 150px;
            width: auto;
        }

        .header-text {
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #8B1538, #A91E4A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .subtitle {
            font-size: 1.2em;
            color: #666;
            font-weight: 300;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }

        .stat-card.best {
            border-left: 10px solid #A5D7A7;
            background: rgba(255, 255, 255, 0.98);
        }

        .stat-card.worst {
            border-left: 10px solid #f9a19a;
            background: rgba(255, 255, 255, 0.98);
        }

        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .card-icon {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 1.5em;
            font-weight: bold;
            color: white;
        }

        .card-icon.best {
            background: linear-gradient(45deg, #4CAF50, #66BB6A);
        }

        .card-icon.worst {
            background: linear-gradient(45deg, #f44336, #EF5350);
        }

        .card-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        }

        .metric-row:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-weight: 500;
            color: #555;
        }

        .metric-value.worst {
            font-weight: 600;
            color: #333;
            padding: 4px 12px;
            background: rgba(139, 21, 56, 0.1);
            border-radius: 20px;
        }

        .metric-value.best {
            font-weight: 600;
            color: #333;
            padding: 4px 12px;
            background: rgba(76, 175, 80, 0.1);
            border-radius: 20px;
        }

        .cm-container {
            text-align: center;
            margin-top: 20px;
        }

        .cm-image {
            max-width: 70%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .kernels-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

       .section-title {
            font-size: 2em;
            margin-bottom: 30px;
            text-align: center;
            background: linear-gradient(45deg, #8B1538, #A91E4A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .kernel-group {
            margin-bottom: 40px;
        }

        .kernel-title {
            font-size: 1.5em;
            font-weight: 600;
            margin-bottom: 20px;
            padding: 15px 20px;
            color: white;
            border-radius: 10px;
            text-align: center;
            background: linear-gradient(45deg, #8B1538, #A91E4A);
        }

        .kernel-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .result-card:hover {
            transform: translateY(-3px);
        }

        .result-card.best {
            border: 2px solid #4CAF50;
        }

        .result-card.worst {
            border: 2px solid #f44336;
        }

        .result-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }

        .result-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
            color: white;
            font-weight: bold;
        }

        .result-icon.best {
            background: #4CAF20;
        }

        .result-icon.worst {
            background: #f44336;
        }

        .result-title {
            font-size: 1.2em;
            font-weight: 600;
        }

        .metrics-list {
            list-style: none;
            margin-bottom: 20px;
        }

        .metrics-list li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metrics-list li:last-child {
            border-bottom: none;
        }

        .metric-name {
            font-weight: 500;
            color: #555;
        }

        .metric-val {
            font-weight: 600;
            color: #333;
        }

        @media (max-width: 768px) {
            .dashboard-container {
                padding: 10px;
            }

            .header h1 {
                font-size: 2em;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }

            .kernel-results {
                grid-template-columns: 1fr;
            }

            .result-card {
                min-width: auto;
            }

            .header-content {
                flex-direction: column;
                text-align: center;
            }

            .header-text {
                text-align: center;
            }

            .logo-ufpe {
                height: 60px;
            }
        }


        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .stat-card, .kernels-section {
            animation: fadeInUp 0.6s ease forwards;
        }

        .stat-card:nth-child(1) { animation-delay: 0.1s; }
        .stat-card:nth-child(2) { animation-delay: 0.2s; }
        .kernels-section { animation-delay: 0.3s; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
			<img src="../../Brasão-Extenso---PNG---RGB.png" alt="Logo UFPE" class="logo-ufpe">
            <h1>SVM - Parameter Evaluation</h1>
            <p class="subtitle">Test accuracy is the metric of choice for the results</p>
        </div>
        <div class="stats-grid">
            <div class="stat-card best">
                <div class="card-header"><div class="card-icon best">🏆</div><div class="card-title">Best Overall Performance</div></div>
                <div class="metric-row"><span class="metric-label">Configuration (C, γ)</span><span class="metric-value best">({{ global_results.max.cost }}, {{ global_results.max.gamma }})</span></div>
                <div class="metric-row"><span class="metric-label">Best Kernel</span><span class="metric-value best">{{ global_results.max.kernel_name }}</span></div>
                <div class="metric-row"><span class="metric-label">Train Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}%</span></div>
                <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value best">{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</span></div>
                <div class="metric-row"><span class="metric-label">Training Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_train) }}s</span></div>
                <div class="metric-row"><span class="metric-label">Testing Time</span><span class="metric-value best">{{ "%.4f"|format(global_results.max.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_test) }}s</span></div>
                <div class="cm-container"><img class="cm-image" src="svm_report_images/cm_global_best.png" alt="Confusion Matrix - Best Overall"></div>
            </div>
            <div class="stat-card worst">
                <div class="card-header"><div class="card-icon worst">📉</div><div class="card-title">Worst Overall Performance</div></div>
                <div class="metric-row"><span class="metric-label">Configuration (C, γ)</span><span class="metric-value worst">({{ global_results.min.cost }}, {{ global_results.min.gamma }})</span></div>
                <div class="metric-row"><span class="metric-label">Worst Kernel</span><span class="metric-value worst">{{ global_results.min.kernel_name }}</span></div>
                <div class="metric-row"><span class="metric-label">Train Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}%</span></div>
                <div class="metric-row"><span class="metric-label">Test Accuracy</span><span class="metric-value worst">{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</span></div>
                <div class="metric-row"><span class="metric-label">Training Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_train) }}s</span></div>
                <div class="metric-row"><span class="metric-label">Testing Time</span><span class="metric-value worst">{{ "%.4f"|format(global_results.min.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_test) }}s</span></div>
                <div class="cm-container"><img class="cm-image" src="svm_report_images/cm_global_worst.png" alt="Confusion Matrix - Worst Overall"></div>
            </div>
        </div>
        <div class="kernels-section">
            <h2 class="section-title">Summary by Kernel</h2>
            {% for kernel_id, data in kernel_results.items() %}
            <div class="kernel-group">
                <div class="kernel-title">Kernel: {{ kernel_names[kernel_id] }}</div>
                <div class="kernel-results">
                    {% if data.max_test %}<div class="result-card best">
                        <div class="result-header"><div class="result-icon best">👍</div><div class="result-title">Best Case Scenario</div></div>
                        <ul class="metrics-list">
                            <li><span class="metric-name">Configuration (C, γ):</span><span class="metric-val">({{ data.max_test.cost }}, {{ data.max_test.gamma }})</span></li>
                            <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}%</span></li>
                            <li><span class="metric-name">Testing Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_test) }}s</span></li>
                            <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}%</span></li>
                            <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_train) }}s</span></li>
                        </ul>
                        <div class="cm-container"><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_best_test.png" alt="CM Best Test"></div>
                    </div>{% endif %}
                    {% if data.min_test %}<div class="result-card worst">
                        <div class="result-header"><div class="result-icon worst">👎</div><div class="result-title">Worst Case Scenario</div></div>
                        <ul class="metrics-list">
                            <li><span class="metric-name">Configuration (C, γ):</span><span class="metric-val">({{ data.min_test.cost }}, {{ data.min_test.gamma }})</span></li>
                            <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}%</span></li>
                            <li><span class="metric-name">Testing Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_test) }}s</span></li>
                            <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}%</span></li>
                            <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_train) }}s</span></li>
                        </ul>
                        <div class="cm-container"><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_worst_test.png" alt="CM Worst Test"></div>
                    </div>{% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
    """

    global_results['max']['kernel_name'] = kernel_str(global_results['max']['kernel_id'])
    global_results['min']['kernel_name'] = kernel_str(global_results['min']['kernel_id'])
    kernel_names = {k: kernel_str(k) for k in kernel_results.keys()}

    template = Template(html_template)
    html_content = template.render(
        global_results=global_results,
        kernel_results=kernel_results,
        kernel_names=kernel_names
    )

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nHTML report generated successfully: '{output_file}'")
    except IOError as e:
        print(f"\nError saving HTML report: {e}")

def setOpts(argv):
	"""Configures and parses the command-line arguments."""
    parser = argparse.ArgumentParser(description='SVM Parameter Tester with HTML Report Generation.')
    parser.add_argument('-dataset', dest='dataset', action='store',
                        default='heart_scale', help='Dataset file name (LIBSVM format).')
    parser.add_argument('-threshold', dest='threshold', type=float, default=None,
                    help="Correlation threshold for feature selection (e.g., 0.1). If not provided, no pruning is applied.")

    args = parser.parse_args(argv)
    return args.dataset, args.threshold

if __name__ == "__main__":
    dataset, threshold = setOpts(sys.argv[1:])
    experiment = svmParameters()
    global_results, kernel_results = experiment.main(dataset, threshold)
    generate_html_report(global_results, kernel_results)
