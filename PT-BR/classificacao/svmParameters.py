# -*- coding: utf-8 -*-
"""
Este script realiza uma avaliação de parâmetros SVM para diferentes kernels
e gera um relatório HTML completo com resultados globais e por kernel,
incluindo as matrizes de confusão médias para cada cenário.

Suporta tanto o formato LIBSVM quanto CSV padrão (formato mELM).
"""

# Dependências para cálculo do SVM
from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import string
from time import process_time
import pandas as pd  # Adicionado para ler CSVs

# Dependências para geração de relatório HTML e gráficos
from jinja2 import Template
import os
import matplotlib.pyplot as plt
import seaborn as sns

#========================================================================
# FUNÇÕES E CLASSES PARA O EXPERIMENTO SVM
#========================================================================

def kernel_str(t):
    """Converte o ID do kernel para seu nome em string."""
    if (t == 0):
        return 'Linear'
    elif (t == 1):
        return 'Polynomial'
    elif (t == 2):
        return 'Radial Basis Function'
    elif (t == 3):
        return 'Sigmoid'
    return 'Unknown'

def load_dataset_generic(filename):
    """
    Carrega o dataset detectando se é CSV (formato mELM) ou LIBSVM.
    
    Lógica para CSV (estilo mELM):
    - Coluna 0: Ignorada (Hash/Nome do arquivo)
    - Coluna 1: Rótulo (Target)
    - Coluna 2 em diante: Features
    """
    print(f"Carregando dataset: {filename}...")
    
    # Se for CSV, usa Pandas
    if filename.lower().endswith('.csv'):
        try:
            # sep=None e engine='python' autodetecta se é vírgula ou ponto-e-vírgula
            df = pd.read_csv(filename, sep=None, engine='python')
            print(f"CSV detectado. Formato: {df.shape}")
            
            # Tratamento de NaN (igual ao mELM)
            df = df.fillna(0)
            
            # Formato mELM padrão:
            # Col 0 = ID (ignorar)
            # Col 1 = Label (Y)
            # Col 2+ = Features (X)
            
            # Extrai Labels (Y) da segunda coluna (índice 1)
            y = df.iloc[:, 1].values.tolist()
            
            # Extrai Features (X) da terceira coluna em diante
            x_df = df.iloc[:, 2:]
            
            # Converte para lista de dicionários (formato que o libsvm exige)
            # Formato: [{1: 0.5, 3: 0.2}, {1: 0.1, ...}]
            x = []
            features_array = x_df.values
            for row in features_array:
                # Cria dicionário index:valor (índices começando de 1)
                # Otimização: libsvm aceita dict esparso
                sample = {i+1: float(val) for i, val in enumerate(row) if val != 0}
                if not sample: # Se a linha for toda zero, adiciona um dummy para não quebrar
                     sample = {1: 0.0}
                x.append(sample)
                
            print(f"CSV convertido para formato SVM com sucesso. Amostras: {len(y)}")
            return y, x
            
        except Exception as e:
            print(f"⚠️ Erro ao processar CSV com Pandas: {e}")
            print("Tentando fallback para o svm_read_problem padrão...")

    # Se não for CSV ou falhar, tenta o formato nativo do libsvm
    return svm_read_problem(filename)

def plot_and_save_cm(cm, title, filename):
    """Gera um gráfico da matriz de confusão e salva como imagem."""
    if cm is None or cm.sum() == 0:
        return

    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm_sum * 100
    cm_percent = np.nan_to_num(cm_percent)

    plt.figure(figsize=(6, 5))
    annot_labels = np.array([[f'{val:.1f}%' for val in row] for row in cm_percent])
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap=plt.cm.Blues,
                cbar_kws={'label': 'Porcentagem (%)'})
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def save_libsvm(y, x, filename):
    """Salva os dados no formato LIBSVM."""
    with open(filename, 'w') as f:
        for label, features in zip(y, x):
            sorted_features = sorted(features.items(), key=lambda item: item[0])
            feature_str = ' '.join(f"{k}:{v}" for k, v in sorted_features)
            f.write(f"{int(label)} {feature_str}\n")

def pruningDataset(y, x, threshold, save_files=True):
    """Filtra features com base em um limiar de correlação."""
    if not x or not y:
        return y, x
    
    print("Calculando correlações das features para poda (pruning)...")

    # Encontra o índice máximo da feature para lidar corretamente com dados esparsos
    max_feature = 0
    for sample in x:
        if sample:
            max_keys = max(sample.keys()) if sample.keys() else 0
            max_feature = max(max_feature, max_keys)

    # Densifica os dados para cálculo de correlação (cuidado com RAM em datasets enormes)
    # Se o dataset for muito grande, essa conversão densa simples pode falhar.
    # Para datasets mELM, geralmente funciona bem.
    n_samples = len(y)
    x_array = np.zeros((n_samples, max_feature))
    
    for i, sample in enumerate(x):
        for feature, value in sample.items():
            if feature > 0 and feature <= max_feature: 
                x_array[i, feature-1] = value

    correlations = np.zeros(x_array.shape[1])
    y_array = np.array(y)
    
    for i in range(x_array.shape[1]):
        if np.std(x_array[:, i]) > 0:
            correlations[i] = np.corrcoef(x_array[:, i], y_array)[0, 1]
        else:
            correlations[i] = 0

    selected_indices = np.where(np.abs(correlations) >= threshold)[0]
    print(f"Poda: Mantidas {len(selected_indices)} de {max_feature} features.")

    if save_files:
        with open('selected_features.csv', 'w') as ff:
            ff.write("Original_Index,New_Index,Correlation\n")
            for new_idx, old_idx in enumerate(selected_indices, 1):
                ff.write(f"{old_idx+1},{new_idx},{correlations[old_idx]:.6f}\n")

    # Mapeia índices antigos para novos índices (base 1)
    feature_mapping = {old_idx+1: new_idx for new_idx, old_idx in enumerate(selected_indices, 1)}
    
    x_pruned = []
    for sample in x:
        pruned_sample = {}
        for k, v in sample.items():
            if k in feature_mapping:
                pruned_sample[feature_mapping[k]] = v
        x_pruned.append(pruned_sample)

    if save_files:
        save_libsvm(y, x_pruned, 'globalPruned.libsvm')

    return (y, x_pruned)

def svmKfold(y, x, t, cost, gamma, k_folds):
    """Realiza validação cruzada k-fold e retorna métricas."""
    k = k_folds
    # Fixa a semente apenas para a divisão (splitting), não para o aleatório interno do SVM
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    
    accuracies_train, accuracies_test, timing_train, timing_test = [], [], [], []
    confusion_matrices_train, confusion_matrices_test = [], []
    unique_labels = sorted(list(set(y)))
    
    kcount = 0
    # Converte listas para arrays para indexação mais fácil com KFold
    y_arr = np.array(y)
    x_arr = np.array(x, dtype=object) # object porque contém dicionários
    
    for train_index, test_index in kf.split(x_arr):
        kcount += 1
        x_train = x_arr[train_index].tolist()
        x_test = x_arr[test_index].tolist()
        y_train = y_arr[train_index].tolist()
        y_test = y_arr[test_index].tolist()
        
        # -q : modo silencioso (quiet mode)
        param_str = f'-t {t} -c {cost} -g {gamma} -q'
        print(f'   [Fold {kcount}/{k}] Treinando...', end='\r')
        
        start_time_train = process_time()
        m = svm_train(y_train, x_train, param_str)
        end_time_train = process_time()
        
        # Predição no Treino
        p_label_train, p_acc_train, _ = svm_predict(y_train, x_train, m, '-q')
        
        # Predição no Teste
        start_time_test = process_time()
        p_label_test, p_acc_test, _ = svm_predict(y_test, x_test, m, '-q')
        end_time_test = process_time()
        
        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])
        timing_train.append(end_time_train - start_time_train)
        timing_test.append(end_time_test - start_time_test)
        
        cm_train = confusion_matrix(y_train, p_label_train, labels=unique_labels)
        cm_test = confusion_matrix(y_test, p_label_test, labels=unique_labels)
        confusion_matrices_train.append(cm_train)
        confusion_matrices_test.append(cm_test)
        
    print(f'   [Fold {kcount}/{k}] Concluído.     ')
    
    # Trata caso onde matrizes de confusão podem ter tamanhos diferentes se classes faltarem nos folds
    # Assumindo classificação padrão com todas as classes presentes
    try:
        avg_cm_train = np.mean(confusion_matrices_train, axis=0)
        avg_cm_test = np.mean(confusion_matrices_test, axis=0)
    except:
        avg_cm_train = None
        avg_cm_test = None

    return np.mean(accuracies_train), np.std(accuracies_train), \
           np.mean(accuracies_test), np.std(accuracies_test), \
           np.mean(timing_train), np.std(timing_train), \
           np.mean(timing_test), np.std(timing_test), \
           avg_cm_train, avg_cm_test

class svmParameters():
    def main(self, dataset, threshold, k_folds):
        """Método principal que executa a busca de parâmetros."""
        
        # USANDO A NOVA FUNÇÃO DE CARGA GENÉRICA
        y, x = load_dataset_generic(dataset)
        
        if threshold is not None:
            print(f"Aplicando poda de features com limiar: {threshold}")
            y, x = pruningDataset(y, x, threshold, save_files=True)

        # Espaço de busca reduzido para teste rápido, expanda se necessário
        # cost_vector = [1, 10, 100, 1000]
        cost_vector = [1, 1000] 
        gamma_vector = [1] # [0.1, 1]

        # Inicializa variáveis
        max_acc = -1
        max_cm, min_cm = None, None
        max_kernel, min_kernel = -1, -1
        min_acc = 101
        
        # Placeholders para resultados
        max_res = {} # Dicionário para armazenar métricas máximas
        min_res = {} # Dicionário para armazenar métricas mínimas
        
        kernel_results = {}

        # 0: Linear, 1: Polynomial, 2: RBF, 3: Sigmoid
        for t in range(4):
            all_runs_for_kernel = []
            # Kernel Linear (t=0) não usa gamma, então executa apenas uma vez por C
            gammas_to_test = [0] if t == 0 else gamma_vector
            
            for c in cost_vector:
                for g in gammas_to_test:
                    print(f'\n========== Testando Kernel: {kernel_str(t)}, Custo: {c}, Gamma: {g} ==========')
                    
                    mean_train, std_train, mean_test, std_test, \
                    mean_tt, std_tt, mean_tet, std_tet, \
                    avg_cm_train, avg_cm_test = svmKfold(y, x, t, c, g, k_folds)
                    
                    current_result_data = {
                        "accuracy_train": mean_train, "std_train": std_train, 
                        "accuracy_test": mean_test, "std_test": std_test,
                        "time_train": mean_tt, "std_time_train": std_tt, 
                        "time_test": mean_tet, "std_time_test": std_tet,
                        "cost": c, "gamma": g, 
                        "confusion_matrix_train": avg_cm_train, 
                        "confusion_matrix_test": avg_cm_test
                    }
                    all_runs_for_kernel.append(current_result_data)
                    
                    # Atualiza Máximo Global
                    if mean_test > max_acc:
                        max_acc = mean_test
                        max_kernel = t
                        max_res = current_result_data.copy()
                        max_res['kernel_id'] = t
                        
                    # Atualiza Mínimo Global
                    if mean_test < min_acc:
                        min_acc = mean_test
                        min_kernel = t
                        min_res = current_result_data.copy()
                        min_res['kernel_id'] = t

            if all_runs_for_kernel:
                kernel_results[t] = {
                    "max_test": max(all_runs_for_kernel, key=lambda r: r['accuracy_test']),
                    "min_test": min(all_runs_for_kernel, key=lambda r: r['accuracy_test'])
                }
            else:
                kernel_results[t] = {}

        # Salvaguarda se nenhum resultado for encontrado
        if not max_res:
            print("Nenhum resultado computado.")
            return None, None

        global_results = {
            "max": {
                "accuracy_train": max_res['accuracy_train'], "std_train": max_res['std_train'],
                "accuracy_test": max_res['accuracy_test'], "std_test": max_res['std_test'],
                "time_train": max_res['time_train'], "std_time_train": max_res['std_time_train'],
                "time_test": max_res['time_test'], "std_time_test": max_res['std_time_test'],
                "kernel_id": max_res['kernel_id'], "cost": max_res['cost'], 
                "gamma": max_res['gamma'], "confusion_matrix": max_res['confusion_matrix_test']
            },
            "min": {
                "accuracy_train": min_res['accuracy_train'], "std_train": min_res['std_train'],
                "accuracy_test": min_res['accuracy_test'], "std_test": min_res['std_test'],
                "time_train": min_res['time_train'], "std_time_train": min_res['std_time_train'],
                "time_test": min_res['time_test'], "std_time_test": min_res['std_time_test'],
                "kernel_id": min_res['kernel_id'], "cost": min_res['cost'], 
                "gamma": min_res['gamma'], "confusion_matrix": min_res['confusion_matrix_test']
            }
        }

        print(f'\n==========================================\nRESULTADOS GERAIS\n==========================================')
        print(f"Melhor Acurácia de Teste: {max_acc:.2f}% (Kernel: {kernel_str(max_kernel)})")
        print(f"Pior Acurácia de Teste: {min_acc:.2f}% (Kernel: {kernel_str(min_kernel)})")
        return global_results, kernel_results

def generate_html_report(global_results, kernel_results, output_file='svm_report.html'):
    """Gera um relatório HTML a partir dos resultados da avaliação de parâmetros SVM."""
    
    if global_results is None:
        return

    # Usa o diretório do script para salvar a saída
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir_name = 'svm_report_images'
    img_dir = os.path.join(script_dir, img_dir_name)
    os.makedirs(img_dir, exist_ok=True)

    def get_img_path(filename):
        return os.path.join(img_dir, filename)

    plot_and_save_cm(global_results['max']['confusion_matrix'], 'Average CM - Best Overall Performance (Test)', get_img_path('cm_global_best.png'))
    plot_and_save_cm(global_results['min']['confusion_matrix'], 'Average CM - Worst Overall Performance (Test)', get_img_path('cm_global_worst.png'))
    
    for kernel_id, data in kernel_results.items():
        k_name = kernel_str(kernel_id).replace(" ", "_")
        if data.get('max_test'):
            plot_and_save_cm(data['max_test']['confusion_matrix_test'], f'Average CM - Best Test ({kernel_str(kernel_id)})', get_img_path(f'cm_kernel_{k_name}_best_test.png'))
        if data.get('min_test'):
            plot_and_save_cm(data['min_test']['confusion_matrix_test'], f'Average CM - Worst Test ({kernel_str(kernel_id)})', get_img_path(f'cm_kernel_{k_name}_worst_test.png'))

    html_template = """
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
        .logo-ufpe { height: 150px; width: auto; }
        .header h1 { font-size: 2.5em; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }
        .subtitle { font-size: 1.2em; color: #666; font-weight: 300; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .stat-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }
        .stat-card.best { border-left: 10px solid #A5D7A7; background: rgba(255, 255, 255, 0.98); }
        .stat-card.worst { border-left: 10px solid #f9a19a; background: rgba(255, 255, 255, 0.98); }
        .card-header { display: flex; align-items: center; margin-bottom: 20px; }
        .card-icon { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 1.5em; font-weight: bold; color: white; }
        .card-icon.best { background: linear-gradient(45deg, #4CAF50, #66BB6A); }
        .card-icon.worst { background: linear-gradient(45deg, #f44336, #EF5350); }
        .card-title { font-size: 1.3em; font-weight: 600; color: #333; }
        .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); }
        .metric-value.best { font-weight: 600; padding: 4px 12px; background: rgba(76, 175, 80, 0.1); border-radius: 20px; }
        .metric-value.worst { font-weight: 600; padding: 4px 12px; background: rgba(139, 21, 56, 0.1); border-radius: 20px; }
        .cm-container { text-align: center; margin-top: 20px; }
        .cm-image { max-width: 70%; height: auto; border-radius: 5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
        .kernels-section { background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }
        .section-title { font-size: 2em; margin-bottom: 30px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .kernel-title { font-size: 1.5em; font-weight: 600; margin-bottom: 20px; padding: 15px 20px; color: white; border-radius: 10px; text-align: center; background: linear-gradient(45deg, #8B1538, #A91E4A); }
        .kernel-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }
        .result-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 25px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1); }
        .metrics-list li { padding: 8px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); display: flex; justify-content: space-between; align-items: center; }
        .result-card.best { border: 2px solid #4CAF50; } .result-card.worst { border: 2px solid #f44336; }
        .result-icon.best { background: #4CAF20; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; color: white; }
        .result-icon.worst { background: #f44336; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 12px; color: white; }
        .result-header { display: flex; align-items: center; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="header">
            <img src="../../src/ufpe_logo.png" alt="Logo UFPE" class="logo-ufpe">
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
                            <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}%</span></li>
                            <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.max_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}%</span></li>
                            <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_train) }}s</span></li>
                            <li><span class="metric-name">Testing Time:</span><span class="metric-val">{{ "%.4f"|format(data.max_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_test) }}s</span></li>
                        </ul>
                        <div class="cm-container"><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_best_test.png" alt="CM Best Test"></div>
                    </div>{% endif %}
                    {% if data.min_test %}<div class="result-card worst">
                        <div class="result-header"><div class="result-icon worst">👎</div><div class="result-title">Worst Case Scenario</div></div>
                        <ul class="metrics-list">
                            <li><span class="metric-name">Configuration (C, γ):</span><span class="metric-val">({{ data.min_test.cost }}, {{ data.min_test.gamma }})</span></li>
                            <li><span class="metric-name">Train Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}%</span></li>
                            <li><span class="metric-name">Test Accuracy:</span><span class="metric-val">{{ "%.2f"|format(data.min_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}%</span></li>
                            <li><span class="metric-name">Training Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_train) }}s</span></li>
                            <li><span class="metric-name">Testing Time:</span><span class="metric-val">{{ "%.4f"|format(data.min_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_test) }}s</span></li>
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

    output_path = os.path.join(script_dir, output_file)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nRelatório HTML gerado com sucesso: '{output_path}'")
    except IOError as e:
        print(f"\nErro ao salvar relatório HTML: {e}")

def setOpts(argv):
    """Configura e faz o parse dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description='Testador de Parâmetros SVM com Geração de Relatório HTML.')
    parser.add_argument('-dataset', dest='dataset', action='store',
                        default='heart_scale', help='Nome do arquivo do dataset (formato LIBSVM ou CSV).')
    parser.add_argument('-threshold', dest='threshold', type=float, default=None,
                    help="Limiar de correlação para seleção de features (ex: 0.1). Se não informado, nenhuma poda é aplicada.")
    parser.add_argument('-kfold', dest='kfold', type=int, default=10, 
                        help="Número de folds para Validação Cruzada (padrão: 10)")

    args = parser.parse_args(argv)
    return args.dataset, args.threshold, args.kfold

if __name__ == "__main__":
    dataset, threshold, kfold = setOpts(sys.argv[1:])
    
    # Verifica se o pandas está instalado, pois é necessário para CSV
    try:
        import pandas
    except ImportError:
        print("Erro: O dataset parece ser um CSV, mas a biblioteca 'pandas' não está instalada.")
        print("Instale usando: pip install pandas")
        sys.exit(1)

    experiment = svmParameters()
    global_results, kernel_results = experiment.main(dataset, threshold, kfold)
    generate_html_report(global_results, kernel_results)