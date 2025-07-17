# -*- coding: utf-8 -*-
"""
Este script realiza a avaliação de parâmetros de SVM para diferentes kernels
e gera um relatório HTML completo com os resultados globais e por kernel,
incluindo as matrizes de confusão médias.
"""

# Dependências para o cálculo do SVM
from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import string

# Dependências para a geração do relatório HTML e gráficos
from jinja2 import Template
import os
import matplotlib.pyplot as plt
import seaborn as sns

#========================================================================
# FUNÇÕES E CLASSES PARA O EXPERIMENTO SVM
import sys,string
from time import process_time
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

def plot_and_save_cm(cm, title, filename):
    """Gera um gráfico da matriz de confusão e o salva como imagem."""
    if cm is None:
        return
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=plt.cm.Blues)
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_libsvm(y, x, filename):
    """Salva os dados no formato LIBSVM."""
    with open(filename, 'w') as f:
        for label, features in zip(y, x):
            sorted_features = sorted(features.items(), key=lambda item: item[0])
            feature_str = ' '.join(f"{k}:{v}" for k, v in sorted_features)
            f.write(f"{int(label)} {feature_str}\n")

def pruningDataset(y, x, threshold, save_files=True):
    """Filtra características com base em um limiar de correlação."""
    if not x or not y:
        return y, x

    max_feature = 0
    for sample in x:
        if sample:
            max_feature = max(max_feature, max(sample.keys()))
    
    x_array = np.zeros((len(x), max_feature))
    for i, sample in enumerate(x):
        for feature, value in sample.items():
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
            ff.write("Original_Index,Selected_Index,Correlation\n")
            for new_idx, old_idx in enumerate(selected_indices, 1):
                ff.write(f"{old_idx+1},{new_idx},{correlations[old_idx]:.6f}\n")

    feature_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices, 1)}
    x_pruned = []
    for sample in x:
        pruned_sample = {feature_mapping[k-1]: v for k, v in sample.items() if (k-1) in selected_indices}
        x_pruned.append(pruned_sample)

    if save_files:
        save_libsvm(y, x_pruned, 'globalPruned.csv')

    return (y, x_pruned)

def svmKfold(y, x, t, cost, gamma, threshold):
    """Executa a validação cruzada k-fold e retorna métricas, incluindo a matriz de confusão média."""
    k = 10
    np.random.seed(1)
    kf = KFold(n_splits=k, shuffle=True)
    
    accuracies_train, accuracies_test = [], []
    confusion_matrices = []
    unique_labels = sorted(np.unique(y))

    for train_index, test_index in kf.split(x):
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        y_train_pruned, x_train_pruned = pruningDataset(y_train, x_train, threshold, save_files=False)
        
        param_str = f'-t {t} -c {cost} -g {gamma} -q'
        m = svm_train(y_train_pruned, x_train_pruned, param_str)
        
        _, p_acc_train, _ = svm_predict(y_train_pruned, x_train_pruned, m)
        p_label_test, p_acc_test, _ = svm_predict(y_test, x_test, m)
        
        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])
        
        cm = confusion_matrix(y_test, p_label_test, labels=unique_labels)
        confusion_matrices.append(cm)

    avg_cm = np.mean(confusion_matrices, axis=0)
    
    return np.mean(accuracies_train), np.std(accuracies_train), \
           np.mean(accuracies_test), np.std(accuracies_test), \
           avg_cm

class svmParameters():
    def main(self, dataset, threshold):
        """Método principal que executa a busca de parâmetros e retorna os resultados."""
        y, x = svm_read_problem(dataset)
        y, x = pruningDataset(y, x, threshold, save_files=True)

        cost_vector = [10**0, 10**3]
        gamma_vector = [10**0]

        min_acc, max_acc = 101, -1
        max_cm, min_cm = None, None
        min_kernel, max_kernel = -1, -1
        min_cost, max_cost, min_gamma, max_gamma = 0, 0, 0, 0
        min_mean_accuracy_train, min_std_accuracy_train, min_mean_accuracy_test, min_std_accuracy_test = 101, 101, 101, 101
        max_mean_accuracy_train, max_std_accuracy_train, max_mean_accuracy_test, max_std_accuracy_test = -1, -1, -1, -1
        
        kernel_results = {}

        for t in range(4):
            kernel_max_acc, kernel_min_acc = -1, 101
            kernel_max_data, kernel_min_data = {}, {}
            
            print(f'\n=== Testando Kernel: {kernel_str(t)} ===')
            for c in cost_vector:
                for g in gamma_vector:
                    mean_train, std_train, mean_test, std_test, avg_cm = \
                        svmKfold(y, x, t, c, g, threshold)
                    
                    if mean_test > kernel_max_acc:
                        kernel_max_acc = mean_test
                        kernel_max_data = {"accuracy_test": mean_test, "std_test": std_test, "confusion_matrix": avg_cm}
                    
                    if mean_test < kernel_min_acc:
                        kernel_min_acc = mean_test
                        kernel_min_data = {"accuracy_test": mean_test, "std_test": std_test, "confusion_matrix": avg_cm}

                    if mean_test > max_acc:
                        max_acc = mean_test
                        max_kernel, max_cost, max_gamma = t, c, g
                        max_mean_accuracy_train, max_std_accuracy_train = mean_train, std_train
                        max_mean_accuracy_test, max_std_accuracy_test = mean_test, std_test
                        max_cm = avg_cm

                    if mean_test < min_acc:
                        min_acc = mean_test
                        min_kernel, min_cost, min_gamma = t, c, g
                        min_mean_accuracy_train, min_std_accuracy_train = mean_train, std_train
                        min_mean_accuracy_test, min_std_accuracy_test = mean_test, std_test
                        min_cm = avg_cm
            
            kernel_results[t] = {"max": kernel_max_data, "min": kernel_min_data}

        global_results = {
            "max": {"accuracy_train": max_mean_accuracy_train, "std_train": max_std_accuracy_train,
                    "accuracy_test": max_mean_accuracy_test, "std_test": max_std_accuracy_test,
                    "kernel_id": max_kernel, "cost": max_cost, "gamma": max_gamma, "confusion_matrix": max_cm},
            "min": {"accuracy_train": min_mean_accuracy_train, "std_train": min_std_accuracy_train,
                    "accuracy_test": min_mean_accuracy_test, "std_test": min_std_accuracy_test,
                    "kernel_id": min_kernel, "cost": min_cost, "gamma": min_gamma, "confusion_matrix": min_cm}
        }
        
        print(f'\n==========================================')
        print(f'RESULTADOS GLOBAIS')
        print(f'==========================================')
        print(f"Melhor Acurácia de Teste: {max_mean_accuracy_test:.2f}% (Kernel: {kernel_str(max_kernel)})")
        print(f"Pior Acurácia de Teste: {min_mean_accuracy_test:.2f}% (Kernel: {kernel_str(min_kernel)})")

        return global_results, kernel_results

#========================================================================
# FUNÇÃO PARA GERAR O RELATÓRIO HTML
#========================================================================

def generate_html_report(global_results, kernel_results, output_file='svm_report.html'):
    """Gera um relatório HTML a partir dos resultados da avaliação de parâmetros do SVM."""
    img_dir = 'svm_report_images'
    os.makedirs(img_dir, exist_ok=True)

    # Gera e salva gráficos de Matriz de Confusão
    plot_and_save_cm(global_results['max']['confusion_matrix'], 'MC Média - Melhor Desempenho Global', f'{img_dir}/cm_global_best.png')
    plot_and_save_cm(global_results['min']['confusion_matrix'], 'MC Média - Pior Desempenho Global', f'{img_dir}/cm_global_worst.png')
    
    for kernel_id, data in kernel_results.items():
        k_name = kernel_str(kernel_id).replace(" ", "_")
        plot_and_save_cm(data['max']['confusion_matrix'], f'MC Média - Melhor {kernel_str(kernel_id)}', f'{img_dir}/cm_kernel_{k_name}_best.png')
        plot_and_save_cm(data['min']['confusion_matrix'], f'MC Média - Pior {kernel_str(kernel_id)}', f'{img_dir}/cm_kernel_{k_name}_worst.png')

    html_template = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Resultados SVM</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #fdfdfd; }
            .container { max-width: 1000px; margin: 0 auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 25px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: middle; }
            .metrics-table th { background-color: #0056b3; color: white; text-align: center; }
            .result-section { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #0056b3; }
            h1, h2 { color: #333; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
            .best-result { background-color: #e7f3e7; }
            .worst-result { background-color: #fdeeee; }
            .kernel-summary-table td:nth-child(2) { color: #28a745; font-weight: bold; }
            .kernel-summary-table td:nth-child(3) { color: #dc3545; font-weight: bold; }
            .cm-image { max-width: 350px; display: block; margin: 10px auto; }
            .cm-container { display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; }
            .cm-item { text-align: center; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Avaliação de Parâmetros SVM</h1>
            <div class="result-section">
                <h2>Resultados Globais</h2>
                <p>Estes são os melhores e piores resultados encontrados em todas as combinações de kernels e parâmetros.</p>
                <table class="metrics-table">
                    <tr class="best-result">
                        <td colspan="2" style="text-align:center; font-weight:bold;">Melhor Desempenho Geral</td>
                    </tr>
                    <tr class="best-result"><td>Melhor Kernel</td><td>{{ global_results.max.kernel_name }}</td></tr>
                    <tr class="best-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</td></tr>
                    <tr class="best-result"><td>Configuração (C, &gamma;)</td><td>({{ "%.3f"|format(global_results.max.cost) }}, {{ "%.3f"|format(global_results.max.gamma) }})</td></tr>
                    <tr class="best-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_best.png" alt="Matriz de Confusão - Melhor Global"></td></tr>
                    
                    <tr class="worst-result">
                        <td colspan="2" style="text-align:center; font-weight:bold;">Pior Desempenho Geral</td>
                    </tr>
                    <tr class="worst-result"><td>Pior Kernel</td><td>{{ global_results.min.kernel_name }}</td></tr>
                    <tr class="worst-result"><td>Acurácia Média de Teste</td><td>{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</td></tr>
                    <tr class="worst-result"><td>Configuração (C, &gamma;)</td><td>({{ "%.3f"|format(global_results.min.cost) }}, {{ "%.3f"|format(global_results.min.gamma) }})</td></tr>
                    <tr class="worst-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_worst.png" alt="Matriz de Confusão - Pior Global"></td></tr>
                </table>
            </div>
            <div class="result-section">
                <h2>Resumo por Kernel</h2>
                {% for kernel_id, data in kernel_results.items() %}
                <h3>Kernel: <strong>{{ kernel_names[kernel_id] }}</strong></h3>
                <table class="metrics-table kernel-summary-table">
                    <tr><th>Desempenho</th><th>Acurácia de Teste</th><th>Matriz de Confusão Média</th></tr>
                    <tr>
                        <td>Melhor</td>
                        <td>{{ "%.2f"|format(data.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max.std_test) }}%</td>
                        <td><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_best.png" alt="MC Melhor {{ kernel_names[kernel_id] }}"></td>
                    </tr>
                    <tr>
                        <td>Pior</td>
                        <td>{{ "%.2f"|format(data.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min.std_test) }}%</td>
                        <td><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_worst.png" alt="MC Pior {{ kernel_names[kernel_id] }}"></td>
                    </tr>
                </table>
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
        print(f"\nRelatório HTML gerado com sucesso: '{output_file}'")
    except IOError as e:
        print(f"\nErro ao salvar o relatório HTML: {e}")

#========================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
#========================================================================

def setOpts(argv):
    """Configura e parseia os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(description='Testador de Parâmetros SVM com Geração de Relatório HTML')
    parser.add_argument('-dataset', dest='dataset', action='store',
                        default='heart_scale', help='Nome do arquivo do conjunto de dados (formato LIBSVM)')
    parser.add_argument('-threshold', dest='threshold', action='store',
                        type=float, default=0.1, help='Limiar de correlação para seleção de características')
    
    args = parser.parse_args(argv)
    return args.dataset, args.threshold

if __name__ == "__main__":
    dataset, threshold = setOpts(sys.argv[1:])
    experiment = svmParameters()
    global_results, kernel_results = experiment.main(dataset, threshold)
    generate_html_report(global_results, kernel_results)