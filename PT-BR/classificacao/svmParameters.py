# -*- coding: utf-8 -*-
"""
Este script realiza a avaliação de parâmetros de SVM para diferentes kernels
e gera um relatório HTML completo com os resultados globais e por kernel,
incluindo as matrizes de confusão médias para cada cenário.

"""

# Dependências para o cálculo do SVM
from libsvm.svmutil import *
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import string
from time import process_time

# Dependências para a geração do relatório HTML e gráficos
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

def plot_and_save_cm(cm, title, filename):
    """Gera um gráfico da matriz de confusão e o salva como imagem."""
    if cm is None:
        return

    # Converte para percentuais
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(6, 5))
    # Cria anotações personalizadas com símbolo de porcentagem
    annot_labels = np.array([[f'{val:.1f}%' for val in row] for row in cm_percent])
    sns.heatmap(cm_percent, annot=annot_labels, fmt='', cmap=plt.cm.Blues,
                cbar_kws={'label': 'Percentual (%)'})
    plt.title(title)
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
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
            ff.write("Original_Index,New_Index,Correlation\n")
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
    """Executa a validação cruzada k-fold e retorna métricas, incluindo as matrizes de confusão médias de treino e teste."""
    k = 10
    np.random.seed(1)
    kf = KFold(n_splits=k, shuffle=True)

    accuracies_train, accuracies_test, timing_train, timing_test = [], [], [], []

    # Listas separadas para matrizes de treino e teste
    confusion_matrices_train, confusion_matrices_test = [], []
    unique_labels = sorted(np.unique(y))

    kcount = 0
    for train_index, test_index in kf.split(x):
        kcount += 1
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

        y_train_pruned = y_train
        x_train_pruned = x_train

        param_str = f'-t {t} -c {cost} -g {gamma} -q'
        print(f'\n{str(kcount)}ª execução das {str(k)} previstas.')
        start_time_train = process_time()
        m = svm_train(y_train_pruned, x_train_pruned, param_str)
        end_time_train = process_time()

        print('Acurácia no treino.')
        # Captura os rótulos previstos do treino para a matriz de confusão
        p_label_train, p_acc_train, _ = svm_predict(y_train_pruned, x_train_pruned, m)

        print('Acurácia no teste.')
        start_time_test = process_time()
        p_label_test, p_acc_test, _ = svm_predict(y_test, x_test, m)
        end_time_test = process_time()

        accuracies_train.append(p_acc_train[0])
        accuracies_test.append(p_acc_test[0])
        timing_train.append(end_time_train - start_time_train)
        timing_test.append(end_time_test - start_time_test)

        # Calcula e armazena a matriz de confusão para o treino e teste de cada fold
        cm_train = confusion_matrix(y_train_pruned, p_label_train, labels=unique_labels)
        cm_test = confusion_matrix(y_test, p_label_test, labels=unique_labels)
        confusion_matrices_train.append(cm_train)
        confusion_matrices_test.append(cm_test)

    # Calcula a média das matrizes de treino e teste
    avg_cm_train = np.mean(confusion_matrices_train, axis=0)
    avg_cm_test = np.mean(confusion_matrices_test, axis=0)

    # Retorna ambas as matrizes de confusão
    return np.mean(accuracies_train), np.std(accuracies_train), \
           np.mean(accuracies_test), np.std(accuracies_test), \
           np.mean(timing_train), np.std(timing_train), \
           np.mean(timing_test), np.std(timing_test), \
           avg_cm_train, avg_cm_test

class svmParameters():
    def main(self, dataset, threshold):
        """Método principal que executa a busca de parâmetros e retorna os resultados."""
        y, x = svm_read_problem(dataset)
        if threshold:
            y, x = pruningDataset(y, x, threshold, save_files=True)

        cost_vector = [1, 1000]
        gamma_vector = [1]

        min_acc, max_acc = 101, -1
        max_cm, min_cm = None, None
        min_kernel, max_kernel = -1, -1
        min_cost, max_cost, min_gamma, max_gamma = 0, 0, 0, 0
        min_mean_accuracy_train, min_std_accuracy_train, min_mean_accuracy_test, min_std_accuracy_test = 101, 101, 101, 101
        max_mean_accuracy_train, max_std_accuracy_train, max_mean_accuracy_test, max_std_accuracy_test = -1, -1, -1, -1

        min_mean_time_train, min_std_time_train, min_mean_time_test, min_std_time_test = 101, 101, 101, 101
        max_mean_time_train, max_std_time_train, max_mean_time_test, max_std_time_test = -1, -1, -1, -1
        kernel_results = {}

        for t in range(4):
            kernel_max_acc_test, kernel_min_acc_test = -1, 101
            kernel_max_data_test, kernel_min_data_test = {}, {}

            kernel_max_acc_train, kernel_min_acc_train = -1, 101
            kernel_max_data_train, kernel_min_data_train = {}, {}

            for c in cost_vector:
                for g in gamma_vector:
                    if (t != 0):
                        print(f'\n========== Experimentando Kernel: {kernel_str(t)}, peso de custo: {c}, peso de curvatura: {g} ==========')
                    else:
                        print(f'\n========== Experimentando Kernel: {kernel_str(t)}, peso de custo: {c} ==========')

                    # Captura as duas matrizes de confusão retornadas
                    mean_train, std_train, mean_test, std_test, mean_time_train, std_time_train, mean_time_test, std_time_test, avg_cm_train, avg_cm_test = \
                        svmKfold(y, x, t, c, g, threshold)

                    # Dicionário agora armazena ambas as matrizes
                    current_result_data = {
                        "accuracy_train": mean_train, "std_train": std_train,
                        "accuracy_test": mean_test, "std_test": std_test,
                        "time_train": mean_time_train, "std_time_train": std_time_train,
                        "time_test": mean_time_test, "std_time_test": std_time_test,
                        "cost": c, "gamma": g,
                        "confusion_matrix_train": avg_cm_train,
                        "confusion_matrix_test": avg_cm_test
                    }

                    # Usa .copy() para garantir que uma cópia dos dados seja salva, e não uma referência.
                    if mean_test > kernel_max_acc_test:
                        kernel_max_acc_test = mean_test
                        kernel_max_data_test = current_result_data.copy()

                    if mean_test < kernel_min_acc_test:
                        kernel_min_acc_test = mean_test
                        kernel_min_data_test = current_result_data.copy()

                    if mean_train > kernel_max_acc_train:
                        kernel_max_acc_train = mean_train
                        kernel_max_data_train = current_result_data.copy()

                    if mean_train < kernel_min_acc_train:
                        kernel_min_acc_train = mean_train
                        kernel_min_data_train = current_result_data.copy()

                    # Lógica para resultados globais (baseada em acurácia de teste)
                    if mean_test > max_acc:
                        max_acc = mean_test
                        max_kernel, max_cost, max_gamma = t, c, g
                        max_mean_accuracy_train, max_std_accuracy_train = mean_train, std_train
                        max_mean_accuracy_test, max_std_accuracy_test = mean_test, std_test
                        max_mean_time_train, max_std_time_train = mean_time_train, std_time_train
                        max_mean_time_test, max_std_time_test = mean_time_test, std_time_test
                        max_cm = avg_cm_test # Global é baseado em teste

                    if mean_test < min_acc:
                        min_acc = mean_test
                        min_kernel, min_cost, min_gamma = t, c, g
                        min_mean_accuracy_train, min_std_accuracy_train = mean_train, std_train
                        min_mean_accuracy_test, min_std_accuracy_test = mean_test, std_test
                        min_mean_time_train, min_std_time_train = mean_time_train, std_time_train
                        min_mean_time_test, min_std_time_test = mean_time_test, std_time_test
                        min_cm = avg_cm_test # Global é baseado em teste

            kernel_results[t] = {
                "max_test": kernel_max_data_test, "min_test": kernel_min_data_test,
                "max_train": kernel_max_data_train, "min_train": kernel_min_data_train
            }

        global_results = {
            "max": {"accuracy_train": max_mean_accuracy_train, "std_train": max_std_accuracy_train,
                    "accuracy_test": max_mean_accuracy_test, "std_test": max_std_accuracy_test,
                    "time_train": max_mean_time_train, "std_time_train": max_std_time_train,
                    "time_test": max_mean_time_test, "std_time_test": max_std_time_test,
                    "kernel_id": max_kernel, "cost": max_cost, "gamma": max_gamma, "confusion_matrix": max_cm},
            "min": {"accuracy_train": min_mean_accuracy_train, "std_train": min_std_accuracy_train,
                    "accuracy_test": min_mean_accuracy_test, "std_test": min_std_accuracy_test,
                    "time_train": min_mean_time_train, "std_time_train": min_std_time_train,
                    "time_test": min_mean_time_test, "std_time_test": min_std_time_test,
                    "kernel_id": min_kernel, "cost": min_cost, "gamma": min_gamma, "confusion_matrix": min_cm}
        }

        print(f'\n==========================================')
        print(f'RESULTADOS GLOBAIS')
        print(f'==========================================')
        print(f"Melhor Acurácia de Teste: {max_mean_accuracy_test:.2f}% (Kernel: {kernel_str(max_kernel)})")
        print(f"Pior Acurácia de Teste: {min_mean_accuracy_test:.2f}% (Kernel: {kernel_str(min_kernel)})")

        return global_results, kernel_results

def generate_html_report(global_results, kernel_results, output_file='svm_report.html'):
    """Gera um relatório HTML com a tabela de resumo por kernel simplificada."""
    img_dir = 'svm_report_images'
    os.makedirs(img_dir, exist_ok=True)

    # Gera apenas as imagens de matriz de confusão que serão exibidas no relatório.
    plot_and_save_cm(global_results['max']['confusion_matrix'], 'MC Média - Melhor Desempenho Global (Teste)', f'{img_dir}/cm_global_best.png')
    plot_and_save_cm(global_results['min']['confusion_matrix'], 'MC Média - Pior Desempenho Global (Teste)', f'{img_dir}/cm_global_worst.png')

    for kernel_id, data in kernel_results.items():
        k_name = kernel_str(kernel_id).replace(" ", "_")
        if data.get('max_test'):
            plot_and_save_cm(data['max_test']['confusion_matrix_test'], f'MC Média - Melhor Teste {kernel_str(kernel_id)}', f'{img_dir}/cm_kernel_{k_name}_best_test.png')
        if data.get('min_test'):
            plot_and_save_cm(data['min_test']['confusion_matrix_test'], f'MC Média - Pior Teste {kernel_str(kernel_id)}', f'{img_dir}/cm_kernel_{k_name}_worst_test.png')

    # --- TEMPLATE HTML COM O CSS ANTERIOR E A NOVA TABELA ---
    html_template = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório de Resultados SVM</title>
        <style>

            body { font-family: Arial, sans-serif; margin: 20px; background-color: #fdfdfd; }
            .container { max-width: 1200px; margin: 0 auto; background-color: #fff; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 25px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 12px; text-align: left; vertical-align: middle; }
            .metrics-table th { background-color: #0056b3; color: white; text-align: center; }
            .result-section { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 5px solid #0056b3; }
            h1, h2 { color: #333; border-bottom: 2px solid #0056b3; padding-bottom: 10px; }
            h3 { color: #555; margin-top: 10px; }
            .subtitle { font-size: 1.1em; color: #666; font-style: italic; margin-top: -5px; margin-bottom: 20px; }
            .best-result { background-color: #e7f3e7; }
            .worst-result { background-color: #fdeeee; }
            .cm-image { max-width: 350px; display: block; margin: 10px auto; }
            .results-cell b { color: #000000; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Avaliação de Parâmetros SVM</h1>
            <p class="subtitle">A acurácia no teste é a métrica de escolha para os resultados.</p>

            <div class="result-section">
                <h2>Resultados Globais</h2>
                <p>Estes são os melhores e piores desempenhos gerais, definidos pela acurácia de teste em todas as combinações.</p>
                <table class="metrics-table">
                    <tr class="best-result"><th colspan="2" style="text-align:center; font-weight:bold;">Melhor Desempenho Geral</th></tr>
                    <tr class="best-result"><td><b>Configuração (C, γ)</b></td><td>({{ global_results.max.cost }}, {{ global_results.max.gamma }})</td></tr>
                    <tr class="best-result"><td><b>Melhor Kernel</b></td><td>{{ global_results.max.kernel_name }}</td></tr>
                    <tr class="best-result"><td><b>Acurácia de Treino</b></td><td>{{ "%.2f"|format(global_results.max.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_train) }}%</td></tr>
                    <tr class="best-result"><td><b>Acurácia de Teste</b></td><td>{{ "%.2f"|format(global_results.max.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.max.std_test) }}%</td></tr>
                    <tr class="best-result"><td><b>Tempo Médio de Treino</b></td><td>{{ "%.4f"|format(global_results.max.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_train) }}s</td></tr>
                    <tr class="best-result"><td><b>Tempo Médio de Teste</b></td><td>{{ "%.4f"|format(global_results.max.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.max.std_time_test) }}s</td></tr>
                    <tr class="best-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_best.png" alt="Matriz de Confusão - Melhor Global"></td></tr>

                    <tr class="worst-result"><th colspan="2" style="text-align:center; font-weight:bold;">Pior Desempenho Geral</th></tr>
                    <tr class="worst-result"><td><b>Configuração (C, γ)</b></td><td>({{ global_results.min.cost }}, {{ global_results.min.gamma }})</td></tr>
                    <tr class="worst-result"><td><b>Pior Kernel</b></td><td>{{ global_results.min.kernel_name }}</td></tr>
                    <tr class="worst-result"><td><b>Acurácia de Treino</b></td><td>{{ "%.2f"|format(global_results.min.accuracy_train) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_train) }}%</td></tr>
                    <tr class="worst-result"><td><b>Acurácia de Teste</b></td><td>{{ "%.2f"|format(global_results.min.accuracy_test) }}% &plusmn; {{ "%.2f"|format(global_results.min.std_test) }}%</td></tr>
                    <tr class="worst-result"><td><b>Tempo Médio de Treino</b></td><td>{{ "%.4f"|format(global_results.min.time_train) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_train) }}s</td></tr>
                    <tr class="worst-result"><td><b>Tempo Médio de Teste</b></td><td>{{ "%.4f"|format(global_results.min.time_test) }}s &plusmn; {{ "%.4f"|format(global_results.min.std_time_test) }}s</td></tr>
                    <tr class="worst-result"><td colspan="2"><img class="cm-image" src="svm_report_images/cm_global_worst.png" alt="Matriz de Confusão - Pior Global"></td></tr>
                </table>
            </div>

            <div class="result-section">
                <h2>Resumo por Kernel</h2>
                {% for kernel_id, data in kernel_results.items() %}
                <h3>Kernel: <strong>{{ kernel_names[kernel_id] }}</strong></h3>

                <table class="metrics-table">
                    <tr>
                        <th>Desempenho</th>
                        <th>Resultados</th>
                        <th>Matriz de Confusão Média (Teste)</th>
                    </tr>

                    {% if data.max_test %}
                    <tr class="best-result">
                        <td><b>Melhor Cenário</b></td>
                        <td class="results-cell">
                            <ul>
                                <li><b>Configuração (C, γ):</b> ({{ data.max_test.cost }}, {{ data.max_test.gamma }})<br></li>
                                <li><b>Acurácia de Teste:</b> {{ "%.2f"|format(data.max_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_test) }}%<br></li>
                                <li><b>Tempo de Teste:</b> {{ "%.4f"|format(data.max_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_test) }}s</li>
                                <li><b>Acurácia de Treino:</b> {{ "%.2f"|format(data.max_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.max_test.std_train) }}%<br></li>
                                <li><b>Tempo de Treino:</b> {{ "%.4f"|format(data.max_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.max_test.std_time_train) }}s<br></li>

                            </ul>
                        </td>
                        <td><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_best_test.png" alt="MC Melhor Teste"></td>
                    </tr>
                    {% endif %}

                    {% if data.min_test %}
                    <tr class="worst-result">
                        <td><b>Pior Cenário</b></td>
                        <td class="results-cell">
                            <ul>
                                <li><b>Configuração (C, γ):</b> ({{ data.min_test.cost }}, {{ data.min_test.gamma }})<br></li>
                                <li><b>Acurácia de Teste:</b> {{ "%.2f"|format(data.min_test.accuracy_test) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_test) }}%<br></li>
                                <li><b>Tempo de Teste:</b> {{ "%.4f"|format(data.min_test.time_test) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_test) }}s</li>
                                <li><b>Acurácia de Treino:</b> {{ "%.2f"|format(data.min_test.accuracy_train) }}% &plusmn; {{ "%.2f"|format(data.min_test.std_train) }}%<br></li>
                                <li><b>Tempo de Treino:</b> {{ "%.4f"|format(data.min_test.time_train) }}s &plusmn; {{ "%.4f"|format(data.min_test.std_time_train) }}s<br></li>

                            </ul>
                        </td>
                        <td><img class="cm-image" src="svm_report_images/cm_kernel_{{ kernel_names[kernel_id]|replace(' ', '_') }}_worst_test.png" alt="MC Pior Teste"></td>
                    </tr>
                    {% endif %}
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
    parser = argparse.ArgumentParser(description='Testador de Parâmetros SVM com Geração de Relatório HTML.')
    parser.add_argument('-dataset', dest='dataset', action='store',
                        default='heart_scale', help='Nome do arquivo do conjunto de dados (formato LIBSVM).')
    parser.add_argument('-threshold', dest='threshold', type=float, default=None,
                    help="Limiar de correlação para seleção de características (ex: 0.1). Se não fornecido, não aplica poda.")

    args = parser.parse_args(argv)
    return args.dataset, args.threshold

if __name__ == "__main__":
    dataset, threshold = setOpts(sys.argv[1:])
    experiment = svmParameters()
    global_results, kernel_results = experiment.main(dataset, threshold)
    generate_html_report(global_results, kernel_results)
