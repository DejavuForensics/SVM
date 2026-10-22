# -*- coding: utf-8 -*-
"""
Script de Avaliação de Parâmetros SVM - relatório em português ou inglês
Funcionalidades:
- Auditoria e remoção fold-safe de features suspeitas de vazamento de dados
- Desvio Padrão Real nas Matrizes de Confusão
- Guia interpretativo sobre estabilidade e vazamento de dados
- Relatório e logs em português
- Preservação exata do layout HTML/CSS
"""

from libsvm.svmutil import svm_read_problem, svm_train, svm_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
import sys
import os
import csv
import math
import re
from datetime import datetime
from time import process_time, time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dataclasses import dataclass, field
from typing import List, Optional


BENIGN_LABEL = 0
MALICIOUS_LABEL = 1
CLASS_LABELS_ORDER = [BENIGN_LABEL, MALICIOUS_LABEL]

#========================================================================
# CONFIGURAÇÃO EXPERIMENTAL (Rodada 2)
#========================================================================

@dataclass
class ExperimentConfig:
    """Configuração experimental do svmParameters.

    Centraliza todos os parâmetros do experimento para reprodutibilidade.
    """
    dataset: str
    k_folds: int = 10
    seed: int = 1
    kernels: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    costs: List[float] = field(default_factory=lambda: [1.0, 1000.0])
    gammas: List[float] = field(default_factory=lambda: [1.0])
    cache: int = 500
    epsilon: float = 0.05
    threshold: Optional[float] = None
    leakage_threshold: float = 0.999
    max_time: float = 3600.0
    normalization: str = "none"
    demo: bool = False
    quick_test: bool = False
    force_all_kernels: bool = False
    gammas_user_provided: bool = False
    full_report: bool = False
    language: str = "pt-br"


KERNEL_NAME_TO_ID = {
    'linear':     0,
    'polinomial': 1,
    'rbf':        2,
    'sigmoid':    3,
}

# ----------------------------------------------------------------------
# Rodada 7.1A: paleta suave/dessaturada para boxplots comparativos.
# Mesma cor para o mesmo kernel em todos os boxplots; sem legenda
# (o eixo X já identifica os kernels).
# ----------------------------------------------------------------------
KERNEL_COLORS = {
    "Linear":                "#5C9AE0",
    "Polinomial":            "#6896CB",
    "Radial Basis Function": "#779CC6",
    "Sigmoide":              "#8AACD2",
}

# ----------------------------------------------------------------------
# Rodada 7.2A: critérios operacionais dos diagnósticos automáticos.
#
# Estes limiares NÃO representam verdades universais sobre SVM ou kernels.
# Eles servem para interpretar o comportamento observado em uma execução
# específica, considerando dataset, kernel, hiperparâmetros, normalização,
# folds e métricas. Centralizar os valores aqui permite documentá-los no
# relatório e ajustá-los sem precisar editar a lógica de diagnose_svm_performance.
# ----------------------------------------------------------------------
DIAGNOSTIC_THRESHOLDS = {
    # Desempenho geral por F1-score (0–1)
    "excellent_f1": 0.90,
    "good_f1":      0.80,
    "watch_f1":     0.70,

    # Desempenho geral por recall / taxa de detecção (0–1)
    "excellent_recall": 0.90,
    "good_recall":      0.80,
    "watch_recall":     0.70,

    # Desempenho geral por taxa de falsos positivos (0–1)
    "excellent_fpr": 0.05,
    "good_fpr":      0.15,
    "watch_fpr":     0.20,

    # Gap treino−teste em pontos percentuais (0–100)
    "excellent_gap_pp": 5.0,
    "good_gap_pp":     10.0,
    "watch_gap_pp":    20.0,

    # Variabilidade da acurácia de teste em pontos percentuais (0–100)
    "excellent_std_pp": 5.0,
    "good_std_pp":     10.0,
    "high_std_pp":     12.0,

    # Regras críticas específicas
    "overfit_train_acc": 95.0,  # acurácia de treino para suspeita de overfitting (%)
    "underfit_acc":      60.0,  # acurácia (treino e teste) abaixo desta % indica underfitting
    "collapse_recall":    0.05, # recall < 5% indica colapso de classe

    # Regras específicas por kernel (Rodada 7.2B)
    "sigmoide_low_acc":  70.0,  # acc teste abaixo desta % sinaliza instabilidade da Sigmoide
    "linear_low_acc":    75.0,  # acc teste abaixo desta % sugere dados não-lineares para Linear

    # Apoio (referência geral de desempenho), não critério isolado
    "excellent_acc_test": 90.0, # acurácia de teste a partir desta % é "excelente"
}

# ----------------------------------------------------------------------
# Rodada 7.2A: rótulos curtos para cada status diagnóstico.
# Mapeamento exposto para futuras exibições no relatório; não muda a
# semântica atual (a função diagnose_svm_performance continua retornando
# os status técnicos em inglês: 'critical', 'warning', 'good', etc.).
# ----------------------------------------------------------------------
DIAGNOSTIC_STATUS = {
    "excellent": "Excelente",
    "good":      "Bom",
    "watch":     "Observação",
    "warning":   "Atenção",
    "critical":  "Crítico",
    "unstable":  "Instável",
    "skipped":   "Pulado",
    "poor":      "Fraco",
    "unknown":   "Indefinido",
}


def parse_kernels(kernels_str):
    """Converte 'linear,rbf' ou 'all' em lista ordenada de IDs (sem duplicatas)."""
    kernels_str = (kernels_str or '').strip().lower()
    if kernels_str == 'all':
        return [0, 1, 2, 3]
    parts = [p.strip() for p in kernels_str.split(',') if p.strip()]
    if not parts:
        raise ValueError("Lista de kernels vazia.")
    ids, seen = [], set()
    for p in parts:
        if p not in KERNEL_NAME_TO_ID:
            valid = sorted(set(KERNEL_NAME_TO_ID.keys())) + ['all']
            raise ValueError(f"Kernel inválido: '{p}'. Válidos: {valid}")
        kid = KERNEL_NAME_TO_ID[p]
        if kid not in seen:
            seen.add(kid)
            ids.append(kid)
    return ids


def parse_float_list(arg_str, name):
    """Converte '1,10,100' em [1.0, 10.0, 100.0]."""
    parts = [s.strip() for s in (arg_str or '').split(',') if s.strip()]
    if not parts:
        raise ValueError(f"Lista de {name} vazia.")
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"Valor inválido em -{name}: {e}")


def get_gammas_for_kernel(kernel_id, gamma_vector):
    """Para o kernel linear (t=0), gamma é ignorado pelo LIBSVM e usamos [0]
    internamente. Para os demais kernels, retorna o vetor informado.
    """
    if kernel_id == 0:
        return [0]
    return gamma_vector


#========================================================================
# MÉTRICAS & UTILITÁRIOS
#========================================================================

def calculate_detection_metrics(cm):
    """Calcula métricas de detecção de malware.

    Rodada 8.4: incluídas duas métricas complementares de diagnóstico:
      - f1_macro: média não ponderada dos F1-scores da classe positiva
        (maligna) e da classe negativa (benigna). Útil em cenários com
        possível desbalanceamento.
      - balanced_accuracy: média entre a sensibilidade (recall da classe
        positiva) e a especificidade (recall da classe negativa).
    Nenhum critério de ranking foi alterado — as duas métricas são
    apenas exibidas no relatório completo.
    """
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
        # Rodada 8.4: métricas complementares
        'f1_macro': 0,
        'balanced_accuracy': 0,
        # Rodada 8.5: MCC (Matthews Correlation Coefficient)
        'mcc': 0.0,
        'TP': round(TP), 'TN': round(TN), 'FP': round(FP), 'FN': round(FN)
    }
    if metrics['precision'] + metrics['detection_rate'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['detection_rate']) / \
                              (metrics['precision'] + metrics['detection_rate'])

    # ------------------------------------------------------------
    # Rodada 8.4: F1-Macro (média não ponderada dos F1 das duas classes)
    # ------------------------------------------------------------
    # Classe positiva (maligna): já temos precision e recall = detection_rate
    f1_malware = metrics['f1_score']
    # Classe negativa (benigna): recalculada aqui
    precision_benign = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_benign    = TN / (TN + FP) if (TN + FP) > 0 else 0  # = specificity
    if (precision_benign + recall_benign) > 0:
        f1_benign = 2 * precision_benign * recall_benign / (precision_benign + recall_benign)
    else:
        f1_benign = 0
    metrics['f1_macro'] = (f1_malware + f1_benign) / 2

    # ------------------------------------------------------------
    # Rodada 8.4: Balanced Accuracy = (recall_maligna + specificity_benigna) / 2
    # Equivalente a (detection_rate + (1 - false_positive_rate)) / 2 quando
    # (FP + TN) > 0. Usamos a forma direta com specificity = recall_benign.
    # ------------------------------------------------------------
    metrics['balanced_accuracy'] = (metrics['detection_rate'] + recall_benign) / 2

    # ------------------------------------------------------------
    # Rodada 8.5: MCC (Matthews Correlation Coefficient)
    #   MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    # Calculado diretamente a partir de TP/TN/FP/FN para manter
    # consistência com as demais métricas do script. Se qualquer termo
    # do denominador for zero (o produto também será zero), retornamos
    # 0.0, seguindo o padrão defensivo das demais métricas.
    # Varia em [-1, 1]. Não altera o critério de ranking.
    # ------------------------------------------------------------
    mcc_den_sq = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_den_sq > 0:
        metrics['mcc'] = float(((TP * TN) - (FP * FN)) / math.sqrt(mcc_den_sq))
    else:
        metrics['mcc'] = 0.0

    return metrics

def print_detection_metrics(metrics, title="Métricas"):
    """Imprime métricas no terminal."""
    if metrics is None: return
    print(f"\n{'─'*60}\n  📊 {title}\n{'─'*60}")
    print(f"    Taxa de Detecção (Recall):   {metrics['detection_rate']*100:6.2f}%")
    print(f"    Taxa de Falsos Positivos:    {metrics['false_positive_rate']*100:6.2f}%")
    print(f"    Precisão:                    {metrics['precision']*100:6.2f}%")
    print(f"    Matriz de Confusão (Méd):    [TN: {metrics['TN']}, FP: {metrics['FP']}]")
    print(f"                                 [FN: {metrics['FN']}, TP: {metrics['TP']}]")
    print(f"{'─'*60}\n")

def remap_binary_labels(y):
    labels = sorted(set(y))

    if labels == [-1.0, 1.0] or labels == [-1, 1]:
        mapping = {-1.0: 0, 1.0: 1, -1: 0, 1: 1}
    elif labels == [0.0, 1.0] or labels == [0, 1]:
        mapping = {0.0: 0, 1.0: 1, 0: 0, 1: 1}
    else:
        raise ValueError(f"Rótulos binários não suportados automaticamente: {labels}")

    return [mapping[v] for v in y]

def infer_class_labels(y):
    """
    Detecta automaticamente os dois rótulos de uma base binária.
    Retorna a ordem [classe_negativa, classe_positiva].
    """
    labels = sorted(set(y))

    if len(labels) != 2:
        raise ValueError(f"Base não binária ou rótulos inesperados: {labels}")

    return labels

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

def relpath_from_output(target_abs: str, output_html_abs: str) -> str:
    """
    Retorna o caminho relativo (browser-safe) do arquivo target_abs
    a partir do diretório onde o HTML final será salvo.
    Também normaliza barras para funcionar no Windows/Linux no navegador.
    """
    out_dir = os.path.dirname(os.path.abspath(output_html_abs))
    return os.path.relpath(target_abs, start=out_dir).replace("\\", "/")

def format_time(seconds):
    if seconds < 60: return f"{int(seconds)}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def format_list(values, empty_label="Nenhum"):
    """Converte uma lista em string legível separada por vírgula."""
    if not values:
        return empty_label
    return ", ".join(str(v) for v in values)


def kernel_names_from_ids(kernel_ids):
    """Converte uma lista de IDs de kernel em lista de nomes amigáveis
    usando o mapeamento já existente em kernel_str().
    """
    return [kernel_str(kid) for kid in kernel_ids]


def format_optional(value, none_label="Não aplicado"):
    """Formata valor que pode ser None com rótulo amigável."""
    return none_label if value is None else str(value)


def format_bool_ptbr(value):
    """Formata booleano em PT-BR."""
    return "Sim" if value else "Não"


def fit_correlation_pruning(labels, feature_vectors, threshold=None, leakage_threshold=0.999):
    labels = np.asarray(labels, dtype=float)
    feature_vectors = list(feature_vectors)

    max_feature_id = 0
    for sample in feature_vectors:
        if sample:
            max_feature_id = max(max_feature_id, max(sample.keys()))

    if max_feature_id == 0:
        return [], set(), set(), {}

    dense_matrix = np.zeros((len(feature_vectors), max_feature_id), dtype=float)

    for row_index, sample in enumerate(feature_vectors):
        for feature_id, value in sample.items():
            if feature_id > 0:
                dense_matrix[row_index, feature_id - 1] = value

    suspicious_feature_ids = set()
    constant_feature_ids = set()
    selected_feature_ids = []
    correlations = {}

    for column_index in range(dense_matrix.shape[1]):
        feature_id = column_index + 1
        feature_values = dense_matrix[:, column_index]

        if np.std(feature_values) == 0:
            correlations[feature_id] = 0.0
            constant_feature_ids.add(feature_id)
            continue

        corr = np.corrcoef(feature_values, labels)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        correlations[feature_id] = corr

        if abs(corr) >= leakage_threshold:
            suspicious_feature_ids.add(feature_id)
            continue

        if threshold is None or abs(corr) >= threshold:
            selected_feature_ids.append(feature_id)

    return selected_feature_ids, suspicious_feature_ids, constant_feature_ids, correlations

def apply_feature_selection(feature_vectors, selected_feature_ids):
    selected_feature_ids = sorted(selected_feature_ids)
    selected_feature_ids_set = set(selected_feature_ids)

    remap = {
        old_feature_id: new_feature_id
        for new_feature_id, old_feature_id in enumerate(selected_feature_ids, start=1)
    }

    transformed_vectors = []

    for sample in feature_vectors:
        new_sample = {}
        for feature_id, value in sample.items():
            if feature_id in selected_feature_ids_set:
                new_sample[remap[feature_id]] = value
        transformed_vectors.append(new_sample)

    return transformed_vectors, remap


#========================================================================
# NORMALIZAÇÃO FOLD-SAFE (Rodada 6)
# - Métodos: none, maxabs, minmax.
# - O scaler é AJUSTADO usando APENAS o conjunto de treino do fold,
#   garantindo a propriedade fold-safe (sem data leakage).
# - O formato dict (LIBSVM esparso) é preservado.
#========================================================================

NORMALIZATION_DISPLAY = {
    'none':   'nenhuma',
    'maxabs': 'MaxAbs',
    'minmax': 'Min-Max',
}

VALID_NORMALIZATIONS = ('none', 'maxabs', 'minmax')


def fit_feature_scaler(x_train, method):
    """Ajusta o normalizador usando APENAS o conjunto de treino do fold.

    Retorna um dicionário com os parâmetros ajustados, ou None se
    method == 'none' (nenhuma transformação).

    O scaler nunca é ajustado com dados do conjunto de teste, mantendo
    a propriedade fold-safe.

    Parâmetros
    ----------
    x_train : list[dict[int, float]]
        Lista de amostras de treino no formato LIBSVM esparso (dict).
    method : str
        Um de: 'none', 'maxabs', 'minmax'.
    """
    if method is None or method == 'none':
        return None

    if method == 'maxabs':
        # max(|v|) por feature, ignorando ausentes (que valem 0; |0|=0 não afeta o máximo)
        max_abs = {}
        for sample in x_train:
            for k, v in sample.items():
                av = abs(float(v))
                prev = max_abs.get(k)
                if prev is None or av > prev:
                    max_abs[k] = av
        return {'method': 'maxabs', 'max_abs': max_abs}

    if method == 'minmax':
        # min/max por feature, considerando que ausentes representam 0
        # (semântica de LIBSVM esparso).
        n_train = len(x_train)
        present_values = {}
        for sample in x_train:
            for k, v in sample.items():
                present_values.setdefault(k, []).append(float(v))

        mins, maxs = {}, {}
        for k, values in present_values.items():
            n_present = len(values)
            p_min = min(values)
            p_max = max(values)
            if n_present < n_train:
                # alguma amostra de treino tem a feature ausente -> 0 entra no conjunto
                mins[k] = min(p_min, 0.0)
                maxs[k] = max(p_max, 0.0)
            else:
                mins[k] = p_min
                maxs[k] = p_max
        return {'method': 'minmax', 'min': mins, 'max': maxs}

    raise ValueError(f"Método de normalização desconhecido: {method}")


def transform_with_feature_scaler(x_data, scaler):
    """Aplica o scaler (ajustado no treino) a uma lista de amostras.

    Não realiza clipping para valores fora da faixa observada no treino:
    apenas aplica a transformação aprendida. Preserva o formato dict.
    """
    if scaler is None:
        return x_data

    method = scaler.get('method')

    if method == 'maxabs':
        max_abs = scaler['max_abs']
        out = []
        for sample in x_data:
            new_sample = {}
            for k, v in sample.items():
                ma = max_abs.get(k)
                if ma is not None and ma > 0.0:
                    new_v = float(v) / ma
                else:
                    # Feature não vista no treino (ou max_abs == 0): mantém valor original.
                    # Features constantes já foram removidas pela poda fold-safe,
                    # mas o caso ma == 0 é tratado defensivamente.
                    new_v = float(v)
                if new_v != 0.0:
                    new_sample[k] = new_v
            out.append(new_sample)
        return out

    if method == 'minmax':
        mins = scaler['min']
        maxs = scaler['max']
        feature_ids = sorted(mins.keys())
        out = []
        for sample in x_data:
            new_sample = {}
            for k in feature_ids:
                lo = mins[k]
                hi = maxs[k]
                if hi == lo:
                    # Feature constante no treino — sem informação; convencionamos 0
                    continue
                v = float(sample.get(k, 0.0))  # ausente vira 0 (semântica LIBSVM)
                new_v = (v - lo) / (hi - lo)
                if new_v != 0.0:
                    new_sample[k] = new_v
            out.append(new_sample)
        return out

    raise ValueError(f"Método de normalização desconhecido em scaler: {method}")


def apply_fold_safe_normalization(x_train, x_test, method):
    """Helper de conveniência: ajusta no treino e aplica em treino+teste.
    Retorna (x_train_norm, x_test_norm, scaler_info).
    """
    scaler = fit_feature_scaler(x_train, method)
    x_train_norm = transform_with_feature_scaler(x_train, scaler)
    x_test_norm = transform_with_feature_scaler(x_test, scaler)
    return x_train_norm, x_test_norm, scaler


#========================================================================
# VARIABILIDADE (DESVIO PADRÃO)
# - Interpretação semântica do desvio padrão da *acurácia de teste*.
# - Usado para exibir um status curto no relatório (Baixa/Moderada/Alta)
#   sem poluir os cards.
#========================================================================

def classify_test_std(std_test: float):
    """Classifica o desvio padrão (±) da acurácia de teste.

    Retorna:
      - std_class: low | medium | high  (usado no CSS)
      - std_short: Baixa | Moderada | Alta (texto curto no card)
      - std_tooltip: descrição completa (title/tooltip)
    """
    try:
        s = float(std_test)
    except Exception:
        s = 0.0

    if s < 2.0:
        return (
            "low",
            "Baixa",
            "Desvio baixo (< 2%) — Altamente estável (resultados consistentes entre os folds).",
        )
    if s <= 10.0:
        return (
            "medium",
            "Moderada",
            "Desvio moderado (2%–10%) — Variância aceitável (leve sensibilidade às amostras).",
        )
    return (
        "high",
        "Alta",
        "Desvio alto (> 10%) — Instável (performance flutua; possível overfitting em partes dos dados).",
    )


def localize_diagnostic(diagnosis, language):
    """Translate diagnostic prose for the generated report, preserving metric values."""
    if language != "en":
        return diagnosis

    replacements = {
        "Excelente": "Excellent", "Bom": "Good", "Observação": "Watch",
        "Atenção": "Warning", "Crítico": "Critical", "Instável": "Unstable",
        "Pulado": "Skipped", "Fraco": "Poor", "Indefinido": "Unknown",
        "Possível overfitting severo": "Potential severe overfitting",
        "Possível underfitting": "Potential underfitting",
        "Possível colapso de classe": "Potential class collapse",
        "Instabilidade da Sigmoide": "Sigmoid instability",
        "Indício de separabilidade não-linear": "Evidence of nonlinear separability",
        "Alta variabilidade entre folds": "High variability across folds",
        "Taxa elevada de falsos positivos": "High false-positive rate",
        "Métricas-chave abaixo da faixa operacional esperada": "Key metrics below the expected operating range",
        "Acurácia de treino": "Training accuracy",
        "Acurácia de teste": "Test accuracy",
        "Gap treino−teste": "Train–test gap",
        "Desvio padrão da acurácia de teste": "Test accuracy standard deviation",
        "Desvio teste": "Test SD", "Recall observado": "Observed recall",
        "FPR observada": "Observed FPR", "Limiar operacional": "Operating threshold",
        "Kernel Sigmoide": "Sigmoid kernel", "Kernel Linear": "Linear kernel",
        "Sigmoide": "Sigmoid", "Polinomial": "Polynomial", "Radial Basis Function": "Radial Basis Function",
        "Acurácia de treino e de teste ambas abaixo de ": "Both training and test accuracy below ",
        "Recall/Taxa de detecção abaixo de ": "Recall/detection rate below ",
        "Kernel Sigmoide com acurácia de teste abaixo de ": "Sigmoid kernel with test accuracy below ",
        "Kernel Linear com acurácia de teste abaixo de ": "Linear kernel with test accuracy below ",
        "Desvio padrão da acurácia de teste acima de ": "Test accuracy standard deviation above ",
        "FPR acima de ": "FPR above ",
        "F1-score < ": "F1-score < ", "ou recall < ": "or recall < ",
        "Nesta execução, ": "In this run, ",
        "Esta configuração foi classificada como Excelente nesta execução porque ": "This configuration was classified as excellent in this run because ",
        "Esta configuração foi classificada como Boa nesta execução porque ": "This configuration was classified as good in this run because ",
        "apresentou possível overfitting: ": "showed potential overfitting: ",
        "apresentou sinais de underfitting: ": "showed signs of underfitting: ",
        "apresentou baixo desempenho com os ": "showed poor performance with the ",
        "apresentou desempenho limitado, sugerindo ": "showed limited performance, suggesting ",
        "apresentou desempenho limitado, sugerindo possível fronteira não-linear no espaço de features avaliado.": "showed limited performance, suggesting a potentially nonlinear decision boundary in the evaluated feature space.",
        "apresentou FPR alto: ": "showed a high FPR: ",
        " acima do limiar operacional de ": " above the operating threshold of ",
        "Nesta execução, a configuração classificou quase todas as amostras como benignas, resultando em baixa detecção da classe positiva.": "In this run, the configuration classified almost all samples as benign, resulting in low detection of the positive class.",
        "Nesta execução, o kernel Sigmoide apresentou baixo desempenho com os hiperparâmetros avaliados.": "In this run, the Sigmoid kernel showed poor performance with the evaluated hyperparameters.",
        "Nesta execução, o kernel Sigmoide não convergiu para um classificador estável com os hiperparâmetros avaliados.": "In this run, the Sigmoid kernel did not converge to a stable classifier with the evaluated hyperparameters.",
        "Nesta execução, o kernel Linear apresentou desempenho limitado, sugerindo possível fronteira de decisão não-linear.": "In this run, the Linear kernel showed limited performance, suggesting a potentially nonlinear decision boundary.",
        "Nesta execução, o kernel Linear apresentou desempenho limitado, sugerindo possível fronteira não-linear no espaço de features avaliado.": "In this run, the Linear kernel showed limited performance, suggesting a potentially nonlinear decision boundary in the evaluated feature space.",
        "F1-score, recall ou FPR fora da faixa operacional esperada.": "F1-score, recall, or FPR outside the expected operating range.",
        "Esta configuração acionou um critério de atenção nesta execução devido a ": "This configuration triggered a warning criterion in this run due to ",
        "Uma proporção elevada de amostras benignas foi classificada como maliciosa nesta execução — a configuração tende a gerar falsos positivos.": "A large proportion of benign samples were classified as malicious in this run, indicating a tendency to generate false positives.",
        "A configuração apresentou desempenho limitado na captura dos padrões básicos do problema nesta execução.": "The configuration showed limited ability to capture the basic patterns of the problem in this run.",
        "Nesta execução, a configuração entrou em colapso de classe — quase nenhum exemplo positivo foi reconhecido.": "In this run, the configuration collapsed to one class—almost no positive examples were identified.",
        "O modelo se ajustou fortemente ao conjunto de treino, mas não generalizou para os folds de teste com a mesma qualidade.": "The model fit the training data strongly but did not generalize equally well to the test folds.",
        "Nesta execução, a configuração apresentou alta variância entre os folds — a estimativa de desempenho é pouco confiável.": "In this run, the configuration showed high variance across folds, making its performance estimate less reliable.",
        "Desempenho fora do esperado em uma ou mais métricas-chave (F1 ou recall) nesta execução.": "Performance was below expectations on one or more key metrics (F1 or recall) in this run.",
        "Desempenho consistente nas métricas e estabilidade aceitável entre folds.": "Consistent metric performance and acceptable stability across folds.",
        "Métricas de desempenho e estabilidade simultaneamente dentro da faixa de excelência operacional.": "Performance and stability metrics were simultaneously within the operational excellence range.",
        "Configuração candidata para avaliação final ou comparação com experimentos adicionais.": "Candidate configuration for final evaluation or comparison with additional experiments.",
        "Avaliar redução de C e revisar a normalização em novas execuções.": "Consider reducing C and reviewing normalization in subsequent runs.",
        "Avaliar aumento de C ou ajuste de gamma em uma nova busca de hiperparâmetros.": "Consider increasing C or tuning gamma in a new hyperparameter search.",
        "Avaliar pesos de classe (-wi), ajuste de C ou balanceamento dos dados em novas execuções.": "Consider class weights (-wi), tuning C, or balancing the data in subsequent runs.",
        "Avaliar o kernel Radial Basis Function (RBF) como alternativa em novas execuções.": "Consider the Radial Basis Function (RBF) kernel as an alternative in subsequent runs.",
        "Avaliar maior regularização, por exemplo com redução de C.": "Consider stronger regularization, for example by reducing C.",
        "Avaliar ajuste de gamma, pesos de classe ou limiar de decisão para reduzir falsos positivos.": "Consider tuning gamma, class weights, or the decision threshold to reduce false positives.",
        "Revisar hiperparâmetros, como C e gamma, além do método de normalização.": "Review hyperparameters such as C and gamma, as well as the normalization method.",
        "Avaliar uma busca de hiperparâmetros com passos menores para refinar a configuração.": "Consider a finer-grained hyperparameter search to refine the configuration.",
        "Avaliar uma busca de hiperparâmetros com passos menores.": "Consider a finer-grained hyperparameter search.",
        "Métricas em zona intermediária entre os critérios de atenção e os de bom.": "Metrics fall in an intermediate range between warning and good-performance criteria.",
        "Esta configuração ficou em zona de observação nesta execução: uma ou mais métricas ficaram próximas dos limiares de atenção, sem caracterizar falha crítica.": "This configuration was flagged for review in this run: one or more metrics were close to warning thresholds, without indicating a critical failure.",
        "Desempenho aceitável, mas com margem para melhoria em alguma métrica (F1, recall, FPR, gap ou desvio).": "Acceptable performance, with room to improve one or more metrics (F1, recall, FPR, gap, or standard deviation).",
        "Pulado: ": "Skipped: ",
        "O kernel não foi executado nesta rodada por critério operacional de tempo.": "The kernel was not run in this experiment due to the operational time criterion.",
        "Tempo estimado acima do limite configurado.": "Estimated time exceeds the configured limit.",
        "Não executado por estimativa de tempo": "Skipped based on time estimate",
        "Avaliar --force-all-kernels, redução do grid, redução do número de folds ou aumento de -max-time em nova execução.": "Consider --force-all-kernels, reducing the grid or number of folds, or increasing -max-time in a subsequent run.",
        "F1-score: ": "F1-score: ", "Recall: ": "Recall: ", "FPR: ": "FPR: ",
        "Kernel: ": "Kernel: ", "p.p.": "percentage points",
    }
    result = dict(diagnosis)
    result["status_label"] = replacements.get(result.get("status_label"), result.get("status_label"))

    def translate(value):
        if isinstance(value, str):
            patterns = [
                (r"Acurácia de treino > ([\d.]+)% e gap treino−teste > ([\d.]+) p\.p\.",
                 r"Training accuracy > \1% and train–test gap > \2 percentage points."),
                (r"Acurácia de treino e de teste ambas abaixo de ([\d.]+)%\.",
                 r"Both training and test accuracy below \1%."),
                (r"Recall/Taxa de detecção abaixo de ([\d.]+)%\.",
                 r"Recall/detection rate below \1%."),
                (r"Kernel Sigmoide com acurácia de teste abaixo de ([\d.]+)%\.",
                 r"Sigmoid kernel with test accuracy below \1%."),
                (r"Kernel Linear com acurácia de teste abaixo de ([\d.]+)%\.",
                 r"Linear kernel with test accuracy below \1%."),
                (r"Desvio padrão da acurácia de teste acima de ([\d.]+) p\.p\.",
                 r"Test accuracy standard deviation above \1 percentage points."),
                (r"FPR acima de ([\d.]+)%\.", r"FPR above \1%."),
                (r"F1-score < ([\d.]+) ou recall < ([\d.]+)%\.",
                 r"F1-score < \1 or recall < \2%."),
                (r"F1 ≥ ([\d.]+), recall ≥ ([\d.]+)%, FPR ≤ ([\d.]+)%, gap ≤ ([\d.]+) p\.p\. e desvio ≤ ([\d.]+) p\.p\.",
                 r"F1 ≥ \1, recall ≥ \2%, FPR ≤ \3%, gap ≤ \4 percentage points, and standard deviation ≤ \5 percentage points."),
                (r"Acurácia de treino: ([\d.]+)%", r"Training accuracy: \1%"),
                (r"Acurácia de teste: ([\d.]+)%", r"Test accuracy: \1%"),
                (r"Gap treino−teste: ([\d.]+) p\.p\.", r"Train–test gap: \1 percentage points"),
                (r"Desvio teste: ±([\d.]+) p\.p\.", r"Test SD: ±\1 percentage points"),
                (r"Recall observado: ([\d.]+)%", r"Observed recall: \1%"),
                (r"FPR observada: ([\d.]+)%", r"Observed FPR: \1%"),
                (r"Limiar operacional: ([\d.]+)%", r"Operating threshold: \1%"),
                (r"Nesta execução, a configuração apresentou possível overfitting: treino ([\d.]+)%, teste ([\d.]+)% e gap ([\d.]+) p\.p\.",
                 r"In this run, the configuration showed potential overfitting: training accuracy \1%, test accuracy \2%, and train–test gap \3 percentage points."),
                (r"Nesta execução, a configuração apresentou sinais de underfitting: treino ([\d.]+)% e teste ([\d.]+)%\.",
                 r"In this run, the configuration showed signs of underfitting: training accuracy \1% and test accuracy \2%."),
                (r"A métrica apresentou variação elevada entre os folds \(±([\d.]+) p\.p\.\)\.",
                 r"The metric showed high variation across folds (±\1 percentage points)."),
                (r"Nesta execução, a configuração apresentou FPR alto: ([\d.]+)%, acima do limiar operacional de ([\d.]+)%\.",
                 r"In this run, the configuration showed a high FPR of \1%, above the operating threshold of \2%."),
            ]
            for pattern, replacement in patterns:
                value = re.sub(pattern, replacement, value)
            for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
                value = value.replace(source, target)
            return value
        if isinstance(value, list):
            return [translate(item) for item in value]
        return value

    for key in ("problem", "criterion", "evidence", "explanation", "interpretation", "recommendation"):
        result[key] = translate(result.get(key))
    return result

#========================================================================
# DIAGNÓSTICO E ESTIMATIVA
#========================================================================

def estimate_and_filter_kernels(y, x, kernels_to_test, cost_vector, gamma_vector, k_folds,
                                max_time=3600.0, force_all_kernels=False):
    n_samples = len(y)
    max_feature_idx = 0
    if len(x) > 0:
        for row in x[:100]:
            if row:
                max_feature_idx = max(max_feature_idx, max(row.keys()))
    n_features = max_feature_idx

    total_theo = n_samples * n_features
    total_non_zeros = sum(len(row) for row in x)
    density = total_non_zeros / total_theo if total_theo > 0 else 1.0
    base_factor = 2.0e-7

    kernels_kept = []
    skipped_info = {}

    print(f"\n{'='*80}\n⏱️  ESTIMATIVA OPERACIONAL DE TEMPO\n{'='*80}")
    if force_all_kernels:
        print("  ⚙️  --force-all-kernels ativo: nenhum kernel será descartado por tempo estimado.")
    print(f"  ⚙️  Limite por kernel: {format_time(max_time)}")

    total_est = 0
    total_iters = 0

    for t in kernels_to_test:
        k_name = kernel_str(t)
        gammas = get_gammas_for_kernel(t, gamma_vector)
        iters_kernel = len(cost_vector) * len(gammas) * k_folds
        comp_mult = 5.0 if t == 1 else 1.2 if t == 2 else 1.0
        est_time = base_factor * (n_samples ** 2) * n_features * density * comp_mult * iters_kernel

        if est_time > max_time and not force_all_kernels:
            skipped_info[t] = {
                'reason': 'Não executado por estimativa de tempo',
                'detail': f'Est: {format_time(est_time)} > limite {format_time(max_time)}'
            }
            print(f"  ⚠️  Não executando {k_name}: Est. {format_time(est_time)} > Limite {format_time(max_time)}")
            continue

        total_est += est_time
        total_iters += iters_kernel
        kernels_kept.append(t)
        if est_time > max_time and force_all_kernels:
            print(f"  > Selecionado {k_name:<10} (forçado): ~{format_time(est_time)} > limite {format_time(max_time)}")
        else:
            print(f"  > Selecionado {k_name:<10}: ~{format_time(est_time)}")

    print(f"\n  ⏱️  Tempo Total Est.: ~{format_time(total_est)}")
    print("="*80 + "\n")
    return kernels_kept, skipped_info, total_iters

def diagnose_svm_performance(results, kernel_name):
    """
    Rodada 7.2B: diagnóstico mais explicável e contextual.

    O dicionário retornado mantém as chaves antigas usadas pelo template atual:
        status, problem, explanation, recommendation

    e adiciona as novas chaves preparadas para a Rodada 7.2C:
        status_label, criterion, evidence, interpretation, thresholds_used

    A função usa apenas DIAGNOSTIC_THRESHOLDS (e DIAGNOSTIC_STATUS) como
    fonte de critérios — nenhum limiar fica hardcoded.
    """
    diagnosis = {
        "status":          "unknown",
        "status_label":    DIAGNOSTIC_STATUS.get("unknown", "Indefinido"),
        "problem":         None,
        "criterion":       "",
        "evidence":        [],
        "explanation":     "",
        "interpretation":  "",
        "recommendation":  "",
        "thresholds_used": [],
    }
    if not results:
        return diagnosis

    # --- métricas de entrada ---
    acc_test  = results.get('accuracy_test', 0)
    acc_train = results.get('accuracy_train', 0)
    std       = results.get('std_test', 0)
    f1        = results.get('f1_score', 0)
    precision = results.get('precision', 0)
    cm        = results.get('confusion_matrix_test', None)

    # det_rate / fpr: usa o valor explícito se vier; senão deriva da CM
    det_rate_metric = results.get('detection_rate', None)
    fpr_metric      = results.get('false_positive_rate', None)
    det_rate_from_cm, fpr_from_cm = 0.0, 0.0
    if cm is not None and hasattr(cm, 'size') and cm.size == 4:
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        pos = TP + FN
        neg = TN + FP
        det_rate_from_cm = TP / pos if pos > 0 else 0.0
        fpr_from_cm      = FP / neg if neg > 0 else 0.0
    det_rate = det_rate_metric if det_rate_metric is not None else det_rate_from_cm
    fpr      = fpr_metric      if fpr_metric      is not None else fpr_from_cm

    gap = acc_train - acc_test

    # Rodada 7.2B: normalizar o nome do kernel para evitar a inconsistência
    # "Sigmoid" (interno) vs "Sigmoide" (exibido em PT-BR).
    kernel_label = (kernel_name or "").strip().lower()
    is_sigmoide  = ("sigmoide" in kernel_label) or ("sigmoid" in kernel_label)
    is_linear    = "linear" in kernel_label

    # --- cascata de regras (mesma ordem do baseline, agora 100% via constantes) ---
    if acc_train > DIAGNOSTIC_THRESHOLDS["overfit_train_acc"] and gap > DIAGNOSTIC_THRESHOLDS["watch_gap_pp"]:
        diagnosis['status']          = 'critical'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['critical']
        diagnosis['problem']         = 'Possível overfitting severo'
        diagnosis['criterion']       = (
            f"Acurácia de treino > {DIAGNOSTIC_THRESHOLDS['overfit_train_acc']:.1f}% "
            f"e gap treino−teste > {DIAGNOSTIC_THRESHOLDS['watch_gap_pp']:.1f} p.p."
        )
        diagnosis['evidence']        = [
            f"Acurácia de treino: {acc_train:.1f}%",
            f"Acurácia de teste: {acc_test:.1f}%",
            f"Gap treino−teste: {gap:.1f} p.p.",
        ]
        diagnosis['explanation']     = (
            f"Nesta execução, a configuração apresentou possível overfitting: "
            f"treino {acc_train:.1f}%, teste {acc_test:.1f}% e gap {gap:.1f} p.p."
        )
        diagnosis['interpretation']  = (
            "O modelo se ajustou fortemente ao conjunto de treino, mas não generalizou "
            "para os folds de teste com a mesma qualidade."
        )
        diagnosis['recommendation']  = "Avaliar redução de C e revisar a normalização em novas execuções."
        diagnosis['thresholds_used'] = ['overfit_train_acc', 'watch_gap_pp']

    elif acc_train < DIAGNOSTIC_THRESHOLDS["underfit_acc"] and acc_test < DIAGNOSTIC_THRESHOLDS["underfit_acc"]:
        diagnosis['status']          = 'critical'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['critical']
        diagnosis['problem']         = 'Possível underfitting'
        diagnosis['criterion']       = (
            f"Acurácia de treino e de teste ambas abaixo de "
            f"{DIAGNOSTIC_THRESHOLDS['underfit_acc']:.1f}%."
        )
        diagnosis['evidence']        = [
            f"Acurácia de treino: {acc_train:.1f}%",
            f"Acurácia de teste: {acc_test:.1f}%",
        ]
        diagnosis['explanation']     = (
            f"Nesta execução, a configuração apresentou sinais de underfitting: "
            f"treino {acc_train:.1f}% e teste {acc_test:.1f}%."
        )
        diagnosis['interpretation']  = (
            "A configuração apresentou desempenho limitado na captura dos padrões básicos "
            "do problema nesta execução."
        )
        diagnosis['recommendation']  = "Avaliar aumento de C ou ajuste de gamma em uma nova busca de hiperparâmetros."
        diagnosis['thresholds_used'] = ['underfit_acc']

    elif det_rate < DIAGNOSTIC_THRESHOLDS["collapse_recall"]:
        diagnosis['status']          = 'critical'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['critical']
        diagnosis['problem']         = 'Possível colapso de classe'
        diagnosis['criterion']       = (
            f"Recall/Taxa de detecção abaixo de "
            f"{DIAGNOSTIC_THRESHOLDS['collapse_recall']*100:.1f}%."
        )
        diagnosis['evidence']        = [
            f"Recall observado: {det_rate*100:.1f}%",
        ]
        diagnosis['explanation']     = (
            "Nesta execução, a configuração classificou quase todas as amostras como "
            "benignas, resultando em baixa detecção da classe positiva."
        )
        diagnosis['interpretation']  = (
            "Nesta execução, a configuração entrou em colapso de classe — quase nenhum "
            "exemplo positivo foi reconhecido."
        )
        diagnosis['recommendation']  = "Avaliar pesos de classe (-wi), ajuste de C ou balanceamento dos dados em novas execuções."
        diagnosis['thresholds_used'] = ['collapse_recall']

    elif is_sigmoide and acc_test < DIAGNOSTIC_THRESHOLDS["sigmoide_low_acc"]:
        diagnosis['status']          = 'warning'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['warning']
        diagnosis['problem']         = 'Instabilidade da Sigmoide'
        diagnosis['criterion']       = (
            f"Kernel Sigmoide com acurácia de teste abaixo de "
            f"{DIAGNOSTIC_THRESHOLDS['sigmoide_low_acc']:.1f}%."
        )
        diagnosis['evidence']        = [
            f"Kernel: {kernel_name}",
            f"Acurácia de teste: {acc_test:.1f}%",
        ]
        diagnosis['explanation']     = (
            "Nesta execução, o kernel Sigmoide apresentou baixo desempenho com os "
            "hiperparâmetros avaliados."
        )
        diagnosis['interpretation']  = (
            "Nesta execução, o kernel Sigmoide não convergiu para um classificador estável "
            "com os hiperparâmetros avaliados."
        )
        diagnosis['recommendation']  = "Avaliar o kernel Radial Basis Function (RBF) como alternativa em novas execuções."
        diagnosis['thresholds_used'] = ['sigmoide_low_acc']

    elif is_linear and acc_test < DIAGNOSTIC_THRESHOLDS["linear_low_acc"]:
        diagnosis['status']          = 'poor'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['poor']
        diagnosis['problem']         = 'Indício de separabilidade não-linear'
        diagnosis['criterion']       = (
            f"Kernel Linear com acurácia de teste abaixo de "
            f"{DIAGNOSTIC_THRESHOLDS['linear_low_acc']:.1f}%."
        )
        diagnosis['evidence']        = [
            f"Kernel: {kernel_name}",
            f"Acurácia de teste: {acc_test:.1f}%",
        ]
        diagnosis['explanation']     = (
            "Nesta execução, o kernel Linear apresentou desempenho limitado, sugerindo "
            "possível fronteira de decisão não-linear."
        )
        diagnosis['interpretation']  = (
            "Nesta execução, o kernel Linear apresentou desempenho limitado, sugerindo "
            "possível fronteira não-linear no espaço de features avaliado."
        )
        diagnosis['recommendation']  = "Avaliar o kernel Radial Basis Function (RBF) como alternativa em novas execuções."
        diagnosis['thresholds_used'] = ['linear_low_acc']

    elif std > DIAGNOSTIC_THRESHOLDS["high_std_pp"]:
        diagnosis['status']          = 'unstable'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['unstable']
        diagnosis['problem']         = 'Alta variabilidade entre folds'
        diagnosis['criterion']       = (
            f"Desvio padrão da acurácia de teste acima de "
            f"{DIAGNOSTIC_THRESHOLDS['high_std_pp']:.1f} p.p."
        )
        diagnosis['evidence']        = [
            f"Desvio padrão da acurácia de teste: ±{std:.1f} p.p.",
        ]
        diagnosis['explanation']     = (
            f"A métrica apresentou variação elevada entre os folds (±{std:.1f} p.p.)."
        )
        diagnosis['interpretation']  = (
            "Nesta execução, a configuração apresentou alta variância entre os folds — "
            "a estimativa de desempenho é pouco confiável."
        )
        diagnosis['recommendation']  = "Avaliar maior regularização, por exemplo com redução de C."
        diagnosis['thresholds_used'] = ['high_std_pp']

    elif fpr > DIAGNOSTIC_THRESHOLDS["watch_fpr"]:
        diagnosis['status']          = 'warning'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['warning']
        diagnosis['problem']         = 'Taxa elevada de falsos positivos'
        diagnosis['criterion']       = (
            f"FPR acima de {DIAGNOSTIC_THRESHOLDS['watch_fpr']*100:.1f}%."
        )
        diagnosis['evidence']        = [
            f"FPR observada: {fpr*100:.1f}%",
            f"Limiar operacional: {DIAGNOSTIC_THRESHOLDS['watch_fpr']*100:.1f}%",
        ]
        diagnosis['explanation']     = (
            f"Nesta execução, a configuração apresentou FPR alto: {fpr*100:.1f}%, "
            f"acima do limiar operacional de {DIAGNOSTIC_THRESHOLDS['watch_fpr']*100:.1f}%."
        )
        diagnosis['interpretation']  = (
            "Uma proporção elevada de amostras benignas foi classificada como maliciosa "
            "nesta execução — a configuração tende a gerar falsos positivos."
        )
        diagnosis['recommendation']  = "Avaliar ajuste de gamma, pesos de classe ou limiar de decisão para reduzir falsos positivos."
        diagnosis['thresholds_used'] = ['watch_fpr']

    # Rodada 7.2B.1: ATENÇÃO por degradação de F1 ou recall abaixo da faixa mínima
    # (depois das regras críticas e específicas, antes da avaliação por conjunto)
    elif f1 < DIAGNOSTIC_THRESHOLDS["watch_f1"] or det_rate < DIAGNOSTIC_THRESHOLDS["watch_recall"]:
        diagnosis['status']          = 'warning'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['warning']
        diagnosis['problem']         = 'Métricas-chave abaixo da faixa operacional esperada'
        diagnosis['criterion']       = (
            f"F1-score < {DIAGNOSTIC_THRESHOLDS['watch_f1']:.2f} "
            f"ou recall < {DIAGNOSTIC_THRESHOLDS['watch_recall']*100:.1f}%."
        )
        diagnosis['evidence']        = [
            f"F1-score: {f1:.4f}",
            f"Recall: {det_rate*100:.1f}%",
            f"FPR: {fpr*100:.1f}%",
            f"Acurácia de teste: {acc_test:.1f}%",
        ]
        diagnosis['explanation']     = (
            "Esta configuração acionou um critério de atenção nesta execução devido a "
            "F1-score, recall ou FPR fora da faixa operacional esperada."
        )
        diagnosis['interpretation']  = (
            "Desempenho fora do esperado em uma ou mais métricas-chave (F1 ou recall) "
            "nesta execução."
        )
        diagnosis['recommendation']  = "Revisar hiperparâmetros, como C e gamma, além do método de normalização."
        diagnosis['thresholds_used'] = ['watch_f1', 'watch_recall']

    # Rodada 7.2B.1: EXCELENTE — exige que todas as 5 métricas estejam na faixa de excelência
    elif (
        f1       >= DIAGNOSTIC_THRESHOLDS["excellent_f1"]      and
        det_rate >= DIAGNOSTIC_THRESHOLDS["excellent_recall"]  and
        fpr      <= DIAGNOSTIC_THRESHOLDS["excellent_fpr"]     and
        gap      <= DIAGNOSTIC_THRESHOLDS["excellent_gap_pp"]  and
        std      <= DIAGNOSTIC_THRESHOLDS["excellent_std_pp"]
    ):
        diagnosis['status']          = 'excellent'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['excellent']
        diagnosis['problem']         = None
        diagnosis['criterion']       = (
            f"F1 ≥ {DIAGNOSTIC_THRESHOLDS['excellent_f1']:.2f}, "
            f"recall ≥ {DIAGNOSTIC_THRESHOLDS['excellent_recall']*100:.1f}%, "
            f"FPR ≤ {DIAGNOSTIC_THRESHOLDS['excellent_fpr']*100:.1f}%, "
            f"gap ≤ {DIAGNOSTIC_THRESHOLDS['excellent_gap_pp']:.1f} p.p. "
            f"e desvio ≤ {DIAGNOSTIC_THRESHOLDS['excellent_std_pp']:.1f} p.p."
        )
        diagnosis['evidence']        = [
            f"F1-score: {f1:.4f}",
            f"Recall: {det_rate*100:.1f}%",
            f"FPR: {fpr*100:.1f}%",
            f"Gap treino−teste: {gap:.1f} p.p.",
            f"Desvio teste: ±{std:.1f} p.p.",
        ]
        diagnosis['explanation']     = (
            "Esta configuração foi classificada como Excelente nesta execução porque "
            "apresentou F1-score, recall, FPR, gap treino-teste e variabilidade dentro "
            "dos critérios operacionais de excelência."
        )
        diagnosis['interpretation']  = (
            "Métricas de desempenho e estabilidade simultaneamente dentro da faixa de "
            "excelência operacional."
        )
        diagnosis['recommendation']  = "Configuração candidata para avaliação final ou comparação com experimentos adicionais."
        diagnosis['thresholds_used'] = [
            'excellent_f1', 'excellent_recall', 'excellent_fpr',
            'excellent_gap_pp', 'excellent_std_pp',
        ]

    # Rodada 7.2B.1: BOM — exige que todas as 5 métricas estejam na faixa "bom" (mais relaxada)
    elif (
        f1       >= DIAGNOSTIC_THRESHOLDS["good_f1"]      and
        det_rate >= DIAGNOSTIC_THRESHOLDS["good_recall"]  and
        fpr      <= DIAGNOSTIC_THRESHOLDS["good_fpr"]     and
        gap      <= DIAGNOSTIC_THRESHOLDS["good_gap_pp"]  and
        std      <= DIAGNOSTIC_THRESHOLDS["good_std_pp"]
    ):
        diagnosis['status']          = 'good'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['good']
        diagnosis['problem']         = None
        diagnosis['criterion']       = (
            f"F1 ≥ {DIAGNOSTIC_THRESHOLDS['good_f1']:.2f}, "
            f"recall ≥ {DIAGNOSTIC_THRESHOLDS['good_recall']*100:.1f}%, "
            f"FPR ≤ {DIAGNOSTIC_THRESHOLDS['good_fpr']*100:.1f}%, "
            f"gap ≤ {DIAGNOSTIC_THRESHOLDS['good_gap_pp']:.1f} p.p. "
            f"e desvio ≤ {DIAGNOSTIC_THRESHOLDS['good_std_pp']:.1f} p.p."
        )
        diagnosis['evidence']        = [
            f"F1-score: {f1:.4f}",
            f"Recall: {det_rate*100:.1f}%",
            f"FPR: {fpr*100:.1f}%",
            f"Gap treino−teste: {gap:.1f} p.p.",
            f"Desvio teste: ±{std:.1f} p.p.",
        ]
        diagnosis['explanation']     = (
            "Esta configuração foi classificada como Boa nesta execução porque "
            "apresentou F1-score, recall, FPR, gap treino-teste e variabilidade dentro "
            "dos critérios operacionais definidos."
        )
        diagnosis['interpretation']  = (
            "Desempenho consistente nas métricas e estabilidade aceitável entre folds."
        )
        diagnosis['recommendation']  = "Avaliar uma busca de hiperparâmetros com passos menores para refinar a configuração."
        diagnosis['thresholds_used'] = [
            'good_f1', 'good_recall', 'good_fpr', 'good_gap_pp', 'good_std_pp',
        ]

    # Rodada 7.2B.1: OBSERVAÇÃO — caso residual (zona intermediária)
    # Configurações que não acionaram nenhuma regra crítica/de atenção, mas também não
    # atingiram simultaneamente os critérios de "Bom" ou "Excelente". Preferir "Observação"
    # em vez de classificar automaticamente como "Bom".
    else:
        diagnosis['status']          = 'watch'
        diagnosis['status_label']    = DIAGNOSTIC_STATUS['watch']
        diagnosis['problem']         = None
        diagnosis['criterion']       = (
            "Métricas em zona intermediária entre os critérios de atenção e os de bom."
        )
        diagnosis['evidence']        = [
            f"F1-score: {f1:.4f}",
            f"Recall: {det_rate*100:.1f}%",
            f"FPR: {fpr*100:.1f}%",
            f"Gap treino−teste: {gap:.1f} p.p.",
            f"Desvio teste: ±{std:.1f} p.p.",
        ]
        diagnosis['explanation']     = (
            "Esta configuração ficou em zona de observação nesta execução: uma ou mais "
            "métricas ficaram próximas dos limiares de atenção, sem caracterizar falha "
            "crítica."
        )
        diagnosis['interpretation']  = (
            "Desempenho aceitável, mas com margem para melhoria em alguma métrica "
            "(F1, recall, FPR, gap ou desvio)."
        )
        diagnosis['recommendation']  = "Avaliar uma busca de hiperparâmetros com passos menores."
        diagnosis['thresholds_used'] = [
            'good_f1', 'good_recall', 'good_fpr', 'good_gap_pp', 'good_std_pp',
        ]

    return diagnosis

#========================================================================
# PLOTAGEM E SVM CORE
#========================================================================

def kernel_str(t):
    # Traduzindo os nomes dos kernels para exibição
    return {0:'Linear', 1:'Polinomial', 2:'Radial Basis Function', 3:'Sigmoide'}.get(t, 'Desconhecido')

def plot_and_save_cm(cm_mean, cm_std, title, filename, language="pt-br"):
    """
    Plota a matriz de confusão no layout desejado para o relatório.

    Convenção interna esperada no código:
        [[TN, FP],
         [FN, TP]]

    Convenção exibida no gráfico:
        [[TP, FP],
         [FN, TN]]

    Exibição:
        - colunas = Rótulo Real
        - linhas  = Rótulo Previsto
        - ordem visual = [Maligno, Benigno]
    """
    if cm_mean is None:
        return

    if cm_mean.shape != (2, 2):
        raise ValueError("A matriz de confusão média deve ser 2x2.")

    if cm_std is not None and cm_std.shape != (2, 2):
        raise ValueError("A matriz de desvio padrão deve ser 2x2.")

    # ------------------------------------------------------------------
    # Reorganização apenas para exibição:
    # interno:  [[TN, FP],
    #            [FN, TP]]
    #
    # display:  [[TP, FP],
    #            [FN, TN]]
    # ------------------------------------------------------------------
    cm_display = np.array([
        [cm_mean[1, 1], cm_mean[0, 1]],  # TP, FP
        [cm_mean[1, 0], cm_mean[0, 0]]   # FN, TN
    ], dtype=float)

    if cm_std is not None:
        cm_std_display = np.array([
            [cm_std[1, 1], cm_std[0, 1]],  # TP, FP
            [cm_std[1, 0], cm_std[0, 0]]   # FN, TN
        ], dtype=float)
    else:
        cm_std_display = np.zeros((2, 2), dtype=float)

    # ------------------------------------------------------------------
    # Normalização por coluna, pois as colunas representam o Rótulo Real
    # ------------------------------------------------------------------
    col_sums = cm_display.sum(axis=0, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_pct = (cm_display / col_sums) * 100.0
    cm_pct = np.nan_to_num(cm_pct)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm_std_pct = (cm_std_display / col_sums) * 100.0
    cm_std_pct = np.nan_to_num(cm_std_pct)

    # ------------------------------------------------------------------
    # Anotação mínima: sigla + média + desvio
    # ------------------------------------------------------------------
    cell_codes = np.array([
        ["TP", "FP"],
        ["FN", "TN"]
    ])

    annot = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            annot[i, j] = (
                f"{cell_codes[i, j]}\n"
                f"{cm_pct[i, j]:.1f}%\n"
                f"±{cm_std_pct[i, j]:.2f}%"
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6.8, 5.6))

    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt='',
        cmap=plt.cm.Blues,
        cbar_kws={'label': 'Percentage (%)' if language == 'en' else 'Porcentagem (%)'},
        annot_kws={"size": 12, "weight": "bold"},
        xticklabels=["Malicious", "Benign"] if language == 'en' else ["Maligno", "Benigno"],
        yticklabels=["Malicious", "Benign"] if language == 'en' else ["Maligno", "Benigno"]
    )

    plt.title(title, fontsize=11, pad=12, fontweight='bold')
    plt.xlabel("True Label" if language == 'en' else "Rótulo Real", fontsize=10, fontweight='bold')
    plt.ylabel("Predicted Label" if language == 'en' else "Rótulo Previsto", fontsize=10, fontweight='bold')
    plt.xticks(fontsize=10, rotation=0)
    plt.yticks(fontsize=10, rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ------------------------------------------------------------------
# Observação histórica:
# As funções save_libsvm() e pruningDataset() foram REMOVIDAS na Rodada 4.
# Motivo: pruningDataset() executava poda supervisionada GLOBAL antes do
# K-Fold, calculando correlação com o dataset inteiro — o que pode causar
# data leakage do conjunto de teste. Sua substituta é a poda fold-safe
# em fit_correlation_pruning() chamada dentro de svmKfold(), e a auditoria
# por fold gravada em svm_feature_audit_*.csv.
# Os arquivos antigos selected_features.csv e globalPruned.libsvm não são
# mais gerados.
# ------------------------------------------------------------------


def svmKfold(y, x, t, cost, gamma, k_folds, class_labels_order,
             seed=1, cache=500, epsilon=0.05,
             threshold=None, leakage_threshold=0.999,
             normalization='none',
             audit_csv_path=None, kernel_name=None,
             callback=None):
    splitter = StratifiedKFold(
        n_splits=k_folds,
        shuffle=True,
        random_state=seed
    )

    acc_tr, acc_te = [], []
    t_tr, t_te = [], []

    detection_rates = []
    false_positive_rates = []
    precisions = []
    f1_scores = []

    # Rodada 8.4: métricas complementares (não afetam ranking)
    f1_macros = []
    balanced_accuracies = []
    # Rodada 8.5: MCC (não afeta ranking)
    mcc_scores = []

    cms_te = []
    suspicious_counts = []
    selected_feature_counts = []
    audit_rows_written = 0

    param = f'-t {t} -c {cost} -g {gamma} -m {cache} -e {epsilon} -q'

    # Abre o CSV de auditoria UMA vez por (kernel, C, gamma); cada fold acrescenta linhas.
    audit_file = None
    audit_writer = None
    audit_selected_file = None
    audit_selected_writer = None
    if audit_csv_path is not None:
        audit_file = open(audit_csv_path, 'a', newline='', encoding='utf-8')
        audit_writer = csv.writer(audit_file)
        selected_audit_path = os.path.splitext(audit_csv_path)[0] + '_selected.csv'
        audit_selected_file = open(selected_audit_path, 'a', newline='', encoding='utf-8')
        audit_selected_writer = csv.writer(audit_selected_file)

    try:
        for fold_number, (tr_idx, te_idx) in enumerate(splitter.split(x, y), start=1):
            x_tr = [x[i] for i in tr_idx]
            x_te = [x[i] for i in te_idx]
            y_tr = [y[i] for i in tr_idx]
            y_te = [y[i] for i in te_idx]

            # Poda supervisionada FOLD-SAFE: ajusta SOMENTE no treino do fold.
            (selected_feature_ids,
             suspicious_feature_ids,
             constant_feature_ids,
             correlations) = fit_correlation_pruning(
                y_tr,
                x_tr,
                threshold=threshold,
                leakage_threshold=leakage_threshold
            )
            suspicious_counts.append(len(suspicious_feature_ids))

            # Fallback: se nada foi selecionado, usa todas as features do treino
            # EXCETO suspeitas E constantes (continua fold-safe e sem leakage).
            if len(selected_feature_ids) == 0:
                all_train_feature_ids = sorted({
                    feature_id
                    for sample in x_tr
                    for feature_id in sample.keys()
                    if feature_id not in suspicious_feature_ids
                    and feature_id not in constant_feature_ids
                })
                selected_feature_ids = all_train_feature_ids

            selected_set = set(selected_feature_ids)

            # Rodada 8.3: registrar quantas features restaram após seleção/auditoria
            # (fold-safe). Contagem feita DEPOIS do fallback; ela representa o número
            # de features efetivamente enviadas ao SVM neste fold.
            selected_feature_counts.append(len(selected_feature_ids))

            # ---- Auditoria fold-safe: registra TODAS as features avaliadas neste fold ----
            if audit_writer is not None:
                kernel_label = kernel_name if kernel_name is not None else kernel_str(t)
                for feature_id in sorted(correlations.keys()):
                    if feature_id in suspicious_feature_ids:
                        status = "suspicious_removed"
                    elif feature_id in constant_feature_ids:
                        status = "constant"
                    elif feature_id in selected_set:
                        status = "selected"
                    else:
                        status = "not_selected"

                    audit_row = [
                        kernel_label,
                        cost,
                        gamma,
                        fold_number,
                        feature_id,
                        f"{correlations[feature_id]:.6f}",
                        status,
                    ]
                    audit_writer.writerow(audit_row)
                    if status == "selected":
                        audit_selected_writer.writerow(audit_row)
                    audit_rows_written += 1
            # ---- fim da auditoria do fold ----

            x_tr_sel, remap = apply_feature_selection(x_tr, selected_feature_ids)
            x_te_sel, _ = apply_feature_selection(x_te, selected_feature_ids)

            # ---- Normalização fold-safe (Rodada 6) ----
            # Ajusta o scaler usando APENAS o treino do fold; aplica em treino e teste.
            # Quando normalization == 'none', os dados passam intactos.
            if normalization and normalization != 'none':
                x_tr_sel, x_te_sel, _scaler_info = apply_fold_safe_normalization(
                    x_tr_sel, x_te_sel, normalization
                )
            # ---- fim da normalização ----

            t0 = process_time()
            model = svm_train(y_tr, x_tr_sel, param)
            t1 = process_time()

            _, p_acc_tr, _ = svm_predict(y_tr, x_tr_sel, model, '-q')
            t2 = process_time()
            p_lbl, p_acc_te, _ = svm_predict(y_te, x_te_sel, model, '-q')
            t3 = process_time()

            acc_tr.append(p_acc_tr[0])
            acc_te.append(p_acc_te[0])

            t_tr.append(t1 - t0)
            t_te.append(t3 - t2)

            cm_te = confusion_matrix(
                y_te,
                p_lbl,
                labels=class_labels_order
            )
            cms_te.append(cm_te)

            fold_metrics = calculate_detection_metrics(cm_te)
            if fold_metrics is not None:
                detection_rates.append(fold_metrics["detection_rate"])
                false_positive_rates.append(fold_metrics["false_positive_rate"])
                precisions.append(fold_metrics["precision"])
                f1_scores.append(fold_metrics["f1_score"])
                # Rodada 8.4: métricas complementares
                f1_macros.append(fold_metrics["f1_macro"])
                balanced_accuracies.append(fold_metrics["balanced_accuracy"])
                # Rodada 8.5: MCC
                mcc_scores.append(fold_metrics["mcc"])

            if callback is not None:
                callback()
    finally:
        if audit_file is not None:
            audit_file.close()
        if audit_selected_file is not None:
            audit_selected_file.close()

    return {
        "accuracy_train_mean": float(np.mean(acc_tr)),
        "accuracy_train_std": float(np.std(acc_tr)),
        "accuracy_test_mean": float(np.mean(acc_te)),
        "accuracy_test_std": float(np.std(acc_te)),

        "time_train_mean": float(np.mean(t_tr)),
        "time_train_std": float(np.std(t_tr)),
        "time_test_mean": float(np.mean(t_te)),
        "time_test_std": float(np.std(t_te)),

        "detection_rate_mean": float(np.mean(detection_rates)) if detection_rates else 0.0,
        "detection_rate_std": float(np.std(detection_rates)) if detection_rates else 0.0,

        "false_positive_rate_mean": float(np.mean(false_positive_rates)) if false_positive_rates else 0.0,
        "false_positive_rate_std": float(np.std(false_positive_rates)) if false_positive_rates else 0.0,

        "precision_mean": float(np.mean(precisions)) if precisions else 0.0,
        "precision_std": float(np.std(precisions)) if precisions else 0.0,

        "f1_score_mean": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "f1_score_std": float(np.std(f1_scores)) if f1_scores else 0.0,

        # Rodada 8.4: métricas complementares (não afetam ranking)
        "f1_macro_mean": float(np.mean(f1_macros)) if f1_macros else 0.0,
        "f1_macro_std":  float(np.std(f1_macros))  if f1_macros else 0.0,
        "balanced_accuracy_mean": float(np.mean(balanced_accuracies)) if balanced_accuracies else 0.0,
        "balanced_accuracy_std":  float(np.std(balanced_accuracies))  if balanced_accuracies else 0.0,
        # Rodada 8.5: MCC (não afeta ranking)
        "mcc_mean": float(np.mean(mcc_scores)) if mcc_scores else 0.0,
        "mcc_std":  float(np.std(mcc_scores))  if mcc_scores else 0.0,

        "confusion_matrix_mean": np.mean(cms_te, axis=0),
        "confusion_matrix_std": np.std(cms_te, axis=0),

        "times_train_per_fold": t_tr,
        "times_test_per_fold": t_te,

        "accuracy_train_per_fold": list(acc_tr),
        "accuracy_test_per_fold": list(acc_te),

        "suspicious_feature_count_mean": float(np.mean(suspicious_counts)) if suspicious_counts else 0.0,
        "suspicious_feature_count_max": int(np.max(suspicious_counts)) if suspicious_counts else 0,

        # Rodada 8.3: contagem de features usadas após seleção/auditoria (fold-safe)
        "selected_feature_count_mean": float(np.mean(selected_feature_counts)) if selected_feature_counts else 0.0,
        "selected_feature_count_std":  float(np.std(selected_feature_counts))  if selected_feature_counts else 0.0,

        "audit_rows_written": audit_rows_written,
    }

def model_rank_key(result_data):
    return (
        result_data["f1_score"],
        result_data["detection_rate"],
        -result_data["false_positive_rate"]
    )

class svmParameters():
    def main(self, config: ExperimentConfig):
        y, x = svm_read_problem(config.dataset)
        y = remap_binary_labels(y)
        class_labels_order = infer_class_labels(y)
        print(f"Labels detectados: {class_labels_order}")

        # Rodada 8.3: dimensionalidade original do dataset (maior id de feature
        # observado no arquivo LIBSVM). Registrado apenas para exibição no
        # relatório — não afeta a lógica de treino/teste.
        total_features_original = 0
        for sample in x:
            if sample:
                m = max(sample.keys())
                if m > total_features_original:
                    total_features_original = m

        # Estrutura padrão de auditoria fold-safe (Rodada 4).
        # Substitui a antiga variável vestigial `dirty_count` e os arquivos
        # globalPruned.libsvm / selected_features.csv (pruningDataset removida).
        audit_info = {
            "audit_csv": None,
            "audit_csv_path": None,
            "rows_written": 0,
            # Rodada 8.3
            "total_features_original": total_features_original,
        }

        kernels_run, skip_info, tot_iters = estimate_and_filter_kernels(
            y, x,
            config.kernels,
            config.costs,
            config.gammas,
            config.k_folds,
            max_time=config.max_time,
            force_all_kernels=config.force_all_kernels,
        )
        if not kernels_run:
            # Nenhum kernel será treinado -> não há treino -> não geramos auditoria
            # (evita CSV enganoso sem dados).
            return None, {}, skip_info, audit_info

        # CSV de auditoria fold-safe: header criado UMA vez antes dos folds.
        result_dir = os.path.join(os.getcwd(), "svm-results")
        os.makedirs(result_dir, exist_ok=True)
        audit_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        audit_csv_filename = f"svm_feature_audit_{audit_stamp}.csv"
        audit_csv_path = os.path.join(result_dir, audit_csv_filename)
        audit_selected_filename = f"svm_feature_audit_{audit_stamp}_selected.csv"
        audit_selected_path = os.path.join(result_dir, audit_selected_filename)
        audit_header = [
            "kernel", "C", "gamma", "fold",
            "feature_id_original", "correlation", "status"
        ]
        with open(audit_csv_path, 'w', newline='', encoding='utf-8') as f_csv:
            csv.writer(f_csv).writerow(audit_header)
        with open(audit_selected_path, 'w', newline='', encoding='utf-8') as f_csv:
            csv.writer(f_csv).writerow(audit_header)
        audit_info["audit_csv"] = audit_csv_filename
        audit_info["audit_csv_path"] = audit_csv_path
        audit_info["audit_selected_csv"] = audit_selected_filename
        audit_info["audit_selected_csv_path"] = audit_selected_path
        print(f"📝 Auditoria fold-safe será gravada em: {audit_csv_filename}")

        max_res, min_res = {}, {}
        res = {}

        start_t = time()
        curr_iter = 0

        def cb():
            nonlocal curr_iter
            curr_iter += 1
            print_progress_info(curr_iter, tot_iters, start_t, "Executando...")

        for t in kernels_run:
            runs = []
            gs = get_gammas_for_kernel(t, config.gammas)

            for c in config.costs:
                for g in gs:
                    print_progress_info(
                        curr_iter,
                        tot_iters,
                        start_t,
                        f"K:{kernel_str(t)} C:{c}"
                    )

                    fold_results = svmKfold(
                        y, x, t, c, g, config.k_folds,
                        class_labels_order=class_labels_order,
                        seed=config.seed,
                        cache=config.cache,
                        epsilon=config.epsilon,
                        threshold=config.threshold,
                        leakage_threshold=config.leakage_threshold,
                        normalization=config.normalization,
                        audit_csv_path=audit_csv_path,
                        kernel_name=kernel_str(t),
                        callback=cb
                    )

                    # Acumula linhas de auditoria escritas por este (kernel, C, gamma)
                    audit_info["rows_written"] += fold_results.get("audit_rows_written", 0)

                    sys.stdout.write("\r" + " " * 120 + "\r")

                    dm = calculate_detection_metrics(
                        fold_results["confusion_matrix_mean"]
                    )

                    print(
                        f"✅ Concluído: {kernel_str(t)} | "
                        f"C={c} | G={g} | "
                        f"Acc: {fold_results['accuracy_test_mean']:.2f}% | "
                        f"F1: {fold_results['f1_score_mean']:.4f}"
                    )

                    if dm:
                        print_detection_metrics(dm, f"Resultados {kernel_str(t)}")

                    diag = diagnose_svm_performance(
                        {
                            "accuracy_train": fold_results["accuracy_train_mean"],
                            "accuracy_test": fold_results["accuracy_test_mean"],
                            "std_test": fold_results["accuracy_test_std"],
                            "f1_score": fold_results["f1_score_mean"],
                            "std_f1_score": fold_results["f1_score_std"],
                            "detection_rate": fold_results["detection_rate_mean"],
                            "std_detection_rate": fold_results["detection_rate_std"],
                            "false_positive_rate": fold_results["false_positive_rate_mean"],
                            "std_false_positive_rate": fold_results["false_positive_rate_std"],
                            "precision": fold_results["precision_mean"],
                            "std_precision": fold_results["precision_std"],
                            "confusion_matrix_test": fold_results["confusion_matrix_mean"],
                        },
                        kernel_str(t)
                    )

                    std_class, std_short, std_tooltip = classify_test_std(
                        fold_results["accuracy_test_std"]
                    )
                    if config.language == "en":
                        std_short = {"low": "Low", "medium": "Moderate", "high": "High"}[std_class]
                        std_tooltip = {
                            "low": "Low standard deviation (< 2%) — highly stable results across folds.",
                            "medium": "Moderate standard deviation (2%–10%) — acceptable variance and mild sensitivity to the samples.",
                            "high": "High standard deviation (> 10%) — unstable performance across folds; possible overfitting in parts of the data.",
                        }[std_class]

                    dat = {
                        "accuracy_train": fold_results["accuracy_train_mean"],
                        "std_train": fold_results["accuracy_train_std"],

                        "accuracy_test": fold_results["accuracy_test_mean"],
                        "std_test": fold_results["accuracy_test_std"],

                        "detection_rate": fold_results["detection_rate_mean"],
                        "std_detection_rate": fold_results["detection_rate_std"],

                        "false_positive_rate": fold_results["false_positive_rate_mean"],
                        "std_false_positive_rate": fold_results["false_positive_rate_std"],

                        "precision": fold_results["precision_mean"],
                        "std_precision": fold_results["precision_std"],

                        "f1_score": fold_results["f1_score_mean"],
                        "std_f1_score": fold_results["f1_score_std"],

                        # Rodada 8.4: métricas complementares (não afetam ranking)
                        "f1_macro": fold_results["f1_macro_mean"],
                        "std_f1_macro": fold_results["f1_macro_std"],
                        "balanced_accuracy": fold_results["balanced_accuracy_mean"],
                        "std_balanced_accuracy": fold_results["balanced_accuracy_std"],
                        # Rodada 8.5: MCC (não afeta ranking)
                        "mcc": fold_results["mcc_mean"],
                        "std_mcc": fold_results["mcc_std"],

                        "std_class": std_class,
                        "std_short": std_short,
                        "std_tooltip": std_tooltip,

                        "time_train": fold_results["time_train_mean"],
                        "std_time_train": fold_results["time_train_std"],

                        "time_test": fold_results["time_test_mean"],
                        "std_time_test": fold_results["time_test_std"],

                        "cost": c,
                        "gamma": g,

                        "confusion_matrix_test": fold_results["confusion_matrix_mean"],
                        "confusion_matrix_std": fold_results["confusion_matrix_std"],

                        "diagnosis": diag,

                        "times_train": fold_results["times_train_per_fold"],
                        "times_test": fold_results["times_test_per_fold"],

                        "accuracies_train": fold_results["accuracy_train_per_fold"],
                        "accuracies_test": fold_results["accuracy_test_per_fold"],

                        "suspicious_feature_count_mean": fold_results["suspicious_feature_count_mean"],
                        "suspicious_feature_count_max": fold_results["suspicious_feature_count_max"],

                        # Rodada 8.3: contagem fold-safe de features selecionadas
                        "selected_feature_count_mean": fold_results["selected_feature_count_mean"],
                        "selected_feature_count_std":  fold_results["selected_feature_count_std"],
                    }

                    runs.append(dat)

                    # Melhor/pior resultado agora não depende só de acurácia
                    if not max_res or model_rank_key(dat) > model_rank_key(max_res):
                        max_res = {**dat, "kernel_id": t}

                    if not min_res or model_rank_key(dat) < model_rank_key(min_res):
                        min_res = {**dat, "kernel_id": t}

            if runs:
                res[t] = {
                    "max_test": max(runs, key=model_rank_key),
                    "min_test": min(runs, key=model_rank_key)
                }

        print_progress_info(tot_iters, tot_iters, start_t, "Pronto!")
        print("\n")

        glob = {"max": max_res, "min": min_res}
        return glob, res, skip_info, audit_info


def plot_and_save_time_boxplot(values, title, filename, ylabel="Tempo (s)"):
    """Boxplot genérico de uma série única (uma caixa).

    Mantém o nome legado por compatibilidade, mas agora aceita `ylabel`
    para que possa ser reusado para acurácia, tempo, etc.
    """
    plt.figure(figsize=(6, 4))
    plt.boxplot(values, vert=True, medianprops=dict(color='darkred', linewidth=2))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparative_boxplot(series_dict, title, filename, ylabel="Valor"):
    """
    Gera boxplot comparativo entre séries associadas a diferentes labels.

    Pode ser usado para comparar métricas por kernel, como tempo, acurácia
    ou outras séries calculadas ao longo dos folds.

    Rodada 7.1A: aplica preenchimento em cor suave por kernel quando o label
    é reconhecido em KERNEL_COLORS. Bordas em cinza escuro discreto. Sem
    legenda — o eixo X já identifica os kernels.

    Parâmetros
    ----------
    series_dict : dict
        Dicionário no formato:
            {
                "Linear": [valores_fold_1, valores_fold_2, ...],
                "Polinomial": [...],
                ...
            }
    title : str
        Título da figura.
    filename : str
        Caminho do arquivo de saída.
    ylabel : str, opcional
        Rótulo do eixo Y.
    """
    valid_items = [
        (label, values)
        for label, values in series_dict.items()
        if values is not None and len(values) > 0
    ]

    if not valid_items:
        return

    labels = [label for label, _ in valid_items]
    data = [values for _, values in valid_items]

    fig_width = max(7.0, 1.6 * len(labels))
    plt.figure(figsize=(fig_width, 4.6))

    # Cor padrão para labels desconhecidos (mantém consistência visual)
    DEFAULT_BOX_FILL = "#CFCFCF"
    BOX_EDGE         = "#444444"  # cinza escuro discreto
    MEDIAN_COLOR     = "#222222"  # mediana visível
    WHISKER_COLOR    = "#444444"
    CAP_COLOR        = "#444444"
    FLIER_COLOR      = "#777777"

    box_plot = plt.boxplot(
        data,
        tick_labels=labels,
        vert=True,
        patch_artist=True,  # habilita preenchimento das caixas
        medianprops={"linewidth": 2, "color": MEDIAN_COLOR},
        boxprops={"edgecolor": BOX_EDGE, "linewidth": 1.0},
        whiskerprops={"color": WHISKER_COLOR, "linewidth": 1.0},
        capprops={"color": CAP_COLOR, "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "none",
            "markeredgecolor": FLIER_COLOR,
            "linewidth": 0.8,
        },
    )

    # Preenchimento por kernel (cor consistente entre todos os 4 boxplots)
    # Paleta categórica pastel inspirada na referência (azul, lilás, cinza,
    # verde-água e ciano). A amostragem dinâmica evita repetir poucas cores
    # quando a execução contém mais kernels.
    palette_stops = [
        "#7778D8", "#A59ADE", "#B8C2C9", "#82B8C8",
        "#73D5D2", "#59D9E7", "#76D8F2", "#89B8EA",
    ]
    palette = LinearSegmentedColormap.from_list(
        "svm_pastel_categories", palette_stops, N=max(len(labels), 1)
    )
    dynamic_colors = {
        label: palette(i / max(len(labels) - 1, 1))
        for i, label in enumerate(labels)
    }
    for patch, label in zip(box_plot["boxes"], labels):
        patch.set_facecolor(dynamic_colors.get(label, DEFAULT_BOX_FILL))
        patch.set_alpha(0.85)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

#========================================================================
# GERAÇÃO DO RELATÓRIO HTML
#========================================================================

def generate_html_report(global_results, kernel_results, skipped_info, audit_info, dataset_name, config=None):
    """
    Gera o relatório HTML usando Jinja2 + template externo.

    Estrutura esperada (ambos os idiomas na pasta atual):
      - templates/svm_report_template_ptbr.html.j2
      - templates/svm_report_template_en.html.j2
      - static/svm_report_styles.css
      - svm_report_images_<timestamp>/ (gerado automaticamente)

    IMPORTANTE:
      - Paths passados ao HTML são RELATIVOS (compatíveis com navegador).
      - O template deve usar {{ css_path }} e os caminhos de imagens enviados ao contexto.
      - config (ExperimentConfig) é opcional para retro-compatibilidade,
        mas o template PT-BR atual espera todos os campos da Rodada 3.
      - audit_info (Rodada 4) é um dict com:
          {"audit_csv": basename|None,
           "audit_csv_path": absolute_path|None,
           "rows_written": int}
        Substitui o antigo dirty_count vestigial.
    """

    if not global_results or not global_results.get('max'):
        return

    language = config.language if config is not None else "pt-br"
    is_english = language == "en"

    def kernel_display(kernel_id):
        if not is_english:
            return kernel_str(kernel_id)
        return {0: "Linear", 1: "Polynomial", 2: "Radial Basis Function", 3: "Sigmoid"}.get(kernel_id, "Unknown")

    def report_text(text):
        if not is_english:
            return text
        return {
            "Cenário Avaliado": "Evaluated Scenario",
            "Melhor Global": "Best Overall",
            "Pior Global": "Worst Overall",
            "MC Média - Cenário Avaliado": "Mean Confusion Matrix - Evaluated Scenario",
            "MC Média - Melhor Global": "Mean Confusion Matrix - Best Overall",
            "MC Média - Pior Global": "Mean Confusion Matrix - Worst Overall",
            "MC Média - Melhor Teste": "Mean Confusion Matrix - Best Test",
            "MC Média - Pior Teste": "Mean Confusion Matrix - Worst Test",
            "Comparativo da Acurácia de Treino - Melhores Cenários por Kernel": "Training Accuracy Comparison - Best Scenarios by Kernel",
            "Comparativo da Acurácia de Teste - Melhores Cenários por Kernel": "Test Accuracy Comparison - Best Scenarios by Kernel",
            "Comparativo do Tempo de Treino - Melhores Cenários por Kernel": "Training Time Comparison - Best Scenarios by Kernel",
            "Comparativo do Tempo de Teste - Melhores Cenários por Kernel": "Test Time Comparison - Best Scenarios by Kernel",
            "Acurácia (%)": "Accuracy (%)",
        }.get(text, text)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(os.getcwd(), "svm-results")
    os.makedirs(result_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"svm_report_{'en_' if is_english else ''}{stamp}.html"
    #output_file = f"svm_report_.html"
    output_path = os.path.join(result_dir, output_file)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S" if is_english else "%d/%m/%Y %H:%M:%S")


    # Diretórios locais do script de classificação.
    template_dir = os.path.join(script_dir, 'templates')
    image_dir_name = f"svm_report_images_{'en_' if is_english else ''}{stamp}"
    img_dir = os.path.join(result_dir, image_dir_name)
    os.makedirs(img_dir, exist_ok=True)

    def img_disk(name: str) -> str:
        return os.path.join(img_dir, name)

    def img_href(name: str) -> str:
        # Path que entra no HTML
        return f"{image_dir_name}/{name}"

    # Raiz do projeto (um nível acima da pasta classification).
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Logo compartilhada na raiz
    logo_abs = os.path.join(repo_root, "src", "ufpe_logo.png")
    logo_path = relpath_from_output(logo_abs, output_path)
    logo_upe_path = relpath_from_output(
        os.path.join(template_dir, "20141104181643!Logo-upe-site.png"), output_path
    )
    logo_ufpe_path = relpath_from_output(
        os.path.join(template_dir, "Brasão-Sigla---PNG---RGB.png"), output_path
    )
 
    # CSS local (dentro da pasta do script)
    # CSS local (dentro da pasta do script)
    # (versão com indicador de variabilidade/STD)
    css_path = relpath_from_output(
        os.path.join(script_dir, "static", "svm_report_styles.css"), output_path
    )


    # ------------------------------------------------------------
    # Rodada 7.3C.1: detecção de cenário único calculada AQUI,
    # antes dos plot_and_save_cm, para que os títulos visíveis das
    # matrizes PNG também usem 'Cenário Avaliado' quando aplicável.
    # ------------------------------------------------------------
    kernels_run_ids = sorted(kernel_results.keys()) if kernel_results else []
    is_single_scenario = False
    if config is not None and len(kernels_run_ids) == 1:
        only_kernel = kernels_run_ids[0]
        effective_gammas = get_gammas_for_kernel(only_kernel, config.gammas)
        is_single_scenario = (
            len(config.costs) == 1
            and len(effective_gammas) == 1
        )

    # -----------------------------
    # 1) Plot das matrizes globais
    # -----------------------------
    cm_global_best = None
    if global_results['max'].get('confusion_matrix_test') is not None:
        global_best_cm_title = (
            f"{report_text('MC Média - Cenário Avaliado')} ({kernel_display(global_results['max']['kernel_id'])})"
            if is_single_scenario
            else f"{report_text('MC Média - Melhor Global')} ({kernel_display(global_results['max']['kernel_id'])})"
        )
        plot_and_save_cm(
            global_results['max']['confusion_matrix_test'],
            global_results['max'].get('confusion_matrix_std'),
            global_best_cm_title,
            img_disk('cm_global_best.png'), language
        )
        cm_global_best = img_href('cm_global_best.png')

    cm_global_worst = None
    if global_results['min'].get('confusion_matrix_test') is not None:
        plot_and_save_cm(
            global_results['min']['confusion_matrix_test'],
            global_results['min'].get('confusion_matrix_std'),
            f"{report_text('MC Média - Pior Global')} ({kernel_display(global_results['min']['kernel_id'])})",
            img_disk('cm_global_worst.png'), language
        )
        cm_global_worst = img_href('cm_global_worst.png')

    # -----------------------------------
    # 2) Plot por kernel + diagnósticos
    # -----------------------------------
    act_cm = {}
    diagnostics_dict = {}

    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        k_key = k_name.replace(" ", "_")
        best_img, worst_img = None, None

        if data.get('max_test'):
            kernel_best_cm_title = (
                f"{report_text('MC Média - Cenário Avaliado')} ({k_name})"
                if is_single_scenario
                else f"{report_text('MC Média - Melhor Teste')} ({k_name})"
            )
            plot_and_save_cm(
                data['max_test']['confusion_matrix_test'],
                data['max_test'].get('confusion_matrix_std'),
                kernel_best_cm_title,
                img_disk(f"cm_{k_key}_best.png"), language
            )
            best_img = img_href(f"cm_{k_key}_best.png")

            if 'diagnosis' in data['max_test']:
                diagnostics_dict[k_name] = localize_diagnostic(data['max_test']['diagnosis'], language)

        if data.get('min_test'):
            plot_and_save_cm(
                data['min_test']['confusion_matrix_test'],
                data['min_test'].get('confusion_matrix_std'),
                f"{report_text('MC Média - Pior Teste')} ({k_name})",
                img_disk(f"cm_{k_key}_worst.png"), language
            )
            worst_img = img_href(f"cm_{k_key}_worst.png")

        act_cm[k_name] = {'best': best_img, 'worst': worst_img}

    # Kernels pulados (estimativa)
    for k_id, info in skipped_info.items():
        k_name = kernel_display(k_id)
        diagnostics_dict[k_name] = {
            'status':          'skipped',
            'status_label':    'Skipped' if is_english else DIAGNOSTIC_STATUS.get('skipped', 'Pulado'),
            'problem':         'Skipped based on time estimate' if is_english else info['reason'],
            'criterion':       'Estimated time exceeds the configured limit.' if is_english else 'Tempo estimado acima do limite configurado.',
            'evidence':        [info['detail'].replace("Est:", "Estimate:").replace("limite", "limit") if is_english else f"{info['detail']}"],
            'explanation':     f"Skipped: {info['detail'].replace('Est:', 'Estimate:').replace('limite', 'limit')}." if is_english else f"Pulado: {info['detail']}.",
            'interpretation':  'The kernel was not run in this experiment due to the operational time criterion.' if is_english else 'O kernel não foi executado nesta rodada por critério operacional de tempo.',
            'recommendation':  'Consider --force-all-kernels, reducing the grid or number of folds, or increasing -max-time in a subsequent run.' if is_english else 'Avaliar --force-all-kernels, redução do grid, redução do número de folds ou aumento de -max-time em nova execução.',
            'thresholds_used': ['max_time'],
        }

    # ------------------------------------------------
    # 3) Preparar campos esperados pelo template HTML
    # ------------------------------------------------
    global_results['max']['act'] = kernel_display(global_results['max']['kernel_id'])
    global_results['max']['n_hidden'] = f"C={global_results['max']['cost']}, γ={global_results['max']['gamma']}"
    global_results['min']['act'] = kernel_display(global_results['min']['kernel_id'])
    global_results['min']['n_hidden'] = f"C={global_results['min']['cost']}, γ={global_results['min']['gamma']}"

    act_results = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        act_results[k_name] = data

        if data.get('max_test'):
            data['max_test']['n_hidden'] = f"C={data['max_test']['cost']}, γ={data['max_test']['gamma']}"
        if data.get('min_test'):
            data['min_test']['n_hidden'] = f"C={data['min_test']['cost']}, γ={data['min_test']['gamma']}"

    # -------------------
    # 4) Render do template
    # -------------------
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )


    # -----------------------------
    # -----------------------------------------------------------------------
    # Boxplots COMPARATIVOS de Acurácia entre os Melhores Cenários por Kernel
    # (Rodada 6.2: substitui o boxplot single-box do melhor global)
    # -----------------------------------------------------------------------
    comparative_boxplot_acc_train = None
    comparative_acc_train_series = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        if data.get("max_test") and data["max_test"].get("accuracies_train"):
            comparative_acc_train_series[k_name] = data["max_test"]["accuracies_train"]

    if comparative_acc_train_series:
        plot_comparative_boxplot(
            comparative_acc_train_series,
            report_text("Comparativo da Acurácia de Treino - Melhores Cenários por Kernel"),
            img_disk("comparative_boxplot_acc_train.png"),
            ylabel=report_text("Acurácia (%)"),
        )
        comparative_boxplot_acc_train = img_href("comparative_boxplot_acc_train.png")

    comparative_boxplot_acc_test = None
    comparative_acc_test_series = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        if data.get("max_test") and data["max_test"].get("accuracies_test"):
            comparative_acc_test_series[k_name] = data["max_test"]["accuracies_test"]

    if comparative_acc_test_series:
        plot_comparative_boxplot(
            comparative_acc_test_series,
            report_text("Comparativo da Acurácia de Teste - Melhores Cenários por Kernel"),
            img_disk("comparative_boxplot_acc_test.png"),
            ylabel=report_text("Acurácia (%)"),
        )
        comparative_boxplot_acc_test = img_href("comparative_boxplot_acc_test.png")

    # --------------------------------------------------------
    # Comparativo entre os melhores cenários de cada kernel
    # --------------------------------------------------------
    comparative_boxplot_time_train = None
    comparative_train_series = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        if data.get("max_test") and data["max_test"].get("times_train"):
            comparative_train_series[k_name] = data["max_test"]["times_train"]

    if comparative_train_series:
        plot_comparative_boxplot(
            comparative_train_series,
            report_text("Comparativo do Tempo de Treino - Melhores Cenários por Kernel"),
            img_disk("comparative_boxplot_time_train.png"),
            ylabel="Time (s)" if is_english else "Tempo (s)",
        )
        comparative_boxplot_time_train = img_href("comparative_boxplot_time_train.png")

    comparative_boxplot_time_test = None
    comparative_test_series = {}
    for k_id, data in kernel_results.items():
        k_name = kernel_display(k_id)
        if data.get("max_test") and data["max_test"].get("times_test"):
            comparative_test_series[k_name] = data["max_test"]["times_test"]

    if comparative_test_series:
        plot_comparative_boxplot(
            comparative_test_series,
            report_text("Comparativo do Tempo de Teste - Melhores Cenários por Kernel"),
            img_disk("comparative_boxplot_time_test.png"),
            ylabel="Time (s)" if is_english else "Tempo (s)",
        )
        comparative_boxplot_time_test = img_href("comparative_boxplot_time_test.png")

    template_name = "svm_report_template_en.html.j2" if is_english else "svm_report_template_ptbr.html.j2"
    template = env.get_template(template_name)

    # ----------------------------------------------------------------
    # Dados derivados para a seção "Configuração Experimental" (Rodada 3)
    # ----------------------------------------------------------------
    if config is not None:
        kernels_requested_str = format_list([kernel_display(k) for k in config.kernels], "None" if is_english else "Nenhum")
        costs_str = format_list(config.costs)
        gammas_str = format_list(config.gammas)
        threshold_str = format_optional(config.threshold, "Not applied" if is_english else "Não aplicado")
    else:
        kernels_requested_str = "—"
        costs_str = "—"
        gammas_str = "—"
        threshold_str = "—"

    # kernels_run_ids já foi calculado mais acima (Rodada 7.3C.1)
    kernels_run_str = format_list([kernel_display(k) for k in kernels_run_ids], "None" if is_english else "Nenhum")

    kernels_skipped_ids = sorted(skipped_info.keys()) if skipped_info else []
    kernels_skipped_str = format_list([kernel_display(k) for k in kernels_skipped_ids], "None" if is_english else "Nenhum")

    # Rodada 8.2: o detalhamento do relatório passou a ser controlado SOMENTE
    # pela flag --full-report (config.full_report). A quantidade de kernels
    # NÃO controla mais o nível de detalhe. Por padrão, relatório é compacto.
    kernels_full_metrics = bool(config is not None and config.full_report)

    # Rodada 6: rótulo amigável da normalização ('none' -> 'nenhuma', etc.)
    if config is not None:
        normalization_display = (
            {"none": "None", "maxabs": "MaxAbs", "minmax": "Min-Max"}.get(config.normalization, config.normalization)
            if is_english else NORMALIZATION_DISPLAY.get(config.normalization, config.normalization)
        )
    else:
        normalization_display = '—'

    # ------------------------------------------------------------
    # Rodada 8.3: resumo do pré-processamento fold-safe para o cenário em
    # destaque (global_results['max']). Cobre tanto o "cenário avaliado" em
    # execuções de cenário único quanto o "melhor cenário global" em execuções
    # com múltiplos cenários — ambos são resolvidos por global_results['max'].
    # Não altera treino/teste; apenas registra o que já foi calculado.
    # ------------------------------------------------------------
    _highlight = global_results.get('max') if global_results else None
    _sel_mean  = _highlight.get('selected_feature_count_mean', 0.0) if _highlight else 0.0
    _sel_std   = _highlight.get('selected_feature_count_std',  0.0) if _highlight else 0.0
    _total_orig = audit_info.get('total_features_original', 0) if audit_info else 0
    _norm_applied = bool(config is not None and config.normalization != 'none')

    preprocessing_summary = {
        "normalization_method": normalization_display,
        "normalization_applied": _norm_applied,
        "total_features_original": _total_orig,
        "features_after_selection_mean": _sel_mean,
        "features_after_selection_std":  _sel_std,
        # Após a seleção/auditoria, exatamente as MESMAS features são passadas
        # ao scaler no fold (quando normalization != 'none'). Portanto a
        # contagem de features normalizadas é idêntica à selecionada.
        "features_normalized_mean": _sel_mean if _norm_applied else 0.0,
        "features_normalized_std":  _sel_std  if _norm_applied else 0.0,
    }

    # is_single_scenario já foi calculado mais acima (Rodada 7.3C.1)

    rendered = template.render(
        global_results=global_results,
        act_results=act_results,
        cm_global_best=cm_global_best,
        cm_global_worst=cm_global_worst,
        act_cm=act_cm,
        diagnostics=diagnostics_dict,
        audit_info=audit_info,
        css_path=css_path,
        logo_path=logo_path,
        logo_upe_path=logo_upe_path,
        logo_ufpe_path=logo_ufpe_path,
        generated_at=generated_at,
        comparative_boxplot_acc_train=comparative_boxplot_acc_train,
        comparative_boxplot_acc_test=comparative_boxplot_acc_test,
        comparative_boxplot_time_train=comparative_boxplot_time_train,
        comparative_boxplot_time_test=comparative_boxplot_time_test,
        dataset_name=dataset_name,
        # --- novos: Rodada 3 ---
        config=config,
        kernels_requested_str=kernels_requested_str,
        kernels_run_str=kernels_run_str,
        kernels_skipped_str=kernels_skipped_str,
        costs_str=costs_str,
        gammas_str=gammas_str,
        threshold_str=threshold_str,
        # --- novos: Rodada 5 ---
        kernels_full_metrics=kernels_full_metrics,
        # --- novos: Rodada 6 ---
        normalization_display=normalization_display,
        # --- novos: Rodada 8.3 ---
        preprocessing_summary=preprocessing_summary,
        # --- novos: Rodada 7.3C ---
        is_single_scenario=is_single_scenario,
        language=language,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"✅ Relatório HTML gerado: {output_path}")

#========================================================================
# CLI
#========================================================================

class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    """
    Rodada 7.3B: subclasse leve de RawTextHelpFormatter que traduz o prefixo
    'usage:' para 'uso:' no início do --help, preservando todo o restante do
    comportamento do RawTextHelpFormatter (quebras de linha, indentação, etc.).
    """
    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "uso: "
        return super()._format_usage(usage, actions, groups, prefix)


def setOpts(argv):
    parser = argparse.ArgumentParser(
        prog="svmParameters.py",
        description=(
            "Avaliação de parâmetros SVM com validação cruzada, auditoria fold-safe, "
            "normalização opcional e geração de relatório HTML.\n\n"
            "O script aceita datasets no formato LIBSVM e compara configurações de kernel, "
            "C e gamma usando métricas como acurácia, F1-score, recall/taxa de detecção "
            "e taxa de falsos positivos."
        ),
        formatter_class=CustomHelpFormatter,
        epilog=(
            "Exemplos de uso:\n"
            "  python svmParameters.py --demo\n"
            "  python svmParameters.py --demo --quick-test\n"
            "  python svmParameters.py -dataset dados.libsvm -kernels linear,rbf -costs 1,10,100 -gammas 0.01,0.1,1\n"
            "  python svmParameters.py -dataset dados.libsvm -kernels all -kfold 10 -seed 42 -normalization maxabs\n"
            "  python svmParameters.py -dataset dados.libsvm -kernels all -max-time 7200 --force-all-kernels\n\n"
            "  python svmParameters.py -dataset dados.libsvm --language en\n\n"
            "Observações:\n"
            "  - Use --demo para validar rapidamente a instalação com o dataset heart_scale.\n"
            "  - Use --quick-test para uma execução curta de teste funcional.\n"
            "  - Para experimentos finais, informe explicitamente -dataset e revise kernels, C, gamma, normalização, k-fold e seed.\n"
            "  - No kernel Linear, gamma é ignorado pelo LIBSVM e registrado internamente como γ=0 no relatório.\n"
        ),
        add_help=False
    )

    # ------------------------------------------------------------
    # Grupos de argumentos
    # ------------------------------------------------------------
    ajuda            = parser.add_argument_group("Ajuda")
    entrada          = parser.add_argument_group("Entrada de dados")
    validacao        = parser.add_argument_group("Validação e reprodutibilidade")
    preprocessamento = parser.add_argument_group("Pré-processamento fold-safe")
    grid             = parser.add_argument_group("Grid de hiperparâmetros")
    libsvm           = parser.add_argument_group("Parâmetros LIBSVM")
    execucao         = parser.add_argument_group("Controle de execução")

    # ------------------------------------------------------------
    # Ajuda (Rodada 7.3B: tradução do -h/--help padrão para PT-BR)
    # ------------------------------------------------------------
    ajuda.add_argument(
        '-h', '--help',
        action='help',
        help='Exibe esta mensagem de ajuda e encerra.'
    )

    # ------------------------------------------------------------
    # Entrada de dados
    # ------------------------------------------------------------
    entrada.add_argument(
        '-dataset',
        dest='dataset',
        default=None,
        metavar='ARQUIVO',
        help=(
            'Caminho do dataset no formato LIBSVM.\n'
            'Obrigatório quando --demo não for usado.'
        )
    )
    entrada.add_argument(
        '--demo',
        dest='demo',
        action='store_true',
        help=(
            'Executa com o dataset de demonstração heart_scale.\n'
            'Útil para validar instalação, dependências e geração do relatório.'
        )
    )

    # ------------------------------------------------------------
    # Validação e reprodutibilidade
    # ------------------------------------------------------------
    validacao.add_argument(
        '-kfold',
        dest='k_folds',
        type=int,
        default=10,
        metavar='K',
        help=(
            'Número de folds da validação cruzada estratificada.\n'
            'Deve ser >= 2. Default: 10.'
        )
    )
    validacao.add_argument(
        '-seed',
        dest='seed',
        type=int,
        default=1,
        metavar='N',
        help=(
            'Semente aleatória usada no particionamento dos folds.\n'
            'Use o mesmo valor para melhorar a reprodutibilidade. Default: 1.'
        )
    )

    # ------------------------------------------------------------
    # Pré-processamento fold-safe
    # ------------------------------------------------------------
    preprocessamento.add_argument(
        '-threshold',
        dest='threshold',
        type=float,
        default=None,
        metavar='VALOR',
        help=(
            'Limiar opcional de seleção por correlação absoluta com o rótulo.\n'
            'Se omitido, não aplica seleção por limiar, mantendo a auditoria fold-safe.\n'
            'Faixa esperada: 0 a 1. Default: não aplicado.'
        )
    )
    preprocessamento.add_argument(
        '-leakage-threshold',
        dest='leakage_threshold',
        type=float,
        default=0.999,
        metavar='VALOR',
        help=(
            'Limiar para marcar features como suspeitas de vazamento de dados.\n'
            'Features com correlação absoluta >= a este valor são removidas dentro do fold.\n'
            'Faixa esperada: (0, 1]. Default: 0.999.'
        )
    )
    preprocessamento.add_argument(
        '-normalization', '--normalization',
        dest='normalization',
        type=str.lower,
        choices=list(VALID_NORMALIZATIONS),
        default='none',
        metavar='MÉTODO',
        help=(
            'Normalização fold-safe aplicada após a seleção de features dentro de cada fold.\n'
            'Opções: none, maxabs, minmax. Default: none.\n'
            'maxabs preserva melhor a esparsidade; minmax facilita comparação com literatura.'
        )
    )

    # ------------------------------------------------------------
    # Grid de hiperparâmetros
    # ------------------------------------------------------------
    grid.add_argument(
        '-kernels', '-kernel', '--kernel',
        dest='kernels',
        type=str,
        default='all',
        metavar='LISTA',
        help=(
            'Kernels a avaliar, separados por vírgula.\n'
            'Opções: linear, polinomial, rbf, sigmoid ou all.\n'
            'Exemplo: -kernels linear,rbf. Default: all.\n'
            'Aliases aceitos: -kernel e --kernel.'
        )
    )
    grid.add_argument(
        '-costs',
        dest='costs',
        type=str,
        default='1,1000',
        metavar='LISTA',
        help=(
            'Lista de valores de C separados por vírgula.\n'
            'C controla a regularização do SVM.\n'
            'Exemplo: -costs 1,10,100. Default: 1,1000.'
        )
    )
    grid.add_argument(
        '-gammas',
        dest='gammas',
        type=str,
        default=None,
        metavar='LISTA',
        help=(
            'Lista de valores de gamma separados por vírgula.\n'
            'Usado em kernels não lineares. No kernel Linear, gamma é ignorado pelo LIBSVM.\n'
            'Exemplo: -gammas 0.01,0.1,1. Default: 1.'
        )
    )

    # ------------------------------------------------------------
    # Parâmetros LIBSVM
    # ------------------------------------------------------------
    libsvm.add_argument(
        '-cache',
        dest='cache',
        type=int,
        default=500,
        metavar='MB',
        help=(
            'Cache do LIBSVM em MB.\n'
            'Valores maiores podem ajudar em bases maiores. Deve ser > 0. Default: 500.'
        )
    )
    libsvm.add_argument(
        '-epsilon',
        dest='epsilon',
        type=float,
        default=0.05,
        metavar='VALOR',
        help=(
            'Tolerância de parada do LIBSVM.\n'
            'Valores menores podem aumentar o tempo de treino. Deve ser > 0. Default: 0.05.'
        )
    )

    # ------------------------------------------------------------
    # Controle de execução
    # ------------------------------------------------------------
    execucao.add_argument(
        '-max-time',
        dest='max_time',
        type=float,
        default=3600.0,
        metavar='SEGUNDOS',
        help=(
            'Limite em segundos usado pela estimativa operacional de tempo.\n'
            'Kernels acima desse limite podem não ser executados, salvo com --force-all-kernels.\n'
            'Default: 3600.'
        )
    )
    execucao.add_argument(
        '--force-all-kernels',
        dest='force_all_kernels',
        action='store_true',
        help=(
            'Executa todos os kernels solicitados, mesmo quando a estimativa de tempo excede -max-time.\n'
            'Use com cautela em bases grandes.'
        )
    )
    execucao.add_argument(
        '--quick-test',
        dest='quick_test',
        action='store_true',
        help=(
            'Executa uma configuração mínima para teste funcional:\n'
            'k_folds=3, kernels=linear, costs=[1], gammas=[1].\n'
            'Não deve ser usado como experimento final.'
        )
    )
    execucao.add_argument(
        '--full-report',
        dest='full_report',
        action='store_true',
        help=(
            'Exibe métricas completas no relatório HTML, incluindo F1-score,\n'
            'taxa de detecção, taxa de falsos positivos, precisão, F1-Macro,\n'
            'Balanced Accuracy e MCC.\n'
            'Por padrão, o relatório é compacto.\n'
            'Independente da quantidade de kernels executados.'
        )
    )
    execucao.add_argument(
        '--language',
        dest='language',
        type=str.lower,
        choices=('pt-br', 'en'),
        default='pt-br',
        metavar='IDIOMA',
        help='Idioma do relatório HTML: pt-br (padrão) ou en. Não altera o idioma do terminal nem dos CSVs.',
    )

    args = parser.parse_args(argv)

    # --- Dataset ---
    if args.demo:
        dataset = 'heart_scale'
        print("⚠️  Modo DEMO ativado — usando dataset de exemplo: heart_scale")
    else:
        if not args.dataset:
            print("❌ Erro: forneça -dataset ou use --demo.")
            sys.exit(1)
        dataset = args.dataset
        if not os.path.isfile(dataset):
            print(f"❌ Erro: dataset não encontrado: {dataset}")
            sys.exit(1)

    gammas_user_provided = args.gammas is not None

    # --- Grid (com override do --quick-test) ---
    if args.quick_test:
        print("⚡ Modo QUICK-TEST ativado — k_folds=3, kernels=linear, costs=[1], gammas=[1].")
        k_folds = 3
        kernels = [0]
        costs = [1.0]
        gammas = [1.0]
        gammas_user_provided = False  # quick-test não conta como input do usuário
    else:
        k_folds = args.k_folds
        try:
            kernels = parse_kernels(args.kernels)
            costs = parse_float_list(args.costs, 'costs')
            gammas = parse_float_list(args.gammas, 'gammas') if gammas_user_provided else [1.0]
        except ValueError as e:
            print(f"❌ Erro de parsing: {e}")
            sys.exit(1)

    # --- Validações ---
    def _fail(msg):
        print(f"❌ Erro: {msg}")
        sys.exit(1)

    if k_folds < 2:
        _fail(f"-kfold deve ser >= 2 (recebido: {k_folds}).")
    if args.seed < 0:
        _fail(f"-seed deve ser >= 0 (recebido: {args.seed}).")
    if args.cache <= 0:
        _fail(f"-cache deve ser > 0 (recebido: {args.cache}).")
    if args.epsilon <= 0:
        _fail(f"-epsilon deve ser > 0 (recebido: {args.epsilon}).")
    if any(c <= 0 for c in costs):
        _fail(f"-costs deve conter apenas valores > 0 (recebido: {costs}).")
    if gammas_user_provided and any(g <= 0 for g in gammas):
        _fail(f"-gammas informados pelo usuário devem ser > 0 (recebido: {gammas}).")
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        _fail(f"-threshold deve estar entre 0 e 1 (recebido: {args.threshold}).")
    if not (0.0 < args.leakage_threshold <= 1.0):
        _fail(f"-leakage-threshold deve estar em (0, 1] (recebido: {args.leakage_threshold}).")
    if args.max_time <= 0:
        _fail(f"-max-time deve ser > 0 (recebido: {args.max_time}).")

    # --- Aviso amigável: gammas + somente linear ---
    if gammas_user_provided and kernels == [0]:
        print("⚠️  Aviso: gamma foi informado, mas será ignorado para o kernel linear.")

    return ExperimentConfig(
        dataset=dataset,
        k_folds=k_folds,
        seed=args.seed,
        kernels=kernels,
        costs=costs,
        gammas=gammas,
        cache=args.cache,
        epsilon=args.epsilon,
        threshold=args.threshold,
        leakage_threshold=args.leakage_threshold,
        max_time=args.max_time,
        normalization=args.normalization,
        demo=args.demo,
        quick_test=args.quick_test,
        force_all_kernels=args.force_all_kernels,
        gammas_user_provided=gammas_user_provided,
        full_report=args.full_report,
        language=args.language,
    )

if __name__ == "__main__":
    config = setOpts(sys.argv[1:])
    experiment = svmParameters()
    glob, res, skip, audit_info = experiment.main(config)
    dataset_name = os.path.basename(config.dataset)
    generate_html_report(glob, res, skip, audit_info, dataset_name, config)
