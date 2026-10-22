import pandas as pd
import argparse
import os
import sys
from sklearn.datasets import dump_svmlight_file

def configurar_diretorios(nome_arquivo_saida):
    """
        Salva a saída na pasta 'Antiviruses' ao lado do script,
        independentemente do diretório de onde o comando foi chamado.
    """
    # 1. Get the absolute path where THIS script is located
    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    # 2. Define and create the 'Antiviruses' folder
    pasta_destino = os.path.join(diretorio_script, "Antiviruses")
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        print(f"📁 Pasta criada: {pasta_destino}")

    # 3. Clean previous paths and force save to the correct folder
    nome_base = os.path.basename(nome_arquivo_saida)
    caminho_final = os.path.join(pasta_destino, nome_base)

    return caminho_final

def carregar_dados(caminho_arquivo):
    """
    Loads .csv or .xlsx intelligently, prioritizing the semicolon (;) separator.
    """
    print(f"📖 Lendo arquivo: {caminho_arquivo} ...")

    if not os.path.exists(caminho_arquivo):
        print(f"❌ Erro: o arquivo '{caminho_arquivo}' não existe.")
        sys.exit(1)

    try:
        if caminho_arquivo.lower().endswith('.csv'):
            # Detecta automaticamente o separador do CSV.
            df = pd.read_csv(caminho_arquivo, sep=None, engine='python')
        elif caminho_arquivo.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(caminho_arquivo)
        else:
            print("❌ Error: Unknown format. Please use .csv or .xlsx")
            sys.exit(1)

        # Remove espaços em branco dos nomes das colunas.
        df.columns = df.columns.str.strip()
        return df

    except Exception as e:
        print(f"❌ Erro ao abrir o arquivo: {e}")
        sys.exit(1)

def identificar_coluna_label(df, label_usuario):
    """
    Checks for the exact label column name provided or the assumed default.
    """
    # Check 1: Exact match of the user's argument (or the default argument)
    if label_usuario in df.columns:
        return label_usuario

    # If it fails, check for the known complex name used by this dataset format
    coluna_padrao_conhecida = "class (0:benign, 1:malware)"
    if coluna_padrao_conhecida in df.columns:
        return coluna_padrao_conhecida

    # If both fail, the script fails exactly where the original must have failed.
    print(f"⚠️  A coluna '{label_usuario}' não foi encontrada exatamente.")
    print("❌ Nenhuma coluna de rótulo foi encontrada. Verifique o CSV.")
    print(f"Colunas disponíveis: {list(df.columns)}")
    sys.exit(1)

def converter_para_libsvm(
    df,
    coluna_label,
    caminho_saida,
    rotulo_negativo=None,
    rotulo_positivo=None,
):
    """
    Processes data and saves it in LIBSVM format.
    """
    print("⚙️  Processando os dados...")

    # Separa X (dados) e y (alvo).
    y = df[coluna_label]
    X = df.drop(columns=[coluna_label])

    # 1. Preenche valores ausentes (NaN) com zero.
    X.fillna(0, inplace=True)

    # 2. Converte variáveis categóricas em colunas binárias (one-hot encoding).
    X = pd.get_dummies(X)

    # 3. Converte rótulos textuais usando um mapeamento explícito para evitar
    #    que a ordem de aparição dos registros troque as classes 0 e 1.
    if y.dtype == 'object':
        if rotulo_negativo is None or rotulo_positivo is None:
            raise ValueError(
                "Para rótulos textuais, informe --negative-label e --positive-label "
                "para fixar o mapeamento para 0 e 1."
            )
        rotulos_presentes = set(y.dropna().unique())
        if rotulo_negativo == rotulo_positivo:
            raise ValueError("Os rótulos negativo e positivo devem ser diferentes.")
        if rotulo_negativo not in rotulos_presentes or rotulo_positivo not in rotulos_presentes:
            raise ValueError(
                "Os rótulos informados não foram encontrados na coluna-alvo. "
                f"Rótulos presentes: {sorted(rotulos_presentes)}"
            )
        if y.isna().any():
            raise ValueError("A coluna-alvo contém valores ausentes.")
        y = (y == rotulo_positivo).astype(int)
        print(
            f"ℹ️  Mapeamento aplicado: {rotulo_negativo!r} → 0 (negativo); "
            f"{rotulo_positivo!r} → 1 (positivo)."
        )
    else:
        y = pd.to_numeric(y, errors='raise')
        rotulos_numericos = set(y.unique())
        if rotulos_numericos not in ({0, 1}, {-1, 1}):
            raise ValueError(
                "Rótulos numéricos devem ser 0/1 ou -1/1. "
                f"Rótulos presentes: {sorted(rotulos_numericos)}"
            )

    print(f"📊 Estatísticas: {X.shape[0]} amostras | {X.shape[1]} atributos (features)")

    try:
        dump_svmlight_file(X, y, caminho_saida, zero_based=False)
        print(f"✅ Conversão concluída. Arquivo salvo em:\n   {caminho_saida}")
    except Exception as e:
        print(f"❌ Erro ao salvar o arquivo LIBSVM: {e}")

# --- Main Flow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversor de CSV/Excel para o formato LIBSVM")
    parser.add_argument("input_file", help="Caminho do arquivo de entrada (.csv ou .xlsx)")
    parser.add_argument("output_file", help="Nome do arquivo de saída (.libsvm)")
    parser.add_argument("--label", default="class (0:benign, 1:malware)", help="Nome exato da coluna-alvo (use aspas se contiver espaços)")
    parser.add_argument("--negative-label", default=None, help="Valor textual da classe negativa; será codificado como 0")
    parser.add_argument("--positive-label", default=None, help="Valor textual da classe positiva; será codificado como 1")
    parser.add_argument("--task", default="classification", help="Parâmetro mantido para compatibilidade")

    args = parser.parse_args()

    # 1. Configura o caminho de saída (pasta Antiviruses).
    final_path = configurar_diretorios(args.output_file)

    # 2. Carrega os dados (CSV/Excel com detecção de separador).
    df = carregar_dados(args.input_file)

    # 3. Valida a coluna-alvo.
    coluna_correta = identificar_coluna_label(df, args.label)

    # 4. Converte e salva o arquivo.
    try:
        converter_para_libsvm(
            df,
            coluna_correta,
            final_path,
            rotulo_negativo=args.negative_label,
            rotulo_positivo=args.positive_label,
        )
    except ValueError as erro:
        parser.error(str(erro))
