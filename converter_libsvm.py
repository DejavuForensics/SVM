import pandas as pd
import argparse
import os
import sys
from sklearn.datasets import dump_svmlight_file

def configurar_diretorios(nome_arquivo_saida):
    """
    Ensures the output is always saved in the 'Antiviruses' folder
    next to the script, regardless of where the user calls the command from.
    """
    # 1. Get the absolute path where THIS script is located
    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    # 2. Define and create the 'Antiviruses' folder
    pasta_destino = os.path.join(diretorio_script, "Antiviruses")
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        print(f"📁 Folder created: {pasta_destino}")

    # 3. Clean previous paths and force save to the correct folder
    nome_base = os.path.basename(nome_arquivo_saida)
    caminho_final = os.path.join(pasta_destino, nome_base)

    return caminho_final

def carregar_dados(caminho_arquivo):
    """
    Loads .csv or .xlsx intelligently, prioritizing the semicolon (;) separator.
    """
    print(f"📖 Reading file: {caminho_arquivo} ...")

    if not os.path.exists(caminho_arquivo):
        print(f"❌ Error: The file '{caminho_arquivo}' does not exist.")
        sys.exit(1)

    try:
        if caminho_arquivo.lower().endswith('.csv'):
            # FINAL FIX: Force semicolon separator for this specific format
            df = pd.read_csv(caminho_arquivo, sep=';', engine='python')
        elif caminho_arquivo.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(caminho_arquivo)
        else:
            print("❌ Error: Unknown format. Please use .csv or .xlsx")
            sys.exit(1)

        # Clean whitespaces in column names (CRUCIAL)
        df.columns = df.columns.str.strip()
        return df

    except Exception as e:
        print(f"❌ Critical error opening file: {e}")
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
    print(f"⚠️  The column '{label_usuario}' was not found exactly.")
    print("❌ No label column found. Check your CSV.")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

def converter_para_libsvm(df, coluna_label, caminho_saida):
    """
    Processes data and saves it in LIBSVM format.
    """
    print("⚙️  Processing data...")

    # Separate X (Data) and y (Target)
    y = df[coluna_label]
    X = df.drop(columns=[coluna_label])

    # 1. Fill missing values (NaN) with 0
    X.fillna(0, inplace=True)

    # 2. Convert text to numbers (One-Hot Encoding)
    # Ex: Column "Protocol" (TCP, UDP) becomes "Protocol_TCP" (0,1)
    X = pd.get_dummies(X)

    # 3. Convert Label to numeric if it is text
    # Ex: "benign" becomes 0, "malware" becomes 1
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
        print("ℹ️  Text labels converted to numeric automatically.")

    print(f"📊 Statistics: {X.shape[0]} samples | {X.shape[1]} attributes (features)")

    try:
        dump_svmlight_file(X, y, caminho_saida, zero_based=False)
        print(f"✅ SUCCESS! File saved at:\n   {caminho_saida}")
    except Exception as e:
        print(f"❌ Error saving LIBSVM file: {e}")

# --- Main Flow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV/Excel to LIBSVM Converter (V3.3 Strict)")
    parser.add_argument("input_file", help="Input file path (.csv or .xlsx)")
    parser.add_argument("output_file", help="Output file name (.libsvm)")
    # Set the default to the complex name to meet the "no argument needed" demand for this specific file
    parser.add_argument("--label", default="class (0:benign, 1:malware)", help="Exact name of the target column (Use quotes if it has spaces)")
    parser.add_argument("--task", default="classification", help="Ignored (kept for compatibility)")

    args = parser.parse_args()

    # 1. Setup paths (Antiviruses folder)
    final_path = configurar_diretorios(args.output_file)

    # 2. Load Data (CSV/Excel + Auto Separator)
    df = carregar_dados(args.input_file)

    # 3. Validate Label Column
    coluna_correta = identificar_coluna_label(df, args.label)

    # 4. Convert and Save
    converter_para_libsvm(df, coluna_correta, final_path)
