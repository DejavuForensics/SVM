import pandas as pd
import sys
import os

def csv_to_libsvm(input_file, output_file):
    print(f"   [INFO] Lendo '{input_file}'...")

    # Tenta ler com ; (padrão mELM). Se falhar, tenta com virgula.
    try:
        df = pd.read_csv(input_file, delimiter=';', engine='python')
        if df.shape[1] < 2:
            raise ValueError("Separador parece errado")
    except:
        print("   [AVISO] Falha ao ler com ';'. Tentando ler com ','...")
        df = pd.read_csv(input_file, delimiter=',', engine='python')

    # Remove colunas vazias
    df = df.dropna(axis=1, how='all')

    # Validação básica
    if df.empty or df.shape[1] < 2:
        raise ValueError("O arquivo CSV parece estar vazio ou não tem colunas suficientes.")

    label_map = {0: -1, 1: 1}
    lines_written = 0

    with open(output_file, 'w') as f_out:
        for _, row in df.iterrows():
            try:
                # Tenta pegar a Label da coluna de índice 1
                original_label = int(row.iloc[1])
                mapped_label = label_map.get(original_label, original_label)
            except:
                continue

            line = str(mapped_label)
            # Features da coluna 2 em diante
            for idx in range(2, len(row)):
                value = row.iloc[idx]
                if pd.notna(value) and float(value) != 0:
                    line += f" {idx-1}:{value}"

            f_out.write(line + '\n')
            lines_written += 1

    print(f"   [INFO] {lines_written} amostras processadas.")
    if lines_written == 0:
        raise ValueError("Nenhuma linha foi convertida.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python converter_libsvm.py <arquivo_entrada.csv> <arquivo_saida.libsvm>")
        sys.exit(1)

    # --- Definições de Caminhos ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Pasta onde este script está
    input_arg = sys.argv[1]
    output_filename = os.path.basename(sys.argv[2]) # Pega só o nome do arquivo, ignora pastas anteriores

    # Define a pasta de destino fixa "Antiviruses"
    target_folder = os.path.join(script_dir, "Antiviruses")

    # Garante que a pasta Antiviruses existe
    if not os.path.exists(target_folder):
        try:
            os.makedirs(target_folder)
            print(f"[INFO] Pasta criada: {target_folder}")
        except OSError as e:
            print(f"❌ ERRO: Não foi possível criar a pasta 'Antiviruses'. {e}")
            sys.exit(1)

    # Define caminhos absolutos para entrada e saída
    input_abs = os.path.abspath(input_arg)
    output_abs = os.path.join(target_folder, output_filename)

    print("="*60)
    print("CONVERSOR CSV -> LIBSVM")
    print("="*60)

    try:
        # Executa a conversão já salvando no local correto
        csv_to_libsvm(input_abs, output_abs)

        # Verifica se criou mesmo
        if not os.path.exists(output_abs) or os.path.getsize(output_abs) == 0:
            print(f"\n❌ ERRO: Arquivo vazio ou não criado.")
            sys.exit(1)

        print(f"\n✅ Conversão concluída com sucesso.")

    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        # Limpeza em caso de erro (arquivo parcial corrompido)
        if os.path.exists(output_abs):
            os.remove(output_abs)
        sys.exit(1)

    # Resumo final
    print("\n" + "="*60)
    print("ARQUIVO SALVO EM:")
    print(f"📂 {output_abs}")
    print("="*60)
