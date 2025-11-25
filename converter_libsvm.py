import pandas as pd
import sys
import os
import shutil

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
    
    # Definições de Caminhos
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Pasta onde este script está
    input_arg = sys.argv[1]
    output_filename = os.path.basename(sys.argv[2]) # Pega só o nome do arquivo, ignora pastas
    
    # Força o caminho de entrada ser absoluto
    input_abs = os.path.abspath(input_arg)
    
    # Força a criação do arquivo INICIALMENTE na mesma pasta do script
    output_abs = os.path.join(script_dir, output_filename)
    
    print("="*60)
    print("CONVERSOR CSV -> LIBSVM")
    print("="*60)

    try:
        csv_to_libsvm(input_abs, output_abs)
        
        # Verifica se criou mesmo
        if not os.path.exists(output_abs) or os.path.getsize(output_abs) == 0:
            print(f"\n❌ ERRO: Arquivo vazio ou não criado.")
            sys.exit(1)
        print(f"\n✅ Conversão inicial concluída.")
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        if os.path.exists(output_abs):
            os.remove(output_abs)
        sys.exit(1)

    # --- MENU DE DESTINO ---
    print("\n" + "-"*60)
    print("PARA ONDE MOVER O ARQUIVO?")
    print("1 - EN-US/classification/  (Inglês)")
    print("2 - PT-BR/classificacao/   (Português)")
    print("0 - Manter aqui (Não mover)")
    print("-"*60)

    raw_choice = input("Digite a opção (0, 1 ou 2): ").strip()
    choice = raw_choice.replace("\\", "").replace("/", "") # Limpeza de input

    # Por padrão, o local final é onde ele já foi criado (na pasta do script)
    final_location = output_abs 

    if choice in ['1', '2']:
        # Define pastas de destino
        if choice == '1':
            target_folder = os.path.join(script_dir, "EN-US", "classification")
        else:
            target_folder = os.path.join(script_dir, "PT-BR", "classificacao")

        # Garante que a pasta existe
        if not os.path.exists(target_folder):
            print(f"   [INFO] Criando pasta: {target_folder}")
            os.makedirs(target_folder)

        destination_path = os.path.join(target_folder, output_filename)

        try:
            if os.path.exists(destination_path):
                os.remove(destination_path) # Remove anterior se existir para não dar erro
            
            shutil.move(output_abs, destination_path)
            final_location = destination_path
            print(f"\n🚀 SUCESSO! Arquivo movido para a pasta selecionada.")
        except Exception as e:
            print(f"\n⚠️ Erro ao mover: {e}")
            print(f"   O arquivo foi mantido no local original por segurança.")
            final_location = output_abs
    
    else:
        # AQUI ESTÁ A LÓGICA QUE VOCÊ PEDIU
        print(f"\n[AVISO] Opção '{raw_choice}' não é um destino de movimentação (ou é inválida).")
        print("        O arquivo foi mantido no mesmo diretório deste script (converter_libsvm.py).")
        # final_location já é output_abs, então não precisamos mexer

    # Resumo final obrigatório
    print("\n" + "="*60)
    print("LOCALIZAÇÃO FINAL DO ARQUIVO:")
    print(f"📂 {final_location}")
    print("="*60)