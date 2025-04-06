import pandas as pd
import sys

def csv_to_libsvm(input_file, output_file):
    # Ler o arquivo CSV
    df = pd.read_csv(input_file, delimiter=';')
    
    # Remover colunas vazias ou com todos valores NaN
    df = df.dropna(axis=1, how='all')
    
    # Mapeamento de rótulos: 0 → -1, 1 → +1
    label_map = {0: -1, 1: 1}
    
    with open(output_file, 'w') as f_out:
        for _, row in df.iterrows():
            # Obter e mapear o rótulo
            original_label = int(row.iloc[1])
            mapped_label = label_map[original_label]
            line = str(mapped_label)
            
            # Escrever as features no formato índice:valor
            for idx in range(2, len(row)):
                value = row.iloc[idx]
                # Verificar se o valor não é NaN e é diferente de zero
                if pd.notna(value) and float(value) != 0:
                    # O índice no libsvm começa em 1, então subtraímos 1 para compensar
                    line += f" {idx-1}:{value}"
            
            f_out.write(line + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python converter_libsvm.py <arquivo_entrada.csv> <arquivo_saida.libsvm>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    csv_to_libsvm(input_file, output_file)
    print(f"Arquivo convertido com sucesso para {output_file}")
