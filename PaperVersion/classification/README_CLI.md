# Guia de uso da linha de comando

Este guia descreve as opções de `svmParameters.py`, como reduzir a execução e como interpretar os arquivos produzidos. Os comandos abaixo partem da raiz de `SVM-new`, salvo indicação em contrário.

## 1. Preparar o ambiente

Instale as dependências listadas em `requirements.txt` na raiz do projeto e entre no diretório do script:

```bash
python -m pip install -r requirements.txt
cd classification
```

Confira as opções diretamente no script:

```bash
python svmParameters.py --help
```

O idioma do relatório HTML pode ser escolhido com `--language`. Se a opção for omitida, o relatório permanece em português (`pt-BR`):

```bash
# Português, comportamento padrão
python svmParameters.py --demo --quick-test

# Relatório em inglês
python svmParameters.py --demo --quick-test --language en
```

Também é possível escrever explicitamente `--language pt-BR`. O idioma não altera o treinamento, as métricas ou o conteúdo dos CSVs de auditoria; afeta os textos do relatório HTML e dos gráficos nele usados. Relatórios em inglês recebem o prefixo `svm_report_en_` no nome do arquivo.

## 2. Verificar a instalação

O dataset `heart_scale` acompanha o projeto. Para testar o fluxo completo com uma grade pequena:

```bash
python svmParameters.py --demo --quick-test
```

`--quick-test` substitui os parâmetros fornecidos por uma execução com 3 folds, kernel linear, `C=1` e `gamma=1` (gamma é ignorado para o kernel linear). Use esse modo apenas para verificar a execução e a geração do relatório; ele não é uma configuração de experimento final.

Para executar o dataset de demonstração sem essa redução:

```bash
python svmParameters.py --demo -kfold 10 -seed 42
```

`heart_scale` é um dataset demonstrativo de classificação binária, não uma base de malware.

## 3. Dataset próprio

O script recebe dados em formato LIBSVM, com uma linha por observação: primeiro o rótulo e depois pares `índice:valor` separados por espaço. Exemplo:

```text
0 1:0.25 3:1.8
1 1:0.91 2:0.4 3:1.2
```

Os rótulos binários aceitos pelo script são `0/1` e `-1/1`. Internamente, o rótulo negativo é mapeado para `0` e o positivo para `1`. Verifique essa convenção antes de interpretar recall, taxa de detecção e matriz de confusão.

Exemplo de execução:

```bash
python svmParameters.py -dataset caminho/para/dados.libsvm -kernels all -costs 1,10,100 -gammas 0.01,0.1,1 -kfold 10 -seed 42 -normalization maxabs
```

O conversor `converter_libsvm.py` pode ser executado a partir da raiz do projeto. Para rótulos textuais, declare as classes negativa e positiva explicitamente; consulte o README principal para o exemplo completo.

## 4. Escolher kernels e reduzir a execução

`-kernels` recebe `all` ou uma lista separada por vírgula. Os nomes reconhecidos são `linear`, `polinomial`, `rbf` e `sigmoid`.

```bash
# Executar somente Linear e RBF
python svmParameters.py -dataset dados.libsvm -kernels linear,rbf

# Executar somente o kernel polinomial
python svmParameters.py -dataset dados.libsvm -kernels polinomial

# Executar todos os kernels suportados
python svmParameters.py -dataset dados.libsvm -kernels all
```

Para reduzir o número de modelos treinados, restrinja também os valores de `C`, `gamma` ou o número de folds. Cada combinação aplicável é avaliada nos folds:

```bash
python svmParameters.py -dataset dados.libsvm -kernels linear,rbf -costs 1,10 -gammas 0.1,1 -kfold 5
```

Gamma não se aplica ao kernel `linear`. O script o ignora nesse kernel; para os demais, cada valor informado entra na grade. Se `-gammas` for omitido, o valor padrão é `1`.

## 5. Estimativa de tempo e kernels pulados

Antes do treinamento, o script calcula uma estimativa operacional para cada kernel. Se a estimativa de um kernel exceder `-max-time`, esse kernel é pulado; os outros kernels solicitados ainda podem ser executados. O limite é **por kernel**, não um limite rígido para a duração total do programa.

```bash
# Pular kernels cuja estimativa exceda 5 minutos
python svmParameters.py -dataset dados.libsvm -kernels all -max-time 300

# Solicitar somente Linear e RBF e permitir estimativas de até 1 hora por kernel
python svmParameters.py -dataset dados.libsvm -kernels linear,rbf -max-time 3600

# Executar todos os kernels solicitados mesmo acima do limite estimado
python svmParameters.py -dataset dados.libsvm -kernels all -max-time 300 --force-all-kernels
```

A estimativa operacional proposta por **WPSS e TMA**, coautoras desta reformulação, é proporcional a:

\[
T_{SVM} \approx c_k\,N^2\,(F\rho)\,|C|\,|\Gamma_k|\,K,
\]

em que \(N\) é o número de amostras, \(F\) o número de features, \(\rho\) a densidade de valores não nulos, \(|C|\) e \(|\Gamma_k|\) as quantidades de valores de C e gamma avaliados para o kernel \(k\), e \(K\) o número de folds. O termo \(F\rho\) aproxima o número médio de valores não nulos por amostra. Na implementação, \(c_k=2\times10^{-7}\), com multiplicadores heurísticos adicionais de 5 para polinomial e 1,2 para RBF.

Esta é uma regra de triagem para planejar a execução, não uma fórmula teórica que determina o tempo do solver LIBSVM. O custo real depende da estrutura dos dados, dos parâmetros e da convergência; cache e tolerância também podem afetá-lo. [Chang e Lin (2011)](https://doi.org/10.1145/1961189.1961199) descrevem a implementação e a otimização do LIBSVM, mas não propõem nem validam os fatores desta estimativa. A constante e os multiplicadores são hipóteses operacionais das autoras e devem ser calibrados comparando estimativas com tempos medidos no ambiente de uso. `--force-all-kernels` desativa a filtragem por estimativa, sem acelerar o treinamento nem impor prazo máximo. Se todos os kernels forem filtrados, não haverá treinamento nem relatório/auditoria daquela execução.

## 6. Opções e valores padrão

| Opção | Padrão | Uso |
|---|---:|---|
| `-dataset ARQUIVO` | — | Dataset no formato LIBSVM; necessário sem `--demo`. |
| `--demo` | desativado | Usa o `heart_scale` incluído no projeto. |
| `-kfold K` | `10` | Número de folds da validação cruzada estratificada; deve ser pelo menos 2. |
| `-seed N` | `1` | Semente do particionamento; deve ser não negativa. |
| `-kernels LISTA` | `all` | Lista de kernels separados por vírgula, ou `all`. Aliases: `-kernel` e `--kernel`. |
| `-costs LISTA` | `1,1000` | Valores positivos de `C`, separados por vírgula. |
| `-gammas LISTA` | `1` | Valores positivos de gamma para kernels não lineares. |
| `-threshold VALOR` | não aplicado | Limiar opcional de correlação absoluta com o rótulo para seleção de features; faixa de 0 a 1. |
| `-leakage-threshold VALOR` | `0.999` | Remove no fold features com correlação absoluta igual ou superior ao limiar; faixa `(0, 1]`. |
| `-normalization MÉTODO` | `none` | Normalização fold-safe: `none`, `maxabs` ou `minmax`. |
| `-cache MB` | `500` | Cache do LIBSVM, em MB; deve ser positivo. |
| `-epsilon VALOR` | `0.05` | Tolerância de parada do LIBSVM; deve ser positiva. |
| `-max-time SEGUNDOS` | `3600` | Limite usado para filtrar kernels pela estimativa operacional. |
| `--force-all-kernels` | desativado | Executa os kernels solicitados mesmo acima do limite estimado. |
| `--quick-test` | desativado | Substitui a grade por 3 folds, Linear, `C=1`, `gamma=1`; teste funcional apenas. |
| `--full-report` | desativado | Inclui no relatório métricas completas, como F1, recall, FPR, precisão, F1-macro, balanced accuracy e MCC. |
| `--language IDIOMA` | `pt-br` | Idioma do relatório HTML: `pt-br` ou `en`; não altera o treinamento nem a auditoria CSV. |
| `-h`, `--help` | — | Exibe a ajuda. |

## 7. Pré-processamento e auditoria de features

A seleção usa a correlação absoluta entre cada feature e o rótulo. Por ser supervisionada, ela é ajustada apenas com os dados de treino de cada fold; o conjunto de teste daquele fold não participa da seleção. A normalização opcional segue a mesma regra: é ajustada no treino do fold e aplicada ao teste correspondente.

`-threshold` é opcional. Quando omitido, o script não aplica um corte de correlação mínima, mas ainda faz a auditoria por fold e remove features suspeitas de vazamento segundo `-leakage-threshold`. Features constantes também são tratadas dentro do fold. A seleção pode variar de um fold para outro.

Cada execução bem-sucedida produz dois CSVs de auditoria:

- `svm_feature_audit_<data-hora>.csv`: registros de todas as features avaliadas, com kernel, `C`, gamma, fold, identificador original, correlação e status;
- `svm_feature_audit_<data-hora>_selected.csv`: somente registros cujo status é `selected`.

O CSV `_selected` contém apenas as linhas de features selecionadas e, portanto, não substitui o CSV completo quando é necessário examinar features removidas ou suspeitas.

## 8. Relatório e arquivos de saída

As saídas são gravadas em `svm-results/` dentro do diretório de onde o comando foi executado. Seguindo os comandos deste guia a partir de `classification/`, os arquivos aparecerão em `classification/svm-results/`.

O diretório pode conter:

- `svm_report_<data-hora>.html`: relatório navegável com resultados e diagnósticos;
- `svm_feature_audit_<data-hora>.csv` e `svm_feature_audit_<data-hora>_selected.csv`: auditorias de features;
- `svm_report_images_<data-hora>/` (português) ou `svm_report_images_en_<data-hora>/` (inglês): imagens próprias daquela execução, incluindo matrizes de confusão e gráficos.

Abra o HTML em um navegador. Os resultados de execução são artefatos locais e estão excluídos do Git pelo `.gitignore`.

O relatório em português ou inglês usa abas para visão geral, comparativos, resumo por kernel, diagnósticos e auditoria. A aba de comparativos só aparece quando há boxplots. Nos resumos e diagnósticos, selecione o kernel para ver uma configuração por vez. O seletor não altera os dados; a impressão/PDF exibe todos os kernels e seções, e o conteúdo permanece em sequência se o JavaScript estiver desativado.

## 9. Reprodutibilidade e interpretação dos resultados

Registre junto aos resultados o dataset e sua proveniência, a versão do código e das dependências, a semente, o número de folds, os kernels e grades avaliados e o pré-processamento. A mesma semente ajuda a repetir os folds, mas não garante resultados idênticos entre ambientes e versões diferentes de bibliotecas ou solver.

O melhor cenário é escolhido entre as configurações avaliadas. Como essa escolha usa os próprios resultados da validação cruzada, apresentar o melhor resultado da mesma busca como estimativa final independente pode produzir otimismo. Para estimar generalização após selecionar features e hiperparâmetros, prefira validação cruzada aninhada ou um conjunto de teste externo que não participe das escolhas.

Não use `--quick-test` para conclusões científicas. A estimativa operacional de tempo tampouco deve ser apresentada como medição de custo computacional; para isso, registre tempos observados de forma controlada, com protocolo e ambiente descritos.
