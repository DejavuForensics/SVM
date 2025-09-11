# SVM

## SVM: Classification/Regression

The SVM is a statistical learning machine. It is not based on the human brain. Its explicit goal is statistical learning theory. 
Classical machines learning aim to find a hyperplane. The hyperplane separates the classes for the target application.
There can be several hyperplanes separating the data correctly. SVM is a classifier that finds a better hyperplane than others.

### Follow the instructions:
In the terminal, install the virtual environment.
```
sudo su
python -m venv venv
source venv/bin/activate
```

In the terminal, install requirements.
```
pip install -r requirements.txt
```

Parameters for using the SVM to recognise patterns or prediction:

-	-dataset: specifies the database path. By default, the _heart_scale_ and _bodyfat_scale_ databases are used for classification and regression, respectively.

```
python svm.py
```


## SVM: K-fold
Cross-validation is a statistical technique. Researchers use it to assess the performance of a machine learning model. It divides the data set into parts, or ‘folds’. You can train and test the model many times on different data subsets. The aim is to ensure that the model generalises well to new and unseen data. The k-fold method is a type of cross-validation. In it, the data set is randomly split into k equal subsets (or folds). The k-fold involves the following steps:

- **Data division**: the data set is divided into k approximately equal parts.

- **Training and testing**: for each of the k folds, the model is trained using k-1 folds and tested on the remaining fold.

- **Repeating the process**: this process is repeated k times, each time with a different fold acting as the test set.

- **Average and standard deviation of results**: performance metrics are calculated for each of the k runs. For each run, we calculate the average and standard deviation. We then average these metrics. This gives the model's final performance.

### Follow the instructions:

Parameters for using the SVM with _k-fold_ cross-validation:

-	-dataset: specifies the database path. By default, the _heart_scale_ and _bodyfat_scale_ databases are used for classification and regression, respectively.

```
python svmKfold.py
```

## SVM Parameters

In statistical machine learning, a major challenge is finding a kernel. The kernel optimizes the decision boundary between classes in an application. In SVM, a Linear kernel, for example, is capable of solving a linearly separable problem such as that seen in Fig. 1 (a). Following the same logic. Sigmoid, RBF, and Sine kernels can solve separable problems. They are separable by Sigmoid, Radial, and Sine functions. This is seen in Fig. 1 (b), Fig. 1 (c) and Fig. 1 (d), respectively. 

A good generalisation in machine learning may depend on a good kernel choice. The best kernel may be subordinate to the problem to be solved. Investigating different kernels has a side effect. It is a costly process. It involves cross-validation and different random starts. Investigating different kernels may be necessary. Otherwise, a machine learning with a mismatched kernel may generate bad results.

As a counter-example, look at the use of the Linear kernel. It was applied to the Sigmoid and Sine distributions shown in Fig. 2 (a) and Fig. 2 (b), respectively. The classification accuracies shown in Fig. 2 (a) and Fig. 2 (b) are 78.71% and 73.00% respectively. You can see this visually. The Linear kernel does not map the boundaries of the Sigmoid and Sinusoid distributions well.

The kernels generalize well. But, this depends on choosing good parameters (C, gamma). The cost parameter C balances margin width and reduces classification error. It balances these factors relative to the training set. The kernel gamma parameter controls the decision limit depending on the classes. There is no universal method for choosing the parameters (C, gamma). C and gamma increase fast. They follow the function 10 to the power of n. Here, n ranges from -3 to 3. The hypothesis is to check if these parameters differ from the standards. The standards are (C, gamma) = ( 10<sup>0</sup>, 10<sup>0</sup>). The parameters can generate better accuracies.

<figure>
  <img src="https://github.com/DejavuForensics/SVM/blob/main/EN-US/SVM_1.png" alt="Successful performances of the _kernels_ compatible with the datasets.">
  <figcaption>Figure 1: Successful performances of the _kernels_ compatible with the datasets.</figcaption>
</figure>

<figure>
  <img src="https://github.com/DejavuForensics/SVM/blob/main/EN-US/SVM_2.png" alt=" Unsuccessful performances of _kernel_ Linear on non-linearly separable datasets." >
  <figcaption>Figure 2: Unsuccessful performances of _kernel_ Linear on non-linearly separable datasets.</figcaption>
</figure>

### Follow the instructions:

Parameters for using the SVM with optimised parameter study:

-	-dataset: specifies the database path. By default, the _heart_scale_ and _bodyfat_scale_ databases are used for classification and regression respectively.

```
python svmParameters.py
```

# Usage in distinct antiviruses

## Antivirus for malicious Google Chrome extensions

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_GoogleChromeExtension_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_GoogleChromeExtension_mELM_format.csv)    Antivirus_Dataset_GoogleChromeExtension_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_GoogleChromeExtension_SVM_format.libsvm

```
Gabriela Leite Pereira, Leonardo Silvino Brito, Sidney Marlon Lopes de Lima,
Antivirus applied to Google Chrome's extension malware,
Computers & Security, 156, 104465 (2025).
https://doi.org/10.1016/j.cose.2025.104465
```


## Antivirus for IoT _malware_ from ARM architectures

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_IoT_ARM_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_IoT_ARM_mELM_format.csv)    Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

```
Tavares-Silva, S.H.M., Lopes-Lima, S.M., Paranhos-Pinheiro, R. et al.
Antivirus solution to IoT malware detection with authorial next-generation sandbox.
The Journal of Supercomputing 81, 151 (2025).
https://doi.org/10.1007/s11227-024-06506-x
```

## Antivirus for IoT _malware_ from SPARC architectures

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_IoT_SPARC_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_IoT_SPARC_mELM_format.csv)    Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

```
Pinheiro Henriques de Araújo, I., Mariano Santiago de Abreu, L., Henrique Mendes Tavares Silva, S. et al.
Antimalware applied to IoT malware detection based on softcore processor endowed with authorial sandbox.
Journal of Computer Virology and Hacking Techniques 20, 729–749 (2024).
https://doi.org/10.1007/s11416-024-00526-0
```

## Antivirus for Citadel _malware_

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_PE32_Citadel_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_PE32_Citadel_mELM_format.csv)    Antivirus_Dataset_PE32_Citadel_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_PE32_Citadel_SVM_format.libsvm


```
Carlos Henrique Macedo dos Santos, Sidney Marlon Lopes de Lima,
XAI-driven antivirus in pattern identification of citadel malware,
Journal of Computational Science, 82 (2024): 102389.
https://doi.org/10.1016/j.jocs.2024.102389.
```

## Antivirus for Java malicious apps
The database is compressed due to the individual file size delimited by github. Download the compressed file (.zip) to your computer and decompress it before running the SVM machine learning.

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_Jar_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_Jar_mELM_format.csv)    Antivirus_Dataset_Jar_SVM_format.libsvm


python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_Jar_SVM_format.libsvm

```
Pinheiro, R.P., Lima, S.M.L., Souza, D.M. et al. 
Antivirus applied to JAR malware detection based on runtime behaviors. 
Scientific Reports - Nature Research 12, 1945 (2022). 
https://doi.org/10.1038/s41598-022-05921-5
```

## Antivirus for PHP malicious apps

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_PHP_batch_1_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_PHP_batch_1_mELM_format.csv)    Antivirus_Dataset_PHP_batch_1_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset [Antivirus_Dataset_PHP_batch_1_SVM_format.libsvm]

```
Lima, S.M.L., Silva, S.H.M.T., Pinheiro, R.P. et al.
Next-generation antivirus endowed with web-server Sandbox applied to audit fileless attack.
Soft Computing 27, 1471–1491 (2023).
https://doi.org/10.1007/s00500-022-07447-4
```

## Antivirus for JavaScript malicious apps
The database is compressed due to the individual file size delimited by github. Download the compressed file (.zip) to your computer and decompress it before running the SVM machine learning.

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_JavaScript_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_JavaScript_mELM_format.csv)    Antivirus_Dataset_JavaScript_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/EN-US/classification/svmParameters.py) -dataset Antivirus_Dataset_JavaScript_SVM_format.libsvm

```
de Lima, S.M.L., Souza, D.M., Pinheiro, R.P. et al. 
Next-generation antivirus for JavaScript malware detection based on dynamic features. 
Knowledge and Information Systems 66, 1337–1370 (2024).
https://doi.org/10.1007/s10115-023-01978-4
```

# PT-BR:
## SVM: Classificação/Predição

O SVM é uma máquina de aprendizado estatístico que não se inspira necessariamente no funcionamento do cérebro humano. Seu objetivo explícito é a teoria do aprendizado estatístico. 
As redes neurais clássicas visam encontrar um hiperplano de modo a separar as classes pertencentes à aplicação alvo.
Podem existir vários hiperplanos separando os dados corretamente. Ao contrário de redes clássicas, a SVM é um classificador que visa encontrar um hiperplano melhor do que os demais.

### Siga as instruções:
No terminal, instale o ambiente virtual
```
sudo su
python -m venv venv
source venv/bin/activate
```

No terminal, instale as dependências.
```
pip install -r requirements.txt
```

Parâmetros de uso do SVM:

-	-dataset: especifica o caminho da base de dados. Por padrão, as bases de dados _heart_scale_ e _bodyfat_scale_ são empregadas na classificação e regressão, respectivamente.

```
python svm.py
```

## SVM - K-fold
A validação cruzada é uma técnica estatística usada para avaliar o desempenho de um modelo de aprendizado de máquina. Ela divide o conjunto de dados em várias partes, ou "dobras", para que o modelo possa ser treinado e testado múltiplas vezes em diferentes subconjuntos dos dados. O objetivo é garantir que o modelo generalize bem para dados novos e não vistos,
O método k-fold é uma forma específica de validação cruzada onde o conjunto de dados é dividido aleatoriamente em k subconjuntos (ou folds) aproximadamente iguais. O k-fold envolve os seguintes passos:

- **Divisão dos dados**: o conjunto de dados é dividido em k partes aproximadamente iguais.

- **Treinamento e teste**: para cada uma das k dobras, o modelo é treinado utilizando k-1 dobras e testado na dobra restante.

- **Repetição do processo**: esse processo é repetido k vezes, cada vez com uma dobra diferente atuando como conjunto de teste.

- **Média e desvio padrão dos resultados**: as métricas de desempenho são calculadas para cada uma das k execuções e, em seguida, a média dessas métricas é computada para obter uma estimativa final do desempenho do modelo.

### Siga as instruções:

Parâmetros de uso do SVM dotado de validação cruzada _k-fold_:

-	-dataset: especifica o caminho da base de dados. Por padrão, as bases de dados _heart_scale_ e _bodyfat_scale_ são empregadas na classificação e regressão, respectivamente.

```
python svmKfold.py
```
## Parâmetros do Classificador SVM

Um dos grandes desafios, em máquinas de aprendizado estatístico, diz respeito a encontrar um _kernel_ de modo que otimize a fronteira de decisão entre as classes de uma dada aplicação. Em SVM, um _kernel_ Linear, por exemplo, é capaz de resolver um problema linearmente separável, como o visto na Fig. 3 (a). Seguindo o mesmo raciocínio, _kernels_ Sigmóide, RBF  e Senoide são capazes de resolver problemas separáveis por função Sigmoidal, Radial e Senoidal, vistos na  Fig. 3 (b), na  Fig. 3 (c) e na Fig. 3 (d), respectivamente. 

Então uma boa capacidade de generalização da _machine learning_ pode depender de uma escolha ajustada do _kernel_. O melhor _kernel_ pode estar subordinado ao problema a ser resolvido. Como efeito colateral, a investigação de diferentes _kernels_ é geralmente um processo custoso envolvendo validação cruzada combinada com diferentes condições iniciais aleatórias. A investigação de distintos _kernels_, no entanto, pode ser necessária, caso contrário a máquina de aprendizado estatístico composta, por um _kernel_ desajustado, por gerar resultados não satisfatórios.
Como contra-exemplo, observe o emprego do _kernel_ Linear aplicado a distribuições Sigmóide e Senoide apresentados na Fig. 4 (a) e na Fig. 4 (b), respectivamente. As precisões das classificações expostas na Fig. 4 (a) e na Fig. Fig. 4 (b) são de 78,71% e 73,00%, respectivamente. Visualmente, é possível observar que o _kernel_ Linear não mapeia as fronteiras de decisões das distribuições Sigmóide e Senoide de forma adequada.

Uma boa capacidade de generalização desses _kernels_ também depende de uma escolha ajustada de parâmetros (C, gamma). O parâmetro de custo C se refere a um ponto de equilíbrio razoável entre a largura da margem do hiperplano e a minimização do erro de classificação em relação ao conjunto de treinamento. O parâmetro do _kernel_ gamma controla o limite de decisão em função das classes. Não existe um método universal no sentido de escolher os parâmetros (C, gamma). No presente trabalho, os parâmetros C e gamma variam exponencialmente em sequências crescentes, matematicamente de acordo com a função 10<sup>n</sup>, onde n={-3, -2, -1, 0, 1, 2, 3 }. A hipótese é verificar se esses parâmetros distintos dos padrões; (C, gamma) = ( 10<sup>0</sup>, 10<sup>0</sup>), são capazes de gerar melhores acurácias.  

<figure>
  <img src="https://github.com/DejavuForensics/SVM/blob/main/PT-BR/SVM_1.png" alt="Atuações bem-sucedidas dos _kernels_ compatíveis com os conjuntos de dados.">
  <figcaption>Figura 3: Atuações bem-sucedidas dos _kernels_ compatíveis com os conjuntos de dados.</figcaption>
</figure>

<figure>
  <img src="https://github.com/DejavuForensics/SVM/blob/main/PT-BR/SVM_2.png" alt="Atuações malsucedidas do _kernel_ Linear em conjuntos de dados não-linearmente separáveis.">
  <figcaption>Figura 4: Atuações malsucedidas do _kernel_ Linear em conjuntos de dados não-linearmente separáveis.</figcaption>
</figure>

### Siga as instruções:

Parâmetros de uso do SVM com estudo de parâmetros otimizados:

-	-dataset: especifica o caminho da base de dados. Por padrão, as bases de dados _heart_scale_ e _bodyfat_scale_ são empregadas na classificação e regressão, respectivamente.

```
python svmParameters.py
```


# Utilização em antivírus distintos

## Antivírus para extensões maliciosas do Google Chrome

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_GoogleChromeExtension_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_GoogleChromeExtension_mELM_format.csv)    Antivirus_Dataset_GoogleChromeExtension_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_GoogleChromeExtension_SVM_format.libsvm

```
Gabriela Leite Pereira, Leonardo Silvino Brito, Sidney Marlon Lopes de Lima,
Antivirus applied to Google Chrome's extension malware,
Computers & Security, 156, 104465 (2025).
https://doi.org/10.1016/j.cose.2025.104465
```


## Antivírus para _malware_ de IoT (Internet of Things - Internet das Coisas) em arquiteturas ARM

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_IoT_ARM_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_IoT_ARM_mELM_format.csv)    Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

```
Tavares-Silva, S.H.M., Lopes-Lima, S.M., Paranhos-Pinheiro, R. et al.
Antivirus solution to IoT malware detection with authorial next-generation sandbox.
The Journal of Supercomputing 81, 151 (2025).
https://doi.org/10.1007/s11227-024-06506-x
```

## Antivírus para _malware_ de IoT (Internet of Things - Internet das Coisas) em arquiteturas SPARC

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_IoT_SPARC_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_IoT_SPARC_mELM_format.csv)    Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_IoT_SPARC_SVM_format.libsvm

```
Pinheiro Henriques de Araújo, I., Mariano Santiago de Abreu, L., Henrique Mendes Tavares Silva, S. et al.
Antimalware applied to IoT malware detection based on softcore processor endowed with authorial sandbox.
Journal of Computer Virology and Hacking Techniques 20, 729–749 (2024).
https://doi.org/10.1007/s11416-024-00526-0
```

## Antivírus para _malware_ Citadel

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_PE32_Citadel_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_PE32_Citadel_mELM_format.csv)    Antivirus_Dataset_PE32_Citadel_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_PE32_Citadel_SVM_format.libsvm


```
Carlos Henrique Macedo dos Santos, Sidney Marlon Lopes de Lima,
XAI-driven antivirus in pattern identification of citadel malware,
Journal of Computational Science, 82 (2024): 102389.
https://doi.org/10.1016/j.jocs.2024.102389.
```

## Antivírus para aplicativos maliciosos Java
O banco de dados está compactado devido ao tamanho individual dos arquivos delimitado pelo github. Baixe o arquivo compactado (.zip) para o seu computador e descompacte-o antes de executar as máquinas de aprendizado SVM.

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_Jar_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_Jar_mELM_format.csv)    Antivirus_Dataset_Jar_SVM_format.libsvm


python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_Jar_SVM_format.libsvm

```
Pinheiro, R.P., Lima, S.M.L., Souza, D.M. et al. 
Antivirus applied to JAR malware detection based on runtime behaviors. 
Scientific Reports - Nature Research 12, 1945 (2022). 
https://doi.org/10.1038/s41598-022-05921-5
```

## Antivírus para aplicativos maliciosos em PHP

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_PHP_batch_1_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_PHP_batch_1_mELM_format.csv)    Antivirus_Dataset_PHP_batch_1_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset [Antivirus_Dataset_PHP_batch_1_SVM_format.libsvm]

```
Lima, S.M.L., Silva, S.H.M.T., Pinheiro, R.P. et al.
Next-generation antivirus endowed with web-server Sandbox applied to audit fileless attack.
Soft Computing 27, 1471–1491 (2023).
https://doi.org/10.1007/s00500-022-07447-4
```

## Antivírus para aplicativos maliciosos em JavaScript
O repositório está compactado devido ao tamanho individual dos arquivos delimitado pelo GitHub. Baixe o arquivo compactado (.zip) para o seu computador e descompacte-o antes de executar as máquinas de aprendizado SVM.

python [converter_libsvm.py](https://github.com/DejavuForensics/SVM/blob/main/converter_libsvm.py)     [Antivirus_Dataset_JavaScript_mELM_format.csv](https://github.com/DejavuForensics/SVM/blob/main/Antiviruses/Antivirus_Dataset_JavaScript_mELM_format.csv)    Antivirus_Dataset_JavaScript_SVM_format.libsvm

python [EN-US/classification/svmParameters.py](https://github.com/DejavuForensics/SVM/blob/main/PT-BR/classificacao/svmParameters.py) -dataset Antivirus_Dataset_JavaScript_SVM_format.libsvm

```
de Lima, S.M.L., Souza, D.M., Pinheiro, R.P. et al. 
Next-generation antivirus for JavaScript malware detection based on dynamic features. 
Knowledge and Information Systems 66, 1337–1370 (2024).
https://doi.org/10.1007/s10115-023-01978-4
```

