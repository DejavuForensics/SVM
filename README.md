# SVM

## SVM: Classification/Regression

The SVM is a statistical learning machine. It is not based on the human brain. Its explicit goal is statistical learning theory. 
Classical neural networks aim to find a hyperplane. The hyperplane separates the classes for the target application.
There can be several hyperplanes separating the data correctly. SVM is a classifier that finds a better hyperplane than others.

### Follow the instructions:
In the terminal, install the virtual environment.
```
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

A good generalisation in machine learning may depend on a good kernel choice. The best kernel may be subordinate to the problem to be solved. Investigating different kernels has a side effect. It is a costly process. It involves cross-validation and different random starts. Investigating different kernels may be necessary. Otherwise, a neural network with a mismatched kernel may generate bad results.

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

# PT-BR:
## SVM: Classificação/Predição

O SVM é uma máquina de aprendizado estatístico que não se inspira necessariamente no funcionamento do cérebro humano. Seu objetivo explícito é a teoria do aprendizado estatístico. 
As redes neurais clássicas visam encontrar um hiperplano de modo a separar as classes pertencentes à aplicação alvo.
Podem existir vários hiperplanos separando os dados corretamente. Ao contrário de redes clássicas, a SVM é um classificador que visa encontrar um hiperplano melhor do que os demais.

### Siga as instruções:
No terminal, instale o _libsvm_.
```
pip install libsvm
```

Parâmetros de uso do SVM:

-	-dataset: especifica o caminho da base de dados. Por padrão, as bases de dados _heart_scale_ e _bodyfat_scale_ são empregadas na classificação e regressão, respectivamente.

```
python svm.py
```
Se surgirem problemas, é provável que sejam causados por incompatibilidades entre as versões das bibliotecas _NumPy_, _SciPy_ e _libsvm_. Nesse caso, recomenda-se realizar a desatualização (_downgrade_) do _SciPy_ para uma versão anterior.
```
pip install "scipy<1.12"
```

## SVM - K-fold
A validação cruzada é uma técnica estatística usada para avaliar o desempenho de um modelo de aprendizado de máquina. Ela divide o conjunto de dados em várias partes, ou "dobras", para que o modelo possa ser treinado e testado múltiplas vezes em diferentes subconjuntos dos dados. O objetivo é garantir que o modelo generalize bem para dados novos e não vistos,
O método k-fold é uma forma específica de validação cruzada onde o conjunto de dados é dividido aleatoriamente em k subconjuntos (ou folds) aproximadamente iguais. O k-fold envolve os seguintes passos:

- **Divisão dos dados**: o conjunto de dados é dividido em k partes aproximadamente iguais.

- **Treinamento e teste**: para cada uma das k dobras, o modelo é treinado utilizando k-1 dobras e testado na dobra restante.

- **Repetição do processo**: esse processo é repetido k vezes, cada vez com uma dobra diferente atuando como conjunto de teste.

- **Média e desvio padrão dos resultados**: as métricas de desempenho são calculadas para cada uma das k execuções e, em seguida, a média dessas métricas é computada para obter uma estimativa final do desempenho do modelo.

### Siga as instruções:
No terminal, instale o _scikit-learn_.
```
pip install scikit-learn
```

Parâmetros de uso do SVM dotado de validação cruzada _k-fold_:

-	-dataset: especifica o caminho da base de dados. Por padrão, as bases de dados _heart_scale_ e _bodyfat_scale_ são empregadas na classificação e regressão, respectivamente.

```
python svmKfold.py
```
## Parâmetros do Classificador SVM

Um dos grandes desafios, em máquinas de aprendizado estatístico, diz respeito a encontrar um _kernel_ de modo que otimize a fronteira de decisão entre as classes de uma dada aplicação. Em SVM, um _kernel_ Linear, por exemplo, é capaz de resolver um problema linearmente separável, como o visto na Fig. 3 (a). Seguindo o mesmo raciocínio, _kernels_ Sigmóide, RBF  e Senoide são capazes de resolver problemas separáveis por função Sigmoidal, Radial e Senoidal, vistos na  Fig. 3 (b), na  Fig. 3 (c) e na Fig. 3 (d), respectivamente. 

Então uma boa capacidade de generalização da _machine learning_ pode depender de uma escolha ajustada do _kernel_. O melhor _kernel_ pode estar subordinado ao problema a ser resolvido. Como efeito colateral, a investigação de diferentes _kernels_ é geralmente um processo custoso envolvendo validação cruzada combinada com diferentes condições iniciais aleatórias. A investigação de distintos _kernels_, no entanto, pode ser necessária, caso contrário a rede neural composta, por um _kernel_ desajustado, por gerar resultados não satisfatórios.
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
