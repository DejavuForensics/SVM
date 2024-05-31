# SVM

## SVM Classifier: Classification/Regression

The SVM is a statistical learning machine. It is not based on the human brain. Its explicit goal is statistical learning theory. 
Classical neural networks aim to find a hyperplane. The hyperplane separates the classes for the target application.
There can be several hyperplanes separating the data correctly. SVM is a classifier that finds a better hyperplane than others.

### Follow the instructions
In the terminal, install _libsvm_.
```
pip install libsvm
```

```
python svm.py
```
## SVM classifier: K-fold
Cross-validation is a statistical technique. Researchers use it to assess the performance of a machine learning model. It divides the data set into parts, or ‘folds’. You can train and test the model many times on different data subsets. The aim is to ensure that the model generalises well to new and unseen data. The k-fold method is a type of cross-validation. In it, the data set is randomly split into k equal subsets (or folds). The k-fold involves the following steps:

- **Data division**: the data set is divided into k approximately equal parts.

- **Training and testing**: for each of the k folds, the model is trained using k-1 folds and tested on the remaining fold.

- **Repeating the process**: this process is repeated k times, each time with a different fold acting as the test set.

- **Average and standard deviation of results**: performance metrics are calculated for each of the k runs. For each run, we calculate the average and standard deviation. We then average these metrics. This gives the model's final performance.

In the terminal, install _libsvm_.
```
pip install libsvm
```

```
python svm.py
```

## SVM Classifier Parameters

One big challenge in machine learning is finding a kernel. It optimises the decision boundary between the classes of a given application. In ELM neural networks, a Linear \textit{kernel} can solve a linearly separable problem. This is the kind seen in \ref{fig:ELM1} (a). Sigmoid, RBF, and Sinusoid \textit{kernels} follow the same reasoning. They can solve problems separable by Sigmoid, Radial, and Sinusoidal functions. This is as seen in Fig. \ref{fig:ELM1} (b), Fig. \ref{fig:ELM1} (c) and Fig. \ref{fig:ELM1} (d), respectively. 

So, the network may generalize well. This depends on choosing an appropriate \textit{kernel}. The best \textit{kernel} may be dependent on the problem to be solved. As a side effect, looking at different \textit{kernels} is costly. It requires cross-validation and different initial conditions. You may need to investigate different \textit{kernels}. A mismatched \textit{kernel} may cause bad results in the neural network.

As a counter-example, look at the Linear \textit{kernel}. It was applied to the Sigmoid and Sine distributions shown in Fig. \ref{fig:ELM2} (a) and Fig. \ref{fig:ELM2} (b), respectively.

# PT-BR:
## Classificador SVM: Reconhecimento de Padrão

O SVM é uma máquina de aprendizado estatístico que não se inspira necessariamente no funcionamento do cérebro humano. Seu objetivo explícito é a teoria do aprendizado estatístico. 
As redes neurais clássicas visam encontrar um hiperplano de modo a separar as classes pertencentes à aplicação alvo.
Podem existir vários hiperplanos separando os dados corretamente. Ao contrário de redes clássicas, a SVM é um classificador que visa encontrar um hiperplano melhor do que os demais.

## Classificador SVM - K-fold
A validação cruzada é uma técnica estatística usada para avaliar o desempenho de um modelo de aprendizado de máquina. Ela divide o conjunto de dados em várias partes, ou "dobras", para que o modelo possa ser treinado e testado múltiplas vezes em diferentes subconjuntos dos dados. O objetivo é garantir que o modelo generalize bem para dados novos e não vistos,
O método k-fold é uma forma específica de validação cruzada onde o conjunto de dados é dividido aleatoriamente em k subconjuntos (ou folds) aproximadamente iguais. O k-fold envolve os seguintes passos:

- **Divisão dos dados**: o conjunto de dados é dividido em k partes aproximadamente iguais.

- **Treinamento e teste**: para cada uma das k dobras, o modelo é treinado utilizando k-1 dobras e testado na dobra restante.

- **Repetição do processo**: esse processo é repetido k vezes, cada vez com uma dobra diferente atuando como conjunto de teste.

- **Média e desvio padrão dos resultados**: as métricas de desempenho são calculadas para cada uma das k execuções e, em seguida, a média dessas métricas é computada para obter uma estimativa final do desempenho do modelo.


\section{Parâmetros do Classificador SVM}
\index{Parâmetros do Classificador SVM}
\label{sec:6.3}
\hspace{\parindent}

Um dos grandes desafios, em máquinas de aprendizado estatístico, diz respeito a encontrar um \textit{kernel} de modo que otimize a fronteira de decisão entre as classes de uma dada aplicação. Em redes neurais ELM, um \textit{kernel} Linear, por exemplo, é capaz de resolver um problema linearmente separável, como o visto na \ref{fig:ELM1} (a). Seguindo o mesmo raciocínio, \textit{kernels} Sigmóide, RBF  e Senoide são capazes de resolver problemas separáveis por função Sigmoidal, Radial e Senoidal, vistos na Fig. \ref{fig:ELM1} (b), na Fig. \ref{fig:ELM1} (c) e na  Fig. \ref{fig:ELM1} (d), respectivamente. 

Então, uma boa capacidade de generalização da rede neural pode depender de uma escolha ajustada do \textit{kernel}. O melhor \textit{kernel} pode estar subordinado ao problema a ser resolvido. Como efeito colateral, a investigação de diferentes \textit{kernels} é geralmente um processo custoso envolvendo validação cruzada combinada com diferentes condições iniciais aleatórias. A investigação de distintos \textit{kernels}, no entanto, pode ser necessária, caso contrário a rede neural composta, por um \textit{kernel} desajustado, por gerar resultados não satisfatórios.
Como contra-exemplo, observe o emprego do \textit{kernel} Linear aplicado a distribuições Sigmóide e Senoide apresentados na Fig. \ref{fig:ELM2} (a) e na Fig. \ref{fig:ELM2} (b), respectivamente. As precisões das classificações expostas na Fig. \ref{fig:ELM2} (a) e na Fig. \ref{fig:ELM2} (b) são de 78,71\% e 73,00\%, respectivamente. Visualmente, é possível observar que o \textit{kernel} Linear não mapeia as fronteiras de decisões das distribuições Sigmóide e Senoide de forma adequada.

Uma boa capacidade de generalização desses \textit{kernels} também depende de uma escolha ajustada de parâmetros $(C, \gamma)$. O parâmetro de custo $C$ se refere a um ponto de equilíbrio razoável entre a largura da margem do hiperplano e a minimização do erro de classificação em relação ao conjunto de treinamento. O parâmetro do \textit{kernel} $\gamma$ controla o limite de decisão em função das classes \cite{Pinheiro}. Não existe um método universal no sentido de escolher os parâmetros $(C, \gamma)$. No presente trabalho, os parâmetros $C$ e $\gamma$ variam exponencialmente em sequências crescentes, matematicamente de acordo com a função $10^n$, onde $n=\left \{-3, -2, -1, 0, 1, 2, 3 \right \}$. A hipótese é verificar se esses parâmetros distintos dos padrões; $(C, \gamma) = (2^{1}, 2^{1})$, são capazes de gerar melhores acurácias.  

\begin{figure}[h]
	\centering
	\includegraphics[width=0.9\textwidth]{Pictures/cap_14/ELM_1.png}
	\caption{Atuações bem-sucedidas dos \textit{kernels} compatíveis com os conjuntos de dados.}
	\label{fig:ELM1}
\end{figure}


\begin{figure}[h]
	\centering
	\includegraphics[width=0.9\textwidth]{Pictures/cap_14/ELM_2.png}
	\caption{Atuações malsucedidas do \textit{kernel} Linear em conjuntos de dados não-linearmente separáveis.}
	\label{fig:ELM2}
\end{figure}

\clearpage

\paragraph{Siga as instruções: } \vspace{0.50cm}
\begin{remark}{1}
Faça o \textit{Download} do \textit{script libsvm\_parameters.py} responsável pelos cliques automatizados no
\href{https://drive.google.com/drive/u/0/folders/1TRkNxE6dyVke4-0gUSNu8E33a2bKZtrc}{\chadded{presente link (pasta Cap. 6)}}. 
\end{remark}

\begin{remark}{2}
No console, para otimizar os parâmetros do classificador SVM. 

\begin{verbatim}
    python libsvm_parameters.py
\end{verbatim}
\end{remark}
