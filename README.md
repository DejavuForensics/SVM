# SVM

## SVM classifier: K-fold
Cross-validation is a statistical technique. Researchers use it to assess the performance of a machine learning model. It divides the data set into parts, or ‘folds’. You can train and test the model many times on different data subsets. The aim is to ensure that the model generalises well to new and unseen data. The k-fold method is a type of cross-validation. In it, the data set is randomly split into k equal subsets (or folds). The k-fold involves the following steps:

- **Data division**: the data set is divided into k approximately equal parts.

- **Training and testing**: for each of the k folds, the model is trained using k-1 folds and tested on the remaining fold.

- **Repeating the process**: this process is repeated k times, each time with a different fold acting as the test set.

- **Average and standard deviation of results**: performance metrics are calculated for each of the k runs. For each run, we calculate the average and standard deviation. We then average these metrics. This gives the model's final performance.




# PT-BR:
## Classificador SVM - K-fold
A validação cruzada é uma técnica estatística usada para avaliar o desempenho de um modelo de aprendizado de máquina. Ela divide o conjunto de dados em várias partes, ou "dobras", para que o modelo possa ser treinado e testado múltiplas vezes em diferentes subconjuntos dos dados. O objetivo é garantir que o modelo generalize bem para dados novos e não vistos,
O método k-fold é uma forma específica de validação cruzada onde o conjunto de dados é dividido aleatoriamente em k subconjuntos (ou folds) aproximadamente iguais. O k-fold envolve os seguintes passos:

- **Divisão dos dados**: o conjunto de dados é dividido em k partes aproximadamente iguais.

- **Treinamento e teste**: para cada uma das k dobras, o modelo é treinado utilizando k-1 dobras e testado na dobra restante.

- **Repetição do processo**: esse processo é repetido k vezes, cada vez com uma dobra diferente atuando como conjunto de teste.

- **Média e desvio padrão dos resultados**: as métricas de desempenho são calculadas para cada uma das k execuções e, em seguida, a média dessas métricas é computada para obter uma estimativa final do desempenho do modelo.

