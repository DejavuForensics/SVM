"""
Código desenvolvido por:

Prof. Dr. Sidney Lima
Universidade Federal de Pernambuco
Departamento de Eletrônica e Sistemas
"""
from libsvm.svmutil import *
from sklearn.model_selection import KFold
import numpy as np
import argparse
import sys,string
from time import process_time
#========================================================================
class svmParameters():
	def main(self, dataset):
		# Carregar e Preparar o Conjunto de Dados
		y, x = svm_read_problem(dataset)
		
		cost_vector = []
		cost_vector.append(10 ** 0)
		cost_vector.append(10 ** 3)   
		gamma_vector = []
		gamma_vector.append(10 ** 0)

		min_acc = 101
		max_acc = -1
		min_kernel = -1
		max_kernel = -1
		min_cost = 0
		max_cost = 0
		min_gamma = 0
		max_gamma = 0
		
		min_mean_accuracy_train = 101
		min_std_accuracy_train = 101
		min_mean_accuracy_test = 101
		min_std_accuracy_test = 101
		
		max_mean_accuracy_train = -1 
		max_std_accuracy_train = -1 
		max_mean_accuracy_test = -1
		max_std_accuracy_test = -1

		for t in range(4):
			for c in range(len(cost_vector)):
				for g in range(len(gamma_vector)):
					mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test, mean_train_time, std_train_time, mean_test_time, std_test_time = \
					svmKfold(y, x, t, cost_vector[c], gamma_vector[g])
					if(mean_accuracy_test<min_acc):
						min_acc = mean_accuracy_test
						min_kernel = t
						min_cost = cost_vector[c]
						min_gamma = gamma_vector[g]
						min_mean_accuracy_train = mean_accuracy_train
						min_std_accuracy_train =std_accuracy_train
						min_mean_accuracy_test = mean_accuracy_test
						min_std_accuracy_test = std_accuracy_test
						min_mean_train_time = mean_train_time
						min_std_train_time = std_train_time
						min_mean_test_time = mean_test_time
						min_std_test_time = std_test_time
					if(mean_accuracy_test>max_acc):
						max_acc = mean_accuracy_test
						max_kernel = t
						max_cost = cost_vector[c]
						max_gamma = gamma_vector[g]
						max_mean_accuracy_train = mean_accuracy_train 
						max_std_accuracy_train = std_accuracy_train 
						max_mean_accuracy_test = mean_accuracy_test
						max_std_accuracy_test = std_accuracy_test
						max_mean_train_time = mean_train_time
						max_std_train_time = std_train_time
						max_mean_test_time = mean_test_time
						max_std_test_time = std_test_time

		print(f'...........................................')
		print(f"Pior Acurácia Média de Treino: {min_mean_accuracy_train:.2f}% ± {min_std_accuracy_train:.2f}%")
		print(f"Pior Acurácia Média de Teste: {min_mean_accuracy_test:.2f}% ± {min_std_accuracy_test:.2f}%")
		print('Pior kernel: ' + kernel_str(min_kernel))
		print(f'Pior conf. de cost: {min_cost:.3f}')
		print(f'Pior conf. de gamma: {min_gamma:.3f}')
		print(f"Pior tempo médio de treino: {min_mean_train_time}% ± {min_std_train_time}")
		print(f"Pior tempo médio de teste: {min_mean_test_time} ± {min_std_test_time}")
		print(f'...........................................')
		print(f"Melhor Acurácia Média de Treino: {max_mean_accuracy_train:.2f}% ± {max_std_accuracy_train:.2f}%")
		print(f"Melhor Acurácia Média de Teste: {max_mean_accuracy_test:.2f}% ± {max_std_accuracy_test:.2f}%")
		print('Melhor Kernel: ' + kernel_str(max_kernel))
		print(f'Melhor conf. de Cost: {max_cost:.3f}')
		print(f'Melhor conf. de Gamma: {max_gamma:.3f}')
		print(f"Melhor tempo médio de treino: {max_mean_train_time}% ± {max_std_train_time}")
		print(f"Melhor tempo médio de teste: {max_mean_test_time} ± {max_std_test_time}")

#========================================================================	
def kernel_str(t):

	if (t==0): 
		str_kernel = 'Linear' 
	elif (t==1): 
		str_kernel = 'Polynomial'
	elif (t==2): 
		str_kernel = 'Radial Basis Function'
	elif (t==3):  
		str_kernel = 'Sigmoid'
	return str_kernel
#========================================================================
def svmKfold(y, x, t, cost, gamma):		
	# Configurar o k-Fold
	k = 10
	# O parâmetro shuffle server para randomizar (embaralhar) as amostras
	np.random.seed(1)
	kf = KFold(n_splits=k, shuffle=True)
	# Realizar a validação cruzada por k-Fold 
	accuracies_train = []
	accuracies_test = []
	processing_time_train = []
	processing_time_test = []
	# Repetição do processo por k vezes
	for train_index, test_index in kf.split(x):
		#Divisão dos dados entre treino e teste
		x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
		y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
		# Treinamento do SVM
		start_time_train = process_time()
		m = svm_train(y_train, x_train, '-t ' + str(t) + ' -c ' + str(cost) + ' -g ' + str(gamma))
		end_time_train = process_time()
		#Acurácia do treino
		p_label, p_acc_train, p_val = svm_predict(y_train, x_train, m)
		# Acurácia do teste
		start_time_test = process_time()
		p_label, p_acc_test, p_val = svm_predict(y_test, x_test, m)   
		end_time_test = process_time()
		accuracies_train.append(p_acc_train[0])
		accuracies_test.append(p_acc_test[0])
		processing_time_train.append(end_time_train - start_time_train)
		processing_time_test.append(end_time_test-start_time_test)

	# Calcular a média e o desvio padrão da acurácia do treino
	mean_accuracy_train = np.mean(accuracies_train)
	std_accuracy_train = np.std(accuracies_train)

	# Calcular a média e o desvio padrão da acurácia do teste
	mean_accuracy_test = np.mean(accuracies_test)
	std_accuracy_test = np.std(accuracies_test)

	# mean train time
	mean_train_time = np.mean(processing_time_train)
	std_train_time = np.std(processing_time_train)
	# mean test time
	
	mean_test_time = np.mean(processing_time_test)
	std_test_time = np.std(processing_time_test)
		
	return mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test, mean_train_time, std_train_time, mean_test_time, std_test_time
#========================================================================
def setOpts(argv):                         
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset',dest='dataset',action='store', 
		default='heart_scale', help='Filename of dataset')
		
	arg = parser.parse_args()
	return(arg.__dict__['dataset'])	
#========================================================================
if __name__ == "__main__":
	opts = setOpts(sys.argv[1:])
	ff = svmParameters()
	ff.main(opts)
