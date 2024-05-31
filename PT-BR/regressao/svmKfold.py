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
#========================================================================
class svmKfold():
	def main(self, dataset):
		# Carregar e Preparar o Conjunto de Dados
		y, x = svm_read_problem(dataset)
		# Configurar o k-Fold
		k = 10
		# O parâmetro shuffle server para randomizar (embaralhar) as amostras
		np.random.seed(1)
		kf = KFold(n_splits=k, shuffle=False)
		# Realizar a validação cruzada por k-Fold 
		accuracies_train = []
		accuracies_test = []
		# Repetição do processo por k vezes
		for train_index, test_index in kf.split(x):
			x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
			y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
			# Treinamento do SVM
			m = svm_train(y_train, x_train, '-s 4')
			#Acurácia do treino
			p_label, p_acc_train, p_val = svm_predict(y_train, x_train, m)
			# Acurácia do teste
			p_label, p_acc_test, p_val = svm_predict(y_test, x_test, m)   
			accuracies_train.append(p_acc_train[1])
			accuracies_test.append(p_acc_test[1])

		# Calcular a média e o desvio padrão da acurácia do treino
		mean_accuracy_train = np.mean(accuracies_train)
		std_accuracy_train = np.std(accuracies_train)

		# Calcular a média e o desvio padrão da acurácia do teste
		mean_accuracy_test = np.mean(accuracies_test)
		std_accuracy_test = np.std(accuracies_test)	
		print(f"Train Squared Error: {mean_accuracy_train:f} ± {std_accuracy_train:f}")
		print(f"Test Squared Error: {mean_accuracy_test:f} ± {std_accuracy_test:f}")
#========================================================================
def setOpts(argv):                         
	parser = argparse.ArgumentParser()
	parser.add_argument('-dataset',dest='dataset',action='store', 
		default='bodyfat_scale', help='Filename of dataset')
		
	arg = parser.parse_args()
	return(arg.__dict__['dataset'])	
#========================================================================
if __name__ == "__main__":
	opts = setOpts(sys.argv[1:])
	ff = svmKfold()
	ff.main(opts)
