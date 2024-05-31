"""
Código desenvolvido por:

Prof. Dr. Sidney Lima
Universidade Federal de Pernambuco
Departamento de Eletrônica e Sistemas
"""
from libsvm.svmutil import *
import argparse
import sys,string
#========================================================================
class svm():
	def main(self, dataset, ):
		# Carregar e Preparar o Conjunto de Dados
		y, x = svm_read_problem(dataset)
		# 90% da base reservada para o treinamento
		ninetyPercent = round(0.9*len(y))
		# Treinamento do SVM com 90% das primeiras amostras.
		m = svm_train(y[:ninetyPercent], x[:ninetyPercent], '-s 4')
		# EMQ de treino da rede pós-treinada com 90% das primeiras amostras.
		print('Train Squared Error')
		p_label, p_acc, p_val = svm_predict(y[:ninetyPercent], x[:ninetyPercent], m)
		# Acurácia de teste da rede pós-treinada contendo as amostras de 90% + 1 até o final.
		print('Test Squared Error')
		p_label, p_acc, p_val = svm_predict(y[ninetyPercent:], x[ninetyPercent:], m)
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
	ff = svm()
	ff.main(opts)
