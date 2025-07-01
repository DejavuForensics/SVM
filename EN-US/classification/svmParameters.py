"""
Code developed by:

Prof. Dr. Sidney Lima
Federal University of Pernambuco
Department of Electronics and Systems
"""
from libsvm.svmutil import *
from sklearn.model_selection import KFold
import numpy as np
import argparse
import sys,string
#========================================================================
class svmParameters():
	def main(self, dataset):
		# Load and Prepare the Dataset
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
					mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test = \
					svmKfold(y, x, t, cost_vector[c], gamma_vector[g])
					if(mean_accuracy_test < min_acc):
						min_acc = mean_accuracy_test
						min_kernel = t
						min_cost = cost_vector[c]
						min_gamma = gamma_vector[g]
						min_mean_accuracy_train = mean_accuracy_train
						min_std_accuracy_train = std_accuracy_train
						min_mean_accuracy_test = mean_accuracy_test
						min_std_accuracy_test = std_accuracy_test
					if(mean_accuracy_test > max_acc):
						max_acc = mean_accuracy_test
						max_kernel = t
						max_cost = cost_vector[c]
						max_gamma = gamma_vector[g]
						max_mean_accuracy_train = mean_accuracy_train 
						max_std_accuracy_train = std_accuracy_train 
						max_mean_accuracy_test = mean_accuracy_test
						max_std_accuracy_test = std_accuracy_test

		print(f'...........................................')
		print(f"Worst Train Mean Accuracy: {min_mean_accuracy_train:.2f}% ± {min_std_accuracy_train:.2f}%")
		print(f"Worst Test Mean Accuracy: {min_mean_accuracy_test:.2f}% ± {min_std_accuracy_test:.2f}%")
		print('Worst kernel: ' + kernel_str(min_kernel))
		print(f'Worst cost conf.: {min_cost:.3f}')
		print(f'Worst gamma conf.: {min_gamma:.3f}')
		print(f'...........................................')
		print(f"Best Train Mean Accuracy: {max_mean_accuracy_train:.2f}% ± {max_std_accuracy_train:.2f}%")
		print(f"Best Test Mean Accuracy: {max_mean_accuracy_test:.2f}% ± {max_std_accuracy_test:.2f}%")
		print('Best Kernel: ' + kernel_str(max_kernel))
		print(f'Best Cost conf.: {max_cost:.3f}')
		print(f'Best Gamma conf.: {max_gamma:.3f}')

#========================================================================	
def kernel_str(t):

	if (t == 0): 
		str_kernel = 'Linear' 
	elif (t == 1): 
		str_kernel = 'Polynomial'
	elif (t == 2): 
		str_kernel = 'Radial Basis Function'
	elif (t == 3):  
		str_kernel = 'Sigmoid'
	return str_kernel
#========================================================================
def svmKfold(y, x, t, cost, gamma):		
	# Set up k-Fold
	k = 10
	# The shuffle parameter serves to randomize (shuffle) the samples
	kf = KFold(n_splits=k, shuffle=True)
	# Perform k-Fold cross-validation
	accuracies_train = []
	accuracies_test = []
	# Repeat the process k times
	for train_index, test_index in kf.split(x):
		# Split the data into training and testing sets
		x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
		y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
		# Train the SVM
		m = svm_train(y_train, x_train, '-t ' + str(t) + ' -c ' + str(cost) + ' -g ' + str(gamma))
		# Training accuracy
		p_label, p_acc_train, p_val = svm_predict(y_train, x_train, m)
		# Test accuracy
		p_label, p_acc_test, p_val = svm_predict(y_test, x_test, m)   
		accuracies_train.append(p_acc_train[0])
		accuracies_test.append(p_acc_test[0])

	# Calculate the mean and standard deviation of training accuracy
	mean_accuracy_train = np.mean(accuracies_train)
	std_accuracy_train = np.std(accuracies_train)

	# Calculate the mean and standard deviation of test accuracy
	mean_accuracy_test = np.mean(accuracies_test)
	std_accuracy_test = np.std(accuracies_test)
		
	return mean_accuracy_train, std_accuracy_train, mean_accuracy_test, std_accuracy_test
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
