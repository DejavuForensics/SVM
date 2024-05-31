"""
Code developed by:

Prof. Dr. Sidney Lima
Federal University of Pernambuco
Department of Electronics and Systems
"""
from libsvm.svmutil import *
import argparse
import sys,string
#========================================================================
class svm():
	def main(self, dataset, ):
		# Load and Prepare the Dataset
		y, x = svm_read_problem(dataset)
		# 90% of the dataset reserved for training
		ninetyPercent = round(0.9*len(y))
		# Train the SVM with the first 90% of the samples.
		m = svm_train(y[:ninetyPercent], x[:ninetyPercent])
		# Training accuracy of the network post-training with the first 90% of the samples.
		print('Train Accuracy')
		p_label, p_acc, p_val = svm_predict(y[:ninetyPercent], x[:ninetyPercent], m)
		# Test accuracy of the network post-training containing samples from 90% + 1 to the end.
		print('Test Accuracy')
		p_label, p_acc, p_val = svm_predict(y[ninetyPercent:], x[ninetyPercent:], m)
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
	ff = svm()
	ff.main(opts)
