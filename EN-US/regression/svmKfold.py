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
import sys, string
#========================================================================
class svmKfold():
    def main(self, dataset):
        # Load and Prepare the Dataset
        y, x = svm_read_problem(dataset)
        # Set up k-Fold
        k = 10
        # The shuffle parameter serves to randomize (shuffle) the samples
        np.random.seed(1)
        kf = KFold(n_splits=k, shuffle=False)
        # Perform k-Fold cross-validation
        accuracies_train = []
        accuracies_test = []
        # Repeat the process k times
        for train_index, test_index in kf.split(x):
            x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            # Train the SVM
            m = svm_train(y_train, x_train, '-s 4')
            # Training accuracy
            p_label, p_acc_train, p_val = svm_predict(y_train, x_train, m)
            # Test accuracy
            p_label, p_acc_test, p_val = svm_predict(y_test, x_test, m)
            accuracies_train.append(p_acc_train[1])
            accuracies_test.append(p_acc_test[1])

        # Calculate the mean and standard deviation of training accuracy
        mean_accuracy_train = np.mean(accuracies_train)
        std_accuracy_train = np.std(accuracies_train)

        # Calculate the mean and standard deviation of test accuracy
        mean_accuracy_test = np.mean(accuracies_test)
        std_accuracy_test = np.std(accuracies_test)    
        print(f"Train Squared Error: {mean_accuracy_train:f} ± {std_accuracy_train:f}")
        print(f"Test Squared Error: {mean_accuracy_test:f} ± {std_accuracy_test:f}")
#========================================================================
def setOpts(argv):                         
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', dest='dataset', action='store', 
        default='bodyfat_scale', help='Filename of dataset')
        
    arg = parser.parse_args()
    return(arg.__dict__['dataset'])    
#========================================================================
if __name__ == "__main__":
    opts = setOpts(sys.argv[1:])
    ff = svmKfold()
    ff.main(opts)
