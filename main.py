# imports
import numpy as np
import os
from reduce_dimension import Principal_Component_Analysis
from preprocessing import get_data
from algorithms import k_nearest_neighbors, MultiLayer_Perceptron, Support_Vector_Machine

from algorithms import k_nearest_neighbors, CNN
from algorithms import k_nearest_neighbors, CNN, logistic
import matplotlib.pyplot as plt

# Main program
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    # MLP
    # MultiLayer_Perceptron(x_train, x_test, y_train, y_test)

    # CNN
    # CNN(x_train, x_test, y_train, y_test)

    #logistic regression
    # logistic(x_train, x_test, y_train, y_test)

    # PCA
    # x_train_2D, x_test_2D, y_train, y_test = Principal_Component_Analysis(x_train, x_test, y_train, y_test, True, False)

    # SVM
    # Support_Vector_Machine(x_train, x_test, y_train, y_test, True)

    # SVM after PCA
    # Support_Vector_Machine(x_train_2D, x_test_2D, y_train, y_test, True)

    # KNN
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)

    # KNN after PCA
    # k_nearest_neighbors(x_train_2D, x_test_2D, y_train, y_test)

