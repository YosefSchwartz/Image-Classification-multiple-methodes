# imports
import numpy as np
import os
from reduce_dimension import Principal_Component_Analysis
from preprocessing import get_data
from algorithms import k_nearest_neighbors, MultiLayer_Perceptron, Support_Vector_Machine

# Main program
if __name__ == '__main__':
    # resize()
    # del_unknown()

    x_train, x_test, y_train, y_test = get_data()
    # KNN
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)

    # MLP
    # MultiLayer_Perceptron(x_train, x_test, y_train, y_test)

    # SVM
    Support_Vector_Machine(x_train, x_test, y_train, y_test, True)
    x_train_2D, x_test_2D, y_train, y_test = Principal_Component_Analysis(x_train, x_test, y_train, y_test,True,True)
    # max_train_0 = max(x_train_2D[0])
    # min_train_0 = min(x_train_2D[0])
    # max_test_0 = max(x_test_2D[0])
    # min_test_0 = min(x_test_2D[0])
    #
    # x_train_2D[0] = x_train_2D[0] - min_train_0
    # x_train_2D[0] = x_train_2D[0] / (max_train_0-min_train_0)
    #
    # x_test_2D[0] = x_test_2D[0] - min_test_0
    # x_test_2D[0] = x_test_2D[0] / (max_test_0 - min_test_0)
    #
    # max_train_1 = max(x_train_2D[1])
    # min_train_1 = min(x_train_2D[1])
    # max_test_1 = max(x_test_2D[1])
    # min_test_1 = min(x_test_2D[1])
    #
    # x_train_2D[1] = x_train_2D[1] - min_train_1
    # x_train_2D[1] = x_train_2D[1] / (max_train_1 - min_train_1)
    #
    # x_test_2D[1] = x_test_2D[1] - min_test_1
    # x_test_2D[1] = x_test_2D[1] / (max_test_1 - min_test_1)

    Support_Vector_Machine(x_train_2D, x_test_2D, y_train, y_test, True)

    # LR
    # model(X, y)
    # CNN



# Reduce dimension PCA / TNSE
    #PCA
    # k_nearest_neighbors(x_train, x_test, y_train, y_test)
    # x_train_2D, x_test_2D, y_train, y_test = Principal_Component_Analysis(x_train, x_test, y_train, y_test,True,True)
    # k_nearest_neighbors(x_train_2D, x_test_2D, y_train, y_test)

