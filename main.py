# imports
import numpy as np
import os
from preprocessing import get_data
from algorithms import k_nearest_neighbors, CNN

# Main program
if __name__ == '__main__':
    X, y = get_data()

    # pre-processing
    # resize()
    # del_unknown()
#
#     # KNN
#     X,Y = get_data()
#     k_nearest_neighbors(X,y)

    CNN(X,y)
    # print(type(os.getcwd()))
    # print(type("hello"))
# # MLP

# SVM

# CNN

# Reduce dimension PCA / TNSE
