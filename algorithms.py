from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import tensorflow as tf
from keras.callbacks import Callback, LambdaCallback
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation

def evaluate_test(model, X_test, y_test, test_acc, test_loss):
    res_test = model.evaluate(X_test, y_test)
    test_loss.append(res_test[0])
    test_acc.append(res_test[1])

import numpy as np


# KNN
def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    neighbors = np.arange(1, 12)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

# MLP
def MultiLayer_Perceptron(x_train, x_test, y_train, y_test):
    model = Sequential([
        # dense layer 1

        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),

        # output layer
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    epochs = 6

    test_acc = []
    test_loss = []
    myCallback = LambdaCallback(on_epoch_end=lambda batch, logs: evaluate_test(model, x_test, y_test, test_acc, test_loss))
    details = model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.25, callbacks=[myCallback])

    val_acc_list = details.history['val_accuracy']
    acc_list = details.history['accuracy']
    loss_list = details.history['loss']
    val_loss_list = details.history['val_loss']

    x_axis = []
    k = 1
    for i in range(1, epochs+1):
        x_axis.append(k)
        k = k + 1

    # Loss graph
    plt.plot(x_axis, test_loss, label="test loss")
    plt.plot(x_axis, val_loss_list, label="validation loss")
    plt.plot(x_axis, loss_list, label="train loss")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation loss')
    plt.legend()
    plt.show()

    # Accuracy graph
    plt.plot(x_axis, test_acc, label="test accuracy")
    plt.plot(x_axis, val_acc_list, label="validation accuracy")
    plt.plot(x_axis, acc_list, label="train accuracy")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation accuracy')
    plt.legend()
    plt.show()


def CNN(X_train, X_test, y_train, y_test):
    ROW = 32
    COL = 32
    CHANNEL = 3
    X_train = X_train.reshape(X_train.shape[0], ROW, COL, CHANNEL)
    X_test = X_test.reshape(X_test.shape[0], ROW, COL, CHANNEL)
    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(32))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    epochs = 6
    test_error = []
    test_acc = []
    myCallback = LambdaCallback(on_epoch_end=lambda batch, logs: evaluate_test(model, X_test, y_test, test_acc, test_error))
    details = model.fit(X_train, y_train, batch_size=128, epochs=epochs, validation_split=0.214, callbacks=[myCallback])

    val_acc_list = details.history['val_accuracy']
    acc_list = details.history['accuracy']
    loss_list = details.history['loss']
    val_loss_list = details.history['val_loss']
    results = model.evaluate(X_test, y_test, verbose=0)

    print("final accuracy: {:5.2f}%".format(100 * results[1]))

    x_axis = []
    k = 1
    for i in range(1, epochs + 1):
        x_axis.append(k)
        k = k + 1

    # Loss graph
    plt.plot(x_axis, test_error, label="test loss")
    plt.plot(x_axis, val_loss_list, label="validation loss")
    plt.plot(x_axis, loss_list, label="train loss")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation loss')
    plt.legend()
    plt.show()

    # Accuracy graph
    plt.plot(x_axis, test_acc, label="test accuracy")
    plt.plot(x_axis, val_acc_list, label="validation accuracy")
    plt.plot(x_axis, acc_list, label="train accuracy")
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.title('train and validation accuracy')
    plt.legend()
    plt.show()


# SVM
def Support_Vector_Machine(x_train, x_test, y_train, y_test, sample_of_data=False):
    ##########################
    # if sample_of_data is true we sample 10% from the hole data
    if sample_of_data is True:
        num_train = len(x_train) // 10
        num_test = len(x_test) // 10
        train_indices = np.random.randint(0, len(x_train), num_train)
        test_indices = np.random.randint(0, len(x_test), num_test)

        x_train = x_train[train_indices, :]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices, :]
        y_test = y_test[test_indices]
    #############################################
    # Run SVM for classification
    model = SVC(C=1, kernel="poly", gamma="auto")
    model.fit(x_train, y_train)

    # Check the accuracy on train and test
    prediction = model.predict(x_train)
    train_acc = 100 * np.sum(prediction == y_train) / len(y_train)
    prediction = model.predict(x_test)
    test_acc = 100 * np.sum(prediction == y_test) / len(y_test)
    print('SVM: train accuracy = {:.2f}%, '
          'test accuracy = {:.2f}%'.format(train_acc, test_acc))


def logistic(X_train, X_test, Y_train, Y_test):
    X_train = X_train[0:3200]
    Y_train = Y_train[0:3200]

    model =LogisticRegression(max_iter=100000)
    model.fit(X_train, Y_train)
    print("Final accuracy: {:5.2f}%".format(100 * model.score(X_test, Y_test)))

