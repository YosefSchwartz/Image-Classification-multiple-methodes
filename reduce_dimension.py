import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_reduced(x_train, x_test, y_train, y_test, sample_of_data=False):
    #############################################
    # Sample 10% from the data
    ##########################
    if sample_of_data is True:
        num_train = len(x_train) // 5
        num_test = len(x_test) // 5
        train_indices = np.random.randint(0, len(x_train), num_train)
        test_indices = np.random.randint(0, len(x_test), num_test)

        x_train = x_train[train_indices, :]
        y_train = y_train[train_indices]
        x_test = x_test[test_indices, :]
        y_test = y_test[test_indices]
    #############################################

    y_train_new = []
    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train_new.append("Adult")
        else:
            y_train_new.append("Kid")

    y_test_new = []
    for i in range(len(y_test)):
        if y_train[i] == 0:
            y_test_new.append("Adult")
        else:
            y_test_new.append("Kid")

    y_train_new_label = pd.DataFrame(y_train_new, columns=["labels"])
    y_test_new_label = pd.DataFrame(y_test_new, columns=["labels"])

    light_blue = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
    colors = {'Adult': light_blue, 'Kid': "r"}

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    s = axs[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train_new_label["labels"].map(colors),
                       cmap=plt.cm.Spectral, s=20, edgecolors='k')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in
               colors.items()]
    legend = axs[0].legend(handles=handles, title='label', loc='best')
    axs[0].add_artist(legend)
    axs[0].set_xticks(())
    axs[0].set_yticks(())
    axs[0].set_title('Train data visualized by 2D')

    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, label=k, markersize=8) for k, v in
               colors.items()]
    s = axs[1].scatter(x_test[:, 0], x_test[:, 1], c=y_test_new_label["labels"].map(colors),
                       cmap=plt.cm.Spectral, s=20, edgecolors='k')
    legend = axs[1].legend(handles=handles, title='label', loc='best')
    axs[1].add_artist(legend)
    axs[1].set_xticks(())
    axs[1].set_yticks(())
    axs[1].set_title('Test data visualized by 2D')

    plt.show()


def Principal_Component_Analysis(x_train, x_test, y_train, y_test, plot=False, sample_of_data=False):
    pca_train = PCA(2)
    x_train = pca_train.fit_transform(x_train)

    pca_test = PCA(2)
    x_test = pca_test.fit_transform(x_test)

    if plot is True:
        plot_reduced(x_train, x_test, y_train, y_test, sample_of_data)

    return x_train, x_test, y_train, y_test
