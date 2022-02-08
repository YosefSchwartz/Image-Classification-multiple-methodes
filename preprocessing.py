from PIL import Image
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


project_path = os.cwd() + "/"
resized_folder = (project_path + "images_resized/")
original_folder = (project_path + "images_original/")
dirs = os.listdir(original_folder)


def resize():
    """
    Resize all images to wXh, and convert RGBA to RGB (to save is as JPG)
    """
    w = 32
    h = 32
    for item in dirs:
        if os.path.isfile(original_folder + item):
            im = Image.open(original_folder + item)
            im = im.convert('RGB')
            imResize = im.resize((w, h), Image.ANTIALIAS)
            imResize.save(resized_folder + item, 'JPEG', quality=90)


def del_unknown():
    """
    Clean images that we haven't information about
    """
    details = pd.read_csv(project_path + "images.csv")
    images_name = details.iloc[:, 0]
    images_name = images_name.values.tolist()
    for item in dirs:
        f, e = os.path.splitext(item)
        if f not in images_name:
            os.remove(resized_folder + item)


def get_data():
    data = []
    labels = []
    details = pd.read_csv(project_path + "images.csv")
    images_names = details.iloc[:, 0]
    images_names = images_names.values.tolist()
    images_labels = details.iloc[:, 3]
    images_labels = images_labels.values.tolist()
    for item in dirs:
        f, e = os.path.splitext(item)
        if f in images_names:
            index = images_names.index(f)
            label = images_labels[index]
            img = Image.open(project_path+"images_resized/"+item)
            array = np.array(img)
            data.append(array)
            labels.append(label)
    return data, labels



def CNN(data, labels):
    # split the data in 60:20:20 for train:valid:test dataset
    train_size = 0.6

    # In the first step we will split the data in training and remaining dataset
    X_train, X_rem, y_train, y_rem = train_test_split(data, labels, train_size=0.6)

    # Now since we want the valid and test size to be equal (20% each of overall data).
    # we have to define valid_size=0.5 (that is 50% of remaining data)
    test_size = 0.5
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)


if __name__ == '__main__':
    # resize()
    # del_unknown()
    data, labels = get_data()
    # print(data)
    CNN(data, labels)
