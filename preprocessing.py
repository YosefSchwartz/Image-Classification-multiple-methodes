from PIL import Image
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


project_path = str(os.getcwd()) + "\\"
resized_folder = (project_path + "images_resized\\")
# original_folder = (project_path + "images_original\\")
# dirs_org = os.listdir(original_folder)
dirs_resize = os.listdir(resized_folder)

# def resize():
#     """
#     Resize all images to wXh, and convert RGBA to RGB (to save is as JPG)
#     """
#     w = 32
#     h = 32
#     for item in dirs_org:
#         if os.path.isfile(original_folder + item):
#             im = Image.open(original_folder + item)
#             im = im.convert('RGB')
#             imResize = im.resize((w, h), Image.ANTIALIAS)
#             imResize.save(resized_folder + item, 'JPEG', quality=90)


# def del_unknown():
#     """
#     Clean images that we haven't information about
#     """
#     details = pd.read_csv(project_path + "images.csv")
#     images_name = details.iloc[:, 0]
#     images_name = images_name.values.tolist()
#     for item in dirs_resize:
#         f, e = os.path.splitext(item)
#         if f not in images_name:
#             os.remove(resized_folder + item)
#

def get_data():
    data = []
    labels = []
    details = pd.read_csv(project_path + "images.csv")
    images_names = details.iloc[:, 0]
    images_names = images_names.values.tolist()
    images_labels = details.iloc[:, 3]
    images_labels = images_labels.values.tolist()
    for item in dirs_resize:
        f, e = os.path.splitext(item)
        if f in images_names:
            index = images_names.index(f)
            label = images_labels[index]
            img = Image.open(project_path+"images_resized\\"+item)
            array = np.array(img)
            data.append(array)
            labels.append(label)

    data = np.array(data)
    data = data.reshape((data.shape[0], -1))
    # Normalize
    data = data / 255

    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    return X_train, X_test, y_train, y_test


