# -*- coding: utf-8 -*-
"""data-set-creation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TmqZomXRAuBw_epgmZ22-usAbd1tZzGD
"""
# Importing the required functions and libraries
import numpy as np
import pandas as pd
from zipfile import ZipFile
import os
from sklearn.svm import LinearSVC, SVC
import argparse
import cv2
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.model_selection import KFold,StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast,RandomRotation,RandomCrop,RandomFlip,RandomZoom, Resizing
import pickle

#Connecting to Kaggle account and downloading the data needed for training
os.environ['KAGGLE_USERNAME'] = "XXX" # username from the json file
os.environ['KAGGLE_KEY'] = "XXX" # key from the json file
!kaggle datasets download -d moltean/fruits
!kaggle datasets download -d chandranaveenkumar/strawberry-texture-pics
!kaggle datasets download -d chandranaveenkumar/temp-test

# Extracting data from ZIP files 
file_name = "/content/fruits.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

file_name = "/content/strawberry-texture-pics.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')

file_name = "/content/temp-test.zip"
with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('done')


# Defining the directory's for training data and labels for each class 
Fruits_images_dict = {
    "healthy_strawberry":"/content/fruits-360_dataset/fruits-360/Training/Strawberry",
    "spoiled_strawberry":"/content/Spoiled_augmented/augmented"
}
Fruits_labels_dict = {
    'healthy_strawberry' : 0,
    'spoiled_strawberry':1
}

# Function for extracting the Local Binary Patterns
class LocalBinaryPatterns:
    #This function have been adapted from explination at "https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/" 
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
        self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, self.numPoints + 3),
        range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
        
#Defining the random Augmentations to be used during the training
transform = tf.keras.Sequential([
    RandomRotation((-0.5,0.5), fill_mode='wrap', interpolation='nearest'),
    RandomFlip("horizontal_and_vertical"),
    RandomZoom((-0.1,-0.5)),
    Resizing(100, 100, interpolation='nearest')
])


# Defining an descriptor for LBP and creating DATA and label holders
desc = LocalBinaryPatterns(20, 4)
data = []
labels = []
# Reading in training data and giving labels
for item,values in Fruits_images_dict.items():
    #print(str(dirs))
    #print(values)
    for dirname, _, filenames in os.walk(values):
        #print(len(filenames))
        for filename in filenames:
            #print(filename)
            image = cv2.imread(str(values+'/'+filename))
            # Augmenting the image
            if values == "/content/fruits-360_dataset/fruits-360/Training/Strawberry":
              image = tf.expand_dims(image, 0)
              augment = transform(image)
              transformed_image = augment[0]
              image = np.array(transformed_image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            data.append(hist)
            #imread function loads image in BGR format by default instead of RGB format. 
            labels.append(Fruits_labels_dict[str(item)])
            # Assining the respective image a label based on which directory it is present

print(labels)

# ML model definition and training
model = SVC()

cv = StratifiedKFold(n_splits=5)
param_grid = {
    'kernel':['linear'],
    'C' : [0.1,1,10,25, 75],
    'gamma': np.logspace(-10,10,10),   
}

model = GridSearchCV(model, param_grid,cv = cv, verbose = 50, n_jobs=-1)
model.fit(data, labels)

Test_images_dict = {
    'healthy_strawberry' : "/content/fruits-360_dataset/fruits-360//Test/Strawberry",
    'spoiled_strawberry' : "/content/Texture_pics/Texture_pics/Spoiled_fruits"
}

t_data = []
t_labels = []

print('\nBest Train Accuracy : %.2f'%model.best_score_, ' Best Params : ', str(model.best_params_))

# Saving the trained model
outputFilename ="Texture_based_helath"
with open(outputFilename, 'wb') as f:
        pickle.dump([model.best_estimator_], f)

#Reading the test data
for dirs,values in Test_images_dict.items():
    #print(str(dirs))
    print(values)
    #os.walk function returns three outputs in following sequence root directory,Sub directories & individual files names in all directories.
    for dirname, _, filenames in os.walk(values):
        #print(len(filenames))
        for filename in filenames:
            #print(filename)
            image = cv2.imread(str(values+'/'+filename))
            resz_image = cv2.resize(image,(100,100),interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(resz_image, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(gray)
            t_data.append(hist)
            #imread function loads image in BGR format by default instead of RGB format. 
            t_labels.append(Fruits_labels_dict[str(dirs)])
            # Assining the respective image a label based on which directory it is present

#Making predictions on test data and anlaysing the metrics 
p_labels = model.best_estimator_.predict(t_data)

fruits = ["healthy_strawberry","spoiled_strawberry"]

print(classification_report(t_labels, p_labels))

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(model, t_data, t_labels,
                                 display_labels=fruits,
                                 cmap=plt.cm.Blues,ax=ax)

plt.show()

diff_test = []
diff_labels = []

