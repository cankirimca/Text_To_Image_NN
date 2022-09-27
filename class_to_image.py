#################################################
###############  Class-to-image Code ############
#################################################

import numpy as np
import PIL
from PIL import Image
import sys
import os
import scipy.io
import h5py

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, stem, scatter, xlim, ylim
from math import pi
import time
from numpy import exp, mean
from numpy.linalg import norm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Flatten, Dense, Reshape, Conv1D
from tensorflow.keras.optimizers import Adam
from google.colab import drive
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
drive.mount("/content/gdrive", force_remount=True)

pf = open("/content/gdrive/MyDrive/dataset/vae1.pickle", "rb")
class_to_image_dataset = pickle.load(pf)
trainX = class_to_image_dataset['trainX']
trainY = class_to_image_dataset['trainY']
testX = class_to_image_dataset['testX']
testY = class_to_image_dataset['testY']
valX = class_to_image_dataset['valX']
valY = class_to_image_dataset['valY']

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
trainY = scaler.fit_transform(trainY)
valY = scaler.transform(valY)
testY = scaler.transform(testY)

model = keras.Sequential()
model.add(Dense(128, input_shape = (102,), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(400, activation='tanh'))
model.compile(loss="cosine_similarity", optimizer = 'adam')

model.fit(to_categorical(trainX-1, 102), trainY, validation_data=(to_categorical(valX-1, 102), valY), epochs=100)

decoder_model = keras.models.load_model("/content/gdrive/MyDrive/dataset/final/decoder_model")
vae_decoder_model = keras.models.load_model("/content/gdrive/MyDrive/dataset/final/vae_decoder_model")

model.save("/content/gdrive/MyDrive/dataset/final/class_to_image_model")
