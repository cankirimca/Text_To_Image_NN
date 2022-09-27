#################################################
################ Data Loading Code ##############
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
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.applications import VGG19
from keras import backend as K

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)


encoder1 = keras.models.load_model("/content/gdrive/MyDrive/dataset/vae/encoder1_model")
ls=400
vae_encoder = keras.models.load_model("/content/gdrive/MyDrive/dataset/vae/vae_encoder_model")

import random
from sklearn.model_selection import train_test_split
def str_to_vector(s, vocab):
    result = []
    tokens = s.split()
    for token in tokens:
        try:
            index = vocab[token.encode()]
        except KeyError:
            index = 0
        #result.append()
    return len(tokens)

def load_data():
    x = []
    y = []
    descriptions = []
    mat = scipy.io.loadmat('/content/gdrive/MyDrive/dataset/imagelabels.mat')
    labels = np.array(mat['labels'])[0]
    #load images in the train set
    for i in range(8189):
        print(i)
        i_str = str(i+1).zfill(5)
        label = labels[i]
        image = Image.open('/content/gdrive/MyDrive/dataset/flowers/image_' + i_str + ".jpg")
        image = image.resize((128,128))
        image = np.array(image)
        #determine the category of the sample
        y.append(image)
        x.append(label)

    shuffled = list(range(8189))
    random.shuffle(shuffled)
    x = np.array(x)
    y = np.array(y)
    x = x[shuffled]
    y = y[shuffled]
    trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)
    valX, testX, valY, testY = train_test_split(testX, testY, test_size=0.25)
                    
    return trainX, valX, testX, trainY, valY, testY

trainX, valX, testX, trainY, valY, testY = load_data()

test_enc, dummy1, dummy2 = vae_encoder.predict(encoder1.predict((testY-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]))/127.5))
train_enc, dummy1, dummy2 = vae_encoder.predict(encoder1.predict((trainY-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]))/127.5))
val_enc, dummy1, dummy2 = vae_encoder.predict(encoder1.predict((valY-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]))/127.5))

decoder_model = keras.models.load_model("/content/gdrive/MyDrive/dataset/vae/decoder_model")
vae_decoder_model = keras.models.load_model("/content/gdrive/MyDrive/dataset/vae/vae_decoder_model")
img1 = np.array([train_enc[500]])
img1 = decoder_model.predict(vae_decoder_model.predict(img1))
plt.figure(figsize=(10, 10))
plt.imshow((img1[0]*127.5-np.reshape([123.68, 116.78, 103.94], [1, 1, 3])).astype("uint8"))

import pickle
with open("vae1.pickle", "wb") as pf:
  pickle.dump({"trainX":trainX,"valX":valX,"testX":testX,"trainY":train_enc,"valY":val_enc,"testY":test_enc},pf)