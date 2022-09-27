#################################################
###### LSTM Code (Description-to-Class) #########
#################################################

import tensorflow
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
import gensim.downloader as gd
from google.colab import drive
from scipy.io import loadmat 
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Flatten, Dense, Reshape, Bidirectional, GRU, Dropout, LSTM
from tensorflow.keras import layers
drive.mount("/content/gdrive")

import pickle
with open('/content/gdrive/MyDrive/dataset/lstm1.pickle', 'rb') as f:
    data = pickle.load(f)

trainX = data['trainX']
valX = data['valX']
testX = data['testX']
trainY = data['trainY']
valY = data['valY']
testY = data['testY']
trainY = to_categorical(trainY-1, 102) #convert to one-hot encoding
valY = to_categorical(valY-1, 102)
testY = to_categorical(testY-1, 102)  

cc = tensorflow.keras.metrics.CategoricalCrossentropy(
    name="categorical_crossentropy", dtype=None, from_logits=False, label_smoothing=0
)
model=keras.Sequential()
model.add(LSTM(128))
model.add(Dense(102,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=["accuracy",cc])
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

h = model.fit(trainX, trainY, epochs=1000, batch_size=64, callbacks=[callback], shuffle=True, validation_data=(valX,valY))

import matplotlib.pyplot as plt

plt.plot(h.history['categorical_crossentropy'])
plt.plot(h.history['val_categorical_crossentropy'])
plt.title('model loss (categorical crossentropy)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("/content/gdrive/MyDrive/dataset/final/lstmloss.png")
plt.show()

pred = model.predict(testX)
correct = 0
for i in range(len(pred)): 
  if np.argmax(pred[i]) == np.argmax(testY[i]):
    correct += 1
accuracy = correct/len(pred)
print("accuracy=", accuracy)

model.save("/content/gdrive/MyDrive/dataset/final/desc_to_class_model")