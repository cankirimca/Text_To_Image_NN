###########################################################
####### Main Program Code (Combining all models) ##########
###########################################################

import gensim.models
import random
import pickle
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import strip_punctuation, remove_stopwords
import gensim.downloader as gd
from google.colab import drive
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.layers import Flatten, Dense, Reshape, Bidirectional, GRU, Dropout, LSTM
from tensorflow.keras import layers
import matplotlib.pyplot as plt
drive.mount("/content/gdrive")

#change this to the current path of the models
path = "/content/gdrive/MyDrive/dcgan/NN Project/LSTM+VAE/"

pf = open(path + "vae1.pickle", "rb")
class_to_image_dataset = pickle.load(pf)
trainY = class_to_image_dataset['trainY']
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
trainY = scaler.fit_transform(trainY)

lstm = keras.models.load_model(path + "desc_to_class_model")
decoder_model = keras.models.load_model(path + "decoder_model")
vae_decoder_model = keras.models.load_model(path + "vae_decoder_model")
class_to_image_model = keras.models.load_model(path + "class_to_image_model")

wv_model = gensim.models.KeyedVectors.load(path + "gensim.model")

def lstm_predict(line):
  new_line = strip_punctuation(line)
  new_line = remove_stopwords(new_line)
  splitted = new_line.split()           
  inp = []
  for k in splitted:
    try:
      inp.append(wv_model[k])
    except KeyError as k:
      pass
      print(str(k))
  while len(inp) < 30:
      inp.append(np.zeros(50,))  
  
  lp = np.argmax(lstm.predict(np.array([inp])))
  print(lp)
  p = class_to_image_model.predict(np.array([to_categorical(lp, 102)]))
  img = scaler.inverse_transform(np.array(p))
  img = decoder_model.predict(vae_decoder_model.predict(img))
  return (img[0]*127.5-np.reshape([123.68, 116.78, 103.94], [1, 1, 3])).astype("uint8")

#change the description to see other images
#the model works best if you pick descriptions from the dataset
plt.imshow(lstm_predict("the flowers have round white petals and a yellow stamen"))