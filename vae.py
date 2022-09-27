#################################################
######### Variational Autoencoder Code ##########
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
drive.mount("/content/gdrive")


image_size = (128, 128)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/gdrive/MyDrive/dataset/flowers",
    validation_split=1-0.7019607843137254,
    labels=None,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/gdrive/MyDrive/dataset/flowers",
    validation_split=1-0.7019607843137254,
    labels=None,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ]
)

def change_inputs(images):
      #return tf.reshape(images, (32, 224*224*3))/255.0, tf.reshape(images, (32, 224*224*3))/255.0 #images-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]),images-np.reshape([123.68, 116.78, 103.94], [1, 1, 3])
  return (data_augmentation(images)-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]))/127.5, (data_augmentation(images)-np.reshape([123.68, 116.78, 103.94], [1, 1, 3]))/127.5
train_ds = train_ds.map(change_inputs)
val_ds = val_ds.map(change_inputs)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

selected_layers = ['block1_conv1', 'block2_conv2',"block3_conv3" ,'block4_conv3','block5_conv4']
selected_layer_weights = [1.0/32., 2.0/32. , 4.0/32. , 8.0/32. , 32.0/32.]

vgg = VGG19(weights='imagenet', include_top=False, input_shape=(128,128,3))
vgg.trainable = False
outputs = tf.concat([layers.Rescaling(w)(layers.Flatten()(vgg.get_layer(l).output)) for l,w in zip(selected_layers, selected_layer_weights)], -1)
model = keras.Model(vgg.input, outputs)

#means = tf.reshape(tf.constant([123.68, 116.78, 103.94]), [1, 1, 3])# This is R-G-B for Imagenet

#@tf.function
def perceptual_loss(input_image , reconstruct_image):
    return K.mean(K.square(model(input_image*127.5) - model(reconstruct_image*127.5)), axis=-1)

input_img = keras.Input(shape=(128,128,3))

#x = data_augmentation(input_img)
x = layers.Conv2D(32, 3, activation='leaky_relu', padding='same', strides=2)(input_img)
x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2D(64, 3, activation='leaky_relu', padding='same', strides=2)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
x = layers.BatchNormalization()(x)
#x = layers.Dropout(0.1)(x)
x = layers.Conv2D(128, 3, activation='leaky_relu', padding='same', strides=2)(x)
encoder1_out = Flatten()(x)

encoder1 = keras.Model(input_img, encoder1_out, name='encoder1')

#--------------------------------------------------------------
ls=128
input_vae_enc= keras.Input(shape=(2048,), name='vae_enc')
z_mean = layers.Dense(ls)(input_vae_enc)
z_log_sigma = layers.Dense(ls)(input_vae_enc)


def sampling(args):
  z_mean, z_log_sigma = args
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], ls),
                            mean=0., stddev=1.0)
  return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

vae_encoder = keras.Model(input_vae_enc, [z_mean, z_log_sigma, z], name='vae_encoder')

#--------------------------------------------------------------

input_vae_dec = keras.Input(shape=(ls,), name='vae_dec')
vae_decoded = layers.Dense(2048, activation='leaky_relu')(input_vae_dec)
vae_decoder = keras.Model(input_vae_dec, vae_decoded, name='vae_decoder')

#--------------------------------------------------------------

latent_inputs = keras.Input(shape=(4*4*128,), name='z_sampling')
x = Reshape((4,4,128))(latent_inputs)
x = layers.Conv2DTranspose(64, 3, activation='leaky_relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2DTranspose(128, 3, activation='leaky_relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2DTranspose(128, 3, activation='leaky_relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2DTranspose(64, 3, activation='leaky_relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
x = layers.Conv2DTranspose(32, 3, activation='leaky_relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D(2, interpolation="bilinear")(x)
decoded = layers.Conv2DTranspose(3, 3, activation='tanh', padding='same')(x)
decoder = keras.Model(latent_inputs, decoded, name='decoder')
#--------------------------------------------------------------

# instantiate VAE model
outputs = decoder(encoder1(input_img))
autoencoder_std = keras.Model(input_img, outputs, name='autoencoder')

z_mean1, z_log_sigma1, z1 = vae_encoder(encoder1(input_img))
#save z_mean1
#decoder(vae_decoder(z_mean1))
outputs2 = decoder(vae_decoder(z1))
vae = keras.Model(input_img, outputs2, name='vae_autoencoder')

reconstruction_loss = K.mean(K.square(input_img - outputs2), axis = [1,2,3])
kl_loss = -0.5*(1 + z_log_sigma1 - K.square(z_mean1) - K.exp(z_log_sigma1))
kl_loss = K.sum(kl_loss, axis=1)
vae_loss = reconstruction_loss*10000. + kl_loss + 0.2*(perceptual_loss(input_img, outputs2))
vae.add_loss(vae_loss)
vae.compile(optimizer='adam', metrics=[perceptual_loss,"mse"])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
h = vae.fit(train_ds, epochs=40, batch_size=32, callbacks=[callback], validation_data=val_ds)

def stretch(x, in_max, in_min):
      return (x - in_min) * 255 / (in_max - in_min)

k=vae.predict(val_ds)
k[:,:,:,0] = stretch(k[:,:,:,0], np.max(k[:,:,:,0]), np.min(k[:,:,:,0]))
k[:,:,:,1] = stretch(k[:,:,:,1], np.max(k[:,:,:,1]), np.min(k[:,:,:,1]))
k[:,:,:,2] = stretch(k[:,:,:,2], np.max(k[:,:,:,2]), np.min(k[:,:,:,2]))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)

  plt.imshow((k[i+45]).astype("uint8"))
  plt.title(str(i))
  plt.axis("off")

#save models
encoder1.save("/content/gdrive/MyDrive/dataset/vae/encoder1_model")
vae_encoder.save("/content/gdrive/MyDrive/dataset/vae/vae_encoder_model")
decoder.save("/content/gdrive/MyDrive/dataset/vae/decoder_model")
vae_decoder.save("/content/gdrive/MyDrive/dataset/vae/vae_decoder_model")