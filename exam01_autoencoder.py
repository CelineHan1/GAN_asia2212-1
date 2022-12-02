import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)
decoded = Dense(784, activation='sigmoid')
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()






