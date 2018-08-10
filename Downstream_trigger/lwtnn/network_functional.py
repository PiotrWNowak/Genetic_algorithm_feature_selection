#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
import random

random.seed()
np.random.seed()

tab = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

inputs = Input(shape=(np.count_nonzero(tab),))
hidden1 = Dense(64, activation='relu')(inputs)
hidden2 = Dense(64, activation='relu')(hidden1)
hidden3 = Dense(64, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.load_weights("custom_model.h5")

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
