#!/usr/bin/python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, linear_model, decomposition, manifold, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import backend as K
import random

random.seed()
np.random.seed()

###########################
#getting and preparing data
###########################
data_org=pd.read_csv("rawdata.txt",sep="	", header=None)
data_org = data_org.sample(frac=1).reset_index(drop=True)
data = pd.DataFrame(np.array(data_org)[:,1:-1])
columns=["seed_chi2PerDoF", "seed_p", "seed_pt", "seed_nLHCbIDs",
        "seed_nbIT", "seed_nLayers", "seed_x", "seed_y",
         "seed_tx", "seed_ty"]
data[data.shape[1]] = np.sqrt(data[6]*data[6]+data[7]*data[7])
columns.append('seed_r')
data[data.shape[1]] = np.arctan(data[7]/data[6])
columns.append('seed_angle')
data[data.shape[1]] = np.arctanh(data[2]/data[1])
columns.append('seed_pseudorapidity')
data[1] = np.log(data[1])
data[2] = np.log(data[2])
data.columns = columns

sc = preprocessing.StandardScaler()
data = pd.DataFrame(sc.fit_transform(data))

x = pd.DataFrame(np.array(data[0:1400000]))
y = pd.DataFrame(np.array(data_org)[0:1400000,0])

x_test = pd.DataFrame(np.array(data[1400000:1700000]))
y_test = pd.DataFrame(np.array(data_org)[1400000:1700000,0])

tab = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

##########################
#setting and running model
##########################
model = Sequential()
model.add(Dense(64, activation='relu', input_dim = np.count_nonzero(tab)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
            optimizer='adam', metrics=['accuracy'])
callback = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='custom_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(x.iloc[:, tab==1].values, y.values, validation_split=0.2, epochs=50, callbacks=callback, verbose=1)

#############################################
#printing training, validate and test results
#############################################
model.load_weights("custom_model.h5")
print('\n\n--------\n')
print('Training result (loss and accuracy)')
print(history.history['loss'][-5], history.history['acc'][-5])
print('\n\n--------\n')
print('Validation result (loss and accuracy)')
print(history.history['val_loss'][-5], history.history['val_acc'][-5])
print('\n--------\n')
print('Test result (loss and accuracy)')
print(model.evaluate(x_test.iloc[:, tab==1].values, y_test.values))
print('\n--------\n')

#########################################
#ploting loss and accuracy through epochs
#########################################
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

K.clear_session()
