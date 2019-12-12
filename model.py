import keras
from keras.datasets import fashion_mnist 
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

X = np.load('x.npy')
y = np.load('y.npy')
Y_one_hot = to_categorical(y)
#(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()


X_train, X_test, y_train, y_test = train_test_split(X, Y_one_hot, test_size=0.20, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

layers = [[64,64], [64,32], [32,64], [32,32], [64,32,16], [16,32,64], [64,64,64], [32,32,32],[16,16,16]]
for l in layers: 
  model = Sequential()
  model.add(Conv2D(l[0], (3,3), input_shape=(28, 28,1)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(Conv2D(l[1], (3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  if(len(l) == 3):
    model.add(Conv2D(l[2], (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Flatten())
  model.add(Dense(64))

  model.add(Dense(43))
  model.add(Activation('softmax'))

  model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
  
  model_saver = ModelCheckpoint(str(layers.index(l))+".{epoch:02d}-{val_acc:.4f}.hdf5", monitor='val_acc', 
                                            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  callbacks_list = [model_saver]
  model.fit(X_train, y_train, batch_size=64, epochs=50,verbose=1,callbacks = callbacks_list
            ,validation_data=(X_valid, y_valid))




