import tensorflow as tf 
import numpy as np 
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout,MaxPooling2D, Activation
from tensorflow.keras.callbacks import TensorBoard
import time




X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))
#We need to always normalize the data before we feed into model. We need to scale our model first.
#For gray scale images 0 represents black wile 255 is white.....So we need to normalize all the pixel values, convert every pixel in the range 0 to 1
X=X/255.0



dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME ="{}-conv-{}-nodes-{}-dense".format(conv_layer,layer_size,dense_layer,int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
            print(NAME)
            model = Sequential()
            model.add(Conv2D(layer_size,(3,3),input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Flatten())    
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
            model.fit(np.array(X),np.array(y), batch_size=3,epochs = 100, validation_split=0,callbacks=[tensorboard])

            model.save('corona.model')
