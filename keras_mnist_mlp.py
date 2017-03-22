'''Modifies the Keras mnist_mlp.py example to demonstrate
how to save/load the model description and model weights files.

Relevant links:
https://deeplearning4j.org/model-import-keras
https://github.com/fchollet/deep-learning-models
'''


from __future__ import print_function

import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.metrics import accuracy_score


batch_size = 128
num_classes = 10
epochs = 1

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_test_class = y_test

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# save and load model

# method 1: save model description and weights in the same HDF5 file
# model.save('trained_model.h5')
# del model
# from keras.models import load_model
# model = load_model('trained_model.h5')

# method 2: save model description (json) and weights (HDF5) separately
model_json = model.to_json()
print("Model description:", model_json, sep='\n')
model.save_weights('model_weights.h5')
del model
from keras.models import model_from_json
model = model_from_json(model_json)
model.load_weights('model_weights.h5')

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)

print('Accuracy:', accuracy_score(y_test_class, y_pred_class))
