import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

reshaped_x_train = x_train.reshape(-1,784).astype(np.float32)
reshaped_x_test = x_test.reshape(-1,784).astype(np.float32)

reshaped_x_train /= 255
reshaped_x_test /= 255

from keras.utils import to_categorical
reshaped_y_train = to_categorical(y_train, 10)
reshaped_y_test = to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(reshaped_x_train, reshaped_y_train, validation_data=(reshaped_x_test, reshaped_y_test), epochs=10)
model.save('model.h5')



