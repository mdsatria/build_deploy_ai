import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
# set parameter
n_class = 10
epoch = 10
n_batch = 32
# ambil data dari lib keras
print('Importing data..')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape sesuai network
print('Reshaping data..')
reshaped_x_train = x_train.reshape(-1,784)
reshaped_x_test = x_test.reshape(-1,784)
# encode label kelas
print('Reshaping label..')
reshaped_y_train = keras.utils.to_categorical(y_train, n_class)
reshaped_y_test = keras.utils.to_categorical(y_test, n_class)
# define architecture
print('Building Model...')
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(n_class, activation='softmax'))
print('Compiling Model...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
model.fit(reshaped_x_train, reshaped_y_train, validation_data=(reshaped_x_test, reshaped_y_test), epochs=epoch, batch_size=n_batch)
# save model
print('Saving model...')
model.save('simple.h5')
print("Saved model to disk")