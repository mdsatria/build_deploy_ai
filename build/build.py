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

print('Reshaping data..')

# reshape sesuai network
reshaped_x_train = x_train.reshape(-1,784).astype(np.float32)
reshaped_x_test = x_test.reshape(-1,784).astype(np.float32)

# normalisasi nilai piksel
reshaped_x_train /= 255
reshaped_x_test /= 255




# encode label kelas
print('Reshaping label..')

from keras.utils import to_categorical

n_class = 10
reshaped_y_train = to_categorical(y_train, n_class)
reshaped_y_test = to_categorical(y_test, n_class)



# define architecture
print('Building Model...')
model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(n_class, activation='softmax'))
#print('Compiling Model...')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
model.fit(reshaped_x_train, reshaped_y_train, validation_data=(reshaped_x_test, reshaped_y_test), epochs=epoch, batch_size=n_batch)
# save model
print('Saving model...')
model.save('model.h5')
print("Saved model to disk")