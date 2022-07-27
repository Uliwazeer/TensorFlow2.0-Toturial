from statistics import mode
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os
from pickletools import optimize
from tabnanny import verbose
from turtle import shape
from tensorflow.keras.datasets import mnist
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = keras.Sequential()
model.add(keras.Input(shape=(None,28)))
model.add(
    layers.simpleRNN(512,return_sequences=True,activation='relu')
    # layers.GRU(256,return_sequences=True,activation='tanh')
    # layers.LSTM(256,return_sequences=True,activation='tanh')
    
)
model.add(layers.simpleRNN(512,activation='relu'))
# model.add(layers.GRU(256,activation='tanh'))
# model.add(layers.LSTM(256,activation='tanh'))
model.add(layers.Dense(10))
# print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=64,verbose=2)
print(model.summary())
