import os
from pickletools import optimize 
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist 
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
(x_train, y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
x_train = x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1,28*28).astype("float32") / 255.0
# Sequential API (Very Convenient , Not Very Flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(256,activation='relu',name='my_layer'))
model.add(layers.Dense(10))
model = keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers])
feature = model.predict(x_train)
for feature in features:
    print(feature.shape)
    
print(feature.shape)
print(model.summary())
import sys
sys.exit()

# Functional API (A bit More Flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu',name='first_layer')(inputs)
x = layers.Dense(256,activation='relu',name='second_layer')(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
print(model.summary())
     
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)