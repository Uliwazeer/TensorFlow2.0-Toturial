from ast import Global
from lib2to3.pgen2 import tokenize
from tabnanny import verbose
from sklearn import metrics

from sklearn.utils import shuffle
import tensorflow_hub as hub
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# ================================================ #
#                  Pretrained-Model                #
# ================================================ #

# (ds_train ,ds_test) , ds_info = tfds.load("mnist",split["train","test"],shuffle_files=True,as_supervised=True,with_info=True,)
# # fig = tfds.show_examples(ds_train,ds_info,rows=4,cols=4)
# # print(ds_info)

# def normalize_img(image,label):
#     # normliaze images 
#     return tf.cast(image,tf.float32) / 255.0 , label
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# BATCH_SIZE = 64
# ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
# ds_train = ds_train.cache())
# ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
# ds_train = ds_train.batch(BATCH_SIZE)
# ds_train = ds_train.perfetch(AUTOTUNE)

# ds_test=ds_test.map(
#         normalize_img, num_parallel_calls=AUTOTUNE)
# ds_test=ds_test.batch(BATCH_SIZE)
# ds_test=ds_test.perfetch(AUTOTUNE)


# model = keras.Sequential([
#     keras.Input((28,28,1)),
#     layers.Conv2D(32,3,activation='relu'),
#     layers.Flatten(),
#     layers.Dense(10),
# ])
# model.compile(
#     optimizer = keras.optimizers.Adam(lr=0.001),
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     matrics = ["accuracy"],
# )

# model.fit(ds_train,epochs=5,verbose=2,)
# model.evaluate(ds_test)

(ds_train,ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train","test"],
    shuffle_files = True,
    as_supervised = True,
    with_info = True,
)


# print(ds_info)
# for text, label in ds_train:
#     print(text)
#     import sys
#     sys.exit()
tokenizer = tfds.features.text.Tokenizer()
def build_vocabulary():
    vocabulary = set()
    for text , _ in ds_train:
        vocabulary.update(tokenizer.tokenize(text.numpy().lower()))
    return vocabulary 
vocabulary = build_vocabulary()
encoder = tfds.features.text.TokenTextEncoder(
    vocabulary , oov_token="<UNK>", lowercase=True,tokenizer = tokenizer
)

def my_encoding(text_tensor,label):
    return encoder.encode(text_tensor.numpy()),label
def encode_map(text,label):
    encoded_text , label = tf.py_function(
        my_encoding , inp=[text,label],Tout=(tf.int64,tf.int64)
    )
    encoded_text.set_shape([None])
    label.set_shape([])


return encoded_text , label

AUTOTUNE = tf.data.experimental.AUTOTUNE 
ds_train = ds_train.map(encode_map,num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(10000)
ds_train = ds_train.padded_batch(32,padded_batch=([None],())
ds_train = ds_train.perfetch(AUTOTUNE)

ds_test = ds_test.map(encode_map)
ds_test=ds_test.padded_batch(32, padded_batch=([None], ())
                             
model = keras.Sequential([
    layers.Masking(mask_value=8),
    layers.Embedding(input_dim=len(vocabulary)+2,output_dim = 32),
    # Batch_Size * 1000
    layers.GlobalAveragePooling1D(),
    # Batch_size * 32
    layers.Dense(64,activation='relu'),
    layers.Dense(1),#less than 0 negatives , greater or equal than 0 positive
    
])


model.compile(
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(3e-4,clipnorm=1),
    metrics["accuracy"],
)

model.fit(ds_train,epochs=10,verbose=2)
model.evaluate(ds_test)