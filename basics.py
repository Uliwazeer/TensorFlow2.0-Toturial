# tensorflow is a dimmensional array (rows , column)
# tensor is an n-dimensional array of data

from turtle import shape

from numpy import dtype, indices
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

# initialization of tensors
X = tf.constant(50.0)
print(X)
X = tf.constant(5,shape(2,2),dtype=tf.float64)
print(X)
X = tf.constant([[1,2,3,4],[9,8,7,6]])
print(X)
Y = tf.ones((4,4))
print(Y)
Y = tf.zeros((3,3))
print(Y)
Y = tf.eye(5) # I for The Identiy Matrix eye(Union Matrix)
print(Y)
Z= tf.random.normal((3,3), mean=0,stddev=1)#Standard Normal Distribution
print(Z)
Z = tf.random.uniform((2,3),minval=0,maxval=1)
print(Z)
Z = tf.range(99)
print(Z)
Z = tf.range(start=1,limit=15,delta=5)
print(Z)
Z = tf.cast(Z,dtype=tf.float64)
print(Z)
# tf.float(16,32,64)
# tf.int(8,16,32,64)
# tf.bool(0,1) or true false
# mathematical operations
q = tf.constant([1,2,3])
w = tf.constant([7,8,5])
e = tf.add(q,w) #  e = q + w
print(e)
e = tf.subtract(q,w) # e = q - w
print(e)
e = tf.divide(q,w) # e = q / w
print(e)
e = tf.multiply(q,w) # e = q * w
print(e)
e = tf.tensordot(q, w, axes=1)  
print(e)
e = tf.reduce_sum(q*w,axis=0)  
print(e)
e = q**5
print(e)
q = tf.random.normal((2,3))
w = tf.random.normal((3,4))
e = tf.matmul(q,w)
print(e)
e = q @ w
print(e)
# indexing
r = tf.constant([0,1,2,3,4,5,6,7,8,9,10])
print(r[:])
print(r[2:])
print(r[3:9])
print(r[:5])
print(r[:-3])
print(r[::3])
print(r[::-2])
indices = tf.constant([1,6])
x_ind = tf.gather(x,indices)
print(x_ind)
t = tf.constant([[1,2],[3,4],[5,6]])
print(t[0,:])
print(t[1:3])
print(t[1:3,:])
# reshaping
u = tf.reshape(u , (2,2))
print(u)
u = tf.transpose(u,perm=[1,0])
print(u)