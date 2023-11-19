import tensorflow as tf
import numpy as np

print('--Constant--')
tensor_20 = tf.constant([[23, 4], [32, 51]])
print(tensor_20)
print(tensor_20.shape)
print(tensor_20.numpy())

numpy_tensor = np.array([[23,  4], [32, 51]])
tensor_from_numpy = tf.constant(numpy_tensor)
print(tensor_from_numpy)

print('--Variables--')

tf2_variable = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(tf2_variable)
print(tf2_variable.numpy())