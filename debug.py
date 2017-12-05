import tensorflow as tf
from VNetOriginal import v_net

#a = tf.ones(shape=(10, 50, 50, 20, 1))
a = tf.ones(shape=(10, 50, 50, 20, 6))
b = v_net(a, 6)