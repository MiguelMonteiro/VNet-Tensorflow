import tensorflow as tf


# These initializers are probably incorrect but they work so far


def xavier_normal_dist(shape):
    return tf.truncated_normal(shape, mean=0, stddev=tf.sqrt(3. / shape[-1] + shape[-2]))


def xavier_uniform_dist(shape):
    lim = tf.sqrt(6. / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-lim, maxval=lim)


def xavier_normal_dist_conv3d(shape):
    return tf.truncated_normal(shape, mean=0,
                               stddev=tf.sqrt(3. / (tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:]))))


def xavier_uniform_dist_conv3d(shape):
    with tf.variable_scope('xavier_glorot_initializer'):
        denominator = tf.cast((tf.reduce_prod(shape[:3]) * tf.reduce_sum(shape[3:])), tf.float32)
        lim = tf.sqrt(6. / denominator)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)


def convolution_layer_3d(layer_input, filter, strides, padding='SAME'):
    assert len(filter) == 5  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 5  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')

    w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')

    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    convolution = tf.nn.conv3d(layer_input, w, strides, padding)
    return convolution + b


def deconvolution_layer_3d(layer_input, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 5  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 5  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    # w = tf.Variable(initial_value=tf.truncated_normal(shape=filter), name='weights')
    w = tf.Variable(initial_value=xavier_uniform_dist_conv3d(shape=filter), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
    deconvolution = tf.nn.conv3d_transpose(layer_input, w, output_shape, strides, padding)
    return deconvolution + b


def max_pooling_3d(layer_input, ksize, strides, padding='VALID'):
    assert len(ksize) == 5  # [batch, depth, rows, cols, channels]
    assert len(strides) == 5  # [batch, depth, rows, cols, channels]
    assert ksize[0] == ksize[4]
    assert ksize[0] == 1
    assert strides[0] == strides[4]
    assert strides[0] == 1
    return tf.nn.max_pool3d(layer_input, ksize, strides, padding)


# parametric leaky relu
def prelu(x):
    with tf.variable_scope('prelu'):
        _alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype,
                                 initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)

