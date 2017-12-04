import tensorflow as tf
from Layers import convolution_layer_3d, deconvolution_layer_3d, prelu


def convolution_block(layer_input, n_channels, num_convolutions):
    x = layer_input
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = prelu(convolution_layer_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1]))
    return x + layer_input


def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    with tf.variable_scope('conv_' + str(1)):
        x = prelu(convolution_layer_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1]))

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = prelu(convolution_layer_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1]))
    return x + layer_input


def down_convolution(layer_input, in_channels):
    with tf.variable_scope('down_convolution'):
        return prelu(convolution_layer_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1]))


def up_convolution(layer_input, output_shape, in_channels):
    with tf.variable_scope('up_convolution'):
        return prelu(deconvolution_layer_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1]))


def v_net(tf_input, input_channels, n_channels=16, output_channels=1):

    with tf.variable_scope('contracting_path'):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, 1, n_channels])
        else:
            with tf.variable_scope('level_0'):
                c0 = prelu(convolution_layer_3d(tf_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1]))

        with tf.variable_scope('level_1'):
            c1 = convolution_block(c0, n_channels, 1)
            c12 = down_convolution(c1, n_channels)

        with tf.variable_scope('level_2'):
            c2 = convolution_block(c12, n_channels * 2, 2)
            c22 = down_convolution(c2, n_channels * 2)

        with tf.variable_scope('level_3'):
            c3 = convolution_block(c22, n_channels * 4, 3)
            c32 = down_convolution(c3, n_channels * 4)

        with tf.variable_scope('level_4'):
            c4 = convolution_block(c32, n_channels * 8, 3)
            c42 = down_convolution(c4, n_channels * 8)

        with tf.variable_scope('level_5'):
            c5 = convolution_block(c42, n_channels * 16, 3)
            c52 = up_convolution(c5, tf.shape(c4), n_channels * 16)

    with tf.variable_scope('expanding_path'):

        with tf.variable_scope('level_4'):
            d4 = convolution_block_2(c52, c4, n_channels * 8, 3)
            d42 = up_convolution(d4, tf.shape(c3), n_channels * 8)

        with tf.variable_scope('level_3'):
            d3 = convolution_block_2(d42, c3, n_channels * 4, 3)
            d32 = up_convolution(d3, tf.shape(c2), n_channels * 4)

        with tf.variable_scope('level_2'):
            d2 = convolution_block_2(d32, c2, n_channels * 2, 2)
            d22 = up_convolution(d2, tf.shape(c1), n_channels * 2)

        with tf.variable_scope('level_1'):
            d1 = convolution_block_2(d22, c1, n_channels, 1)
            with tf.variable_scope('output_layer'):
                logits = convolution_layer_3d(d1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

    return logits
