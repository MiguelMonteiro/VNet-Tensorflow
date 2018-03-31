import tensorflow as tf
from Layers import convolution_3d, deconvolution_3d, prelu


def convolution_block(layer_input, n_channels, num_convolutions, dropout_keep_prob):
    x = layer_input

    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)

    with tf.variable_scope('conv_' + str(num_convolutions)):
        x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
        x = x + layer_input
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    return x


def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions, dropout_keep_prob):

    x = tf.concat((layer_input, fine_grained_features), axis=-1)

    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])
            x = x + layer_input
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)
        return x

    with tf.variable_scope('conv_' + str(1)):
        x = convolution_3d(x, [5, 5, 5, n_channels * 2, n_channels], [1, 1, 1, 1, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
            x = prelu(x)
            x = tf.nn.dropout(x, dropout_keep_prob)

    with tf.variable_scope('conv_' + str(num_convolutions)):
        x = convolution_3d(x, [5, 5, 5, n_channels, n_channels], [1, 1, 1, 1, 1])
        x = x + layer_input
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)

    return x


def down_convolution(layer_input, in_channels, dropout_keep_prob):
    with tf.variable_scope('down_convolution'):
        x = convolution_3d(layer_input, [2, 2, 2, in_channels, in_channels * 2], [1, 2, 2, 2, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)
        return x


def up_convolution(layer_input, output_shape, in_channels, dropout_keep_prob):
    with tf.variable_scope('up_convolution'):
        x = deconvolution_3d(layer_input, [2, 2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 2, 1])
        x = prelu(x)
        x = tf.nn.dropout(x, dropout_keep_prob)
        return x


def v_net(tf_input, dropout_keep_prob, input_channels, output_channels=1, n_channels=16):

    with tf.variable_scope('contracting_path'):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, 1, n_channels])
        else:
            with tf.variable_scope('level_0'):
                c0 = convolution_3d(tf_input, [5, 5, 5, input_channels, n_channels], [1, 1, 1, 1, 1])
                c0 = tf.nn.dropout(c0, dropout_keep_prob)
                c0 = prelu(c0)

        with tf.variable_scope('level_1'):
            c1 = convolution_block(c0, n_channels, 1, dropout_keep_prob)
            c12 = down_convolution(c1, n_channels, dropout_keep_prob)

        with tf.variable_scope('level_2'):
            c2 = convolution_block(c12, n_channels * 2, 2, dropout_keep_prob)
            c22 = down_convolution(c2, n_channels * 2, dropout_keep_prob)

        with tf.variable_scope('level_3'):
            c3 = convolution_block(c22, n_channels * 4, 3, dropout_keep_prob)
            c32 = down_convolution(c3, n_channels * 4, dropout_keep_prob)

        with tf.variable_scope('level_4'):
            c4 = convolution_block(c32, n_channels * 8, 3, dropout_keep_prob)
            c42 = down_convolution(c4, n_channels * 8, dropout_keep_prob)

        with tf.variable_scope('level_5'):
            c5 = convolution_block(c42, n_channels * 16, 3, dropout_keep_prob)
            c52 = up_convolution(c5, tf.shape(c4), n_channels * 16, dropout_keep_prob)

    with tf.variable_scope('expanding_path'):

        with tf.variable_scope('level_4'):
            e4 = convolution_block_2(c52, c4, n_channels * 8, 3, dropout_keep_prob)
            e42 = up_convolution(e4, tf.shape(c3), n_channels * 8, dropout_keep_prob)

        with tf.variable_scope('level_3'):
            e3 = convolution_block_2(e42, c3, n_channels * 4, 3, dropout_keep_prob)
            e32 = up_convolution(e3, tf.shape(c2), n_channels * 4, dropout_keep_prob)

        with tf.variable_scope('level_2'):
            e2 = convolution_block_2(e32, c2, n_channels * 2, 2, dropout_keep_prob)
            e22 = up_convolution(e2, tf.shape(c1), n_channels * 2, dropout_keep_prob)

        with tf.variable_scope('level_1'):
            e1 = convolution_block_2(e22, c1, n_channels, 1, dropout_keep_prob)
            with tf.variable_scope('output_layer'):
                logits = convolution_3d(e1, [1, 1, 1, n_channels, output_channels], [1, 1, 1, 1, 1])

    return logits


