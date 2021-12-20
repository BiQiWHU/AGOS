import tensorflow as tf


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, relu=True, plus_bias=True, bn=False, is_training=False):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    def convolve(i, k):
        return tf.nn.conv2d(i, k,
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        if groups == 1:
            conv_img = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        # split input and weights and convolve them separately
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv_img = tf.concat(axis=3, values=output_groups)

        # bias = tf.reshape(tf.nn.bias_add(conv_img, biases), conv_img.get_shape().as_list())
        out = conv_img
        if plus_bias:
            out = tf.nn.bias_add(conv_img, biases)
        if bn:
            out = tf.layers.batch_normalization(out, training=is_training, scale=False, momentum=0.999)
        if relu:
            out = tf.nn.relu(out, name=scope.name)
        return out


def dilated_conv(x, filter_height, filter_width, num_filters, rate, name,
         padding='SAME', relu=True, bn=False, is_training=False):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    def convolve(i, k, rate):
        return tf.nn.atrous_conv2d(i, k, rate, padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels, num_filters],
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))

        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        conv_img = convolve(x, weights, rate=rate)
        out = tf.nn.bias_add(conv_img, biases)

        if bn:
            out = tf.layers.batch_normalization(out, training=is_training, momentum=0.999)
        if relu:
            out = tf.nn.relu(out, name=scope.name)

        return out


def dilated_conv_group(x, filter_height, filter_width, num_filters, rate, name,
         padding='SAME', relu=True, bn=False, is_training=False):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    def convolve(i, k, rate):
        return tf.nn.atrous_conv2d(i, k, rate, padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width, input_channels, num_filters],
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))

        biases = tf.get_variable('biases',
                                 shape=[num_filters],
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        conv_img_1 = convolve(x, weights, rate=1)
        conv_img_3 = convolve(x, weights, rate=3)
        conv_img_5 = convolve(x, weights, rate=5)
        conv_img_7 = convolve(x, weights, rate=7)

        out_1 = tf.nn.bias_add(conv_img_1, biases)
        out_3 = tf.nn.bias_add(conv_img_3, biases)
        out_5 = tf.nn.bias_add(conv_img_5, biases)
        out_7 = tf.nn.bias_add(conv_img_7, biases)

        if relu:
            out_1 = tf.nn.relu(out_1, name=scope.name)
            out_3 = tf.nn.relu(out_3, name=scope.name)
            out_5 = tf.nn.relu(out_5, name=scope.name)
            out_7 = tf.nn.relu(out_7, name=scope.name)
        if bn:
            out_1 = tf.layers.batch_normalization(out_1, training=is_training, scale=False, center=False,
                                                  momentum=0.999)
            out_3 = tf.layers.batch_normalization(out_3, training=is_training, scale=False, center=False,
                                                  momentum=0.999)
            out_5 = tf.layers.batch_normalization(out_5, training=is_training, scale=False, center=False,
                                                  momentum=0.999)
            out_7 = tf.layers.batch_normalization(out_7, training=is_training, scale=False, center=False,
                                                  momentum=0.999)

        return out_1, out_3, out_5, out_7


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [num_out], trainable=True, initializer=tf.constant_initializer(0.0))
        tf.summary.histogram(name+"/weights", weights)
        tf.summary.histogram(name+"/biases", biases)

        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu is True:
            act = tf.nn.relu(act)

        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def avg_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
