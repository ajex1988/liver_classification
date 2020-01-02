import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

import inception_2d

def inception_2d_fields(img,
                        fields,
                        num_classes=30,
                        is_training=True,
                        dropout_keep_prob=0.6,
                        prediction_fn=layers_lib.softmax,
                        spatial_squeeze=True,
                        reuse=None,
                        scope='InceptionV1_Fields'
                        ):
    with arg_scope([layers.conv2d, layers_lib.fully_connected],
                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                   biases_initializer=tf.constant_initializer(0.2),
                   weights_regularizer=regularizers.l2_regularizer(0.0002),
                   biases_regularizer=regularizers.l2_regularizer(0.0002)):
        net, end_points = inception_2d.inception_v1_base(img, scope=scope, final_endpoint='Mixed_4b')
        with variable_scope.variable_scope('Logits'):
            net = layers_lib.avg_pool2d(net, [5, 5], stride=3, scope='AvgPool_0a_5x5')
            net = layers.conv2d(inputs=net, num_outputs=128, kernel_size=1)
            net = tf.reshape(net, [-1, 1, 1, 4 * 4 * 128])
            net = array_ops.squeeze(net,[1,2],name='Squeeze4Fields')
            net = tf.concat([net,fields],axis=1)
            net = layers.fully_connected(inputs=net, num_outputs=1024)
            net = layers_lib.dropout(net, dropout_keep_prob, scope='Dropout_0b')
            logits = layers.fully_connected(inputs=net,
                                            num_outputs=num_classes,
                                            activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.0),
                                            weights_regularizer=regularizers.l2_regularizer(0.0002),
                                            biases_regularizer=regularizers.l2_regularizer(0.0002),
                                            scope='InnerProduct')
            # logits = layers.conv2d(
            #     net,
            #     num_classes, [1, 1],
            #     activation_fn=None,
            #     normalizer_fn=None,
            #     scope='Conv2d_0c_1x1')
            if spatial_squeeze:
                logits = array_ops.squeeze(logits, [1, 2], name='SpatialSqueeze')

            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')


    return logits, end_points
