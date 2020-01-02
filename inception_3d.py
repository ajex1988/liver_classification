import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.framework.python.ops import arg_scope

def inception_module(input,scope,
                     num_b0,ks_b0,
                     num_b1_0,ks_b1_0,num_b1_1,ks_b1_1,
                     num_b2_0,ks_b2_0,num_b2_1,ks_b2_1,
                     ks_b3_0,num_b3_1,ks_b3_1):
    end_point = scope
    net = input
    with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
            branch_0 = tf.contrib.layers.conv3d(inputs=net, num_outputs=num_b0, kernel_size=ks_b0, scope='Conv3D_0a')
        with tf.variable_scope('Branch_1'):
            branch_1 = tf.contrib.layers.conv3d(inputs=net, num_outputs=num_b1_0, kernel_size=ks_b1_0, scope='Conv3D_0a')
            branch_1 = tf.contrib.layers.conv3d(inputs=branch_1, num_outputs=num_b1_1, kernel_size=ks_b1_1, scope='Conv3D_0b')
        with tf.variable_scope('Branch_2'):
            branch_2 = tf.contrib.layers.conv3d(inputs=net, num_outputs=num_b2_0, kernel_size=ks_b2_0, scope='Conv3D_0a')
            branch_2 = tf.contrib.layers.conv3d(inputs=branch_2, num_outputs=num_b2_1, kernel_size=ks_b2_1, scope='Conv3D_0b')
        with tf.variable_scope('Branch_3'):
            branch_3 = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=ks_b3_0, stride=1, padding='same',scope='MaxPool3D_0a')
            branch_3 = tf.contrib.layers.conv3d(inputs=branch_3, num_outputs=num_b3_1, kernel_size=ks_b3_1, scope='Conv3D_0b')
        net = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4)
    return net

def inception_3d_base(inputs,
                        final_endpoint='Inception_5b',
                        scope='Inception3D'):
    end_points = {}
    with tf.variable_scope(name_or_scope=scope,default_name="Inception3D",values=[inputs]):
        with arg_scope([tf.contrib.layers.conv3d],
                        weights_initializer=initializers.xavier_initializer(),
                        weights_regularizer=regularizers.l2_regularizer(0.0002),
                        biases_initializer=tf.constant_initializer(0.2),
                        biases_regularizer=regularizers.l2_regularizer(0.0002)
                        ):
            with arg_scope([tf.contrib.layers.conv3d],
                           stride=1,
                           padding='SAME'):
                end_point = 'Conv3D_1a'
                net = tf.contrib.layers.conv3d(inputs=inputs,num_outputs=64,kernel_size=7,stride=2,scope='Conv3D_1a')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
                end_point = 'MaxPool3D_2a'
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=2,padding='same',scope='MaxPool3D_2a')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
                end_point = 'Conv3D_2b'
                net = tf.contrib.layers.conv3d(inputs=net,num_outputs=64,kernel_size=1,scope='Conv3D_2b')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
                end_point = 'Conv3D_2c'
                net = tf.contrib.layers.conv3d(inputs=net,num_outputs=192,kernel_size=3,scope='Conv3D_2c')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
                end_point = 'MaxPool3D_3a'
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=2,padding='same',scope='MaxPool3D_3a')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_3a'
                net = inception_module(net,end_point,
                                       64,1,
                                       96,1,128,3,
                                       16,1,32,5,
                                       3,32,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_3b'
                net = inception_module(net,end_point,
                                       128,1,
                                       128,1,192,3,
                                       32,1,96,5,
                                       3,64,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'MaxPool3D_4a'
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=2,padding='same',scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_4a'
                net = inception_module(net,end_point,
                                       192,1,
                                       96,1,208,3,
                                       16,1,48,3,
                                       3,64,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_4b'
                net = inception_module(net,end_point,
                                       160,1,
                                       112,1,224,3,
                                       24,1,64,3,
                                       3,64,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_4c'
                net = inception_module(net,end_point,
                                       128,1,
                                       128,1,256,3,
                                       24,1,64,3,
                                       3,64,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_4d'
                net = inception_module(net,end_point,
                                       112,1,
                                       144,1,288,3,
                                       32,1,64,3,
                                       3,64,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_4e'
                net = inception_module(net,end_point,
                                       256,1,
                                       160,1,320,3,
                                       32,1,128,3,
                                       3,128,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net,end_points

                end_point = 'MaxPool3D_5a'
                net = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=3, stride=2,padding='same', scope=end_point)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_5a'
                net = inception_module(net,end_point,
                                       256,1,
                                       160,1,320,3,
                                       32,1,128,3,
                                       3,128,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

                end_point = 'Inception_5b'
                net = inception_module(net,end_point,
                                       384,1,
                                       192,1,384,3,
                                       48,1,128,3,
                                       3,128,1)
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points

            raise ValueError('Unknown final endpoint {0}'.format(final_endpoint))

def inception_3d_v1(inputs,
                    num_outputs=30,
                    dropout_keep_prob=0.8,
                    scope='Inception3D'):
    with tf.variable_scope(scope, 'InceptionV1', [inputs]) as scope:
        net, end_points = inception_3d_base(inputs=inputs,final_endpoint='Inception_5b')
        with tf.variable_scope('Logits'):
            net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=[1,7,7],stride=1,scope='AvgPool3D')
            end_points['AvgPool3D'] = net
            net = tf.layers.dropout(inputs=net,rate=dropout_keep_prob)
            logits = tf.contrib.layers.fully_connected(inputs=net,num_outputs=num_outputs,activation_fn=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.constant_initializer(0.0),
                                        weights_regularizer=regularizers.l2_regularizer(0.0002),
                                        biases_regularizer=regularizers.l2_regularizer(0.0002),
                                        scope='InnerProduct')
            logits = tf.squeeze(net,[1,2,3],name='SpatialTemporalSqueeze')
            end_points['Logits'] = logits
            return logits, end_points

