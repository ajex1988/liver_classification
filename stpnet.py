import tensorflow as tf
from inception_3d import inception_module
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.framework.python.ops import arg_scope

def stpnet_base(inputs,
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

def stpnet_base_v0(inputs,
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

                # end_point = 'MaxPool3D_4a'
                # net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=2,padding='same',scope=end_point)
                # end_points[end_point] = net
                # if final_endpoint == end_point:
                #     return net, end_points

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
def stpnet_base_v1(inputs,
                        final_endpoint='Inception_5b',
                        scope='Inception3D'):
    # pool the depth channel at last
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
                net = tf.contrib.layers.conv3d(inputs=inputs,num_outputs=64,kernel_size=7,stride=(1,3,3),scope='Conv3D_1a')
                end_points[end_point] = net
                if final_endpoint == end_point:
                    return net, end_points
                end_point = 'MaxPool3D_2a'
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=(1,2,2),padding='same',scope='MaxPool3D_2a')
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
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=(1,2,2),padding='same',scope='MaxPool3D_3a')
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
                net = tf.contrib.layers.max_pool3d(inputs=net,kernel_size=3,stride=(1,2,2),padding='same',scope=end_point)
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
                net = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=3, stride=(1,2,2),padding='same', scope=end_point)
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

def stpnet_v1(inputs,
                    num_outputs=30,
                    dropout_keep_prob=0.8,
                    pooling_kernel_size=[[1,7,7],[2,14,14],[4,28,28]],
                    pooling_stride=[[1,7,7],[2,14,14],[4,28,28]],
                    scope='Inception3D'):
    # Use Max Pooling
    with tf.variable_scope(scope, 'InceptionV1', [inputs]) as scope:
        net, end_points = stpnet_base_v0(inputs=inputs, final_endpoint='Inception_4a')
        with tf.variable_scope('Logits'):
            pyramid_0 = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=pooling_kernel_size[0], stride=pooling_stride[0], scope='AvgPool3D_Pyramid_0')
            pyramid_0_flattened = tf.contrib.layers.flatten(pyramid_0)
            pyramid_1 = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=pooling_kernel_size[1], stride=pooling_stride[1], scope='AvgPool3D_Pyramid_1')
            pyramid_1_flattened = tf.contrib.layers.flatten(pyramid_1)
            pyramid_2 = tf.contrib.layers.max_pool3d(inputs=net, kernel_size=pooling_kernel_size[2], stride=pooling_stride[2], scope='AvgPool3D_Pyramid_2')
            pyramid_2_flattened = tf.contrib.layers.flatten(pyramid_2)
            net = tf.concat([pyramid_0_flattened,pyramid_1_flattened,pyramid_2_flattened],axis=1)
            end_points['AvgPool3D'] = net
            net = tf.layers.dropout(inputs=net, rate=dropout_keep_prob)
            logits = tf.contrib.layers.fully_connected(inputs=net, num_outputs=num_outputs, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(0.0),
                                                       weights_regularizer=regularizers.l2_regularizer(0.0002),
                                                       biases_regularizer=regularizers.l2_regularizer(0.0002),
                                                       scope='InnerProduct')
            end_points['Logits'] = logits
            return logits, end_points

def stpnet_v2(inputs,
                    num_outputs=30,
                    dropout_keep_prob=0.8,
                    pooling_kernel_size=[[1,7,7],[2,14,14],[4,28,28]],
                    pooling_stride=[[1,7,7],[2,14,14],[4,28,28]],
                    scope='Inception3D'):
    # Use Avg Pooling
    with tf.variable_scope(scope, 'InceptionV1', [inputs]) as scope:
        net, end_points = stpnet_base_v0(inputs=inputs, final_endpoint='Inception_4a')
        with tf.variable_scope('Logits'):
            pyramid_0 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[0], stride=pooling_stride[0], scope='AvgPool3D_Pyramid_0')
            pyramid_0_flattened = tf.contrib.layers.flatten(pyramid_0)
            pyramid_1 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[1], stride=pooling_stride[1], scope='AvgPool3D_Pyramid_1')
            pyramid_1_flattened = tf.contrib.layers.flatten(pyramid_1)
            pyramid_2 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[2], stride=pooling_stride[2], scope='AvgPool3D_Pyramid_2')
            pyramid_2_flattened = tf.contrib.layers.flatten(pyramid_2)
            net = tf.concat([pyramid_0_flattened,pyramid_1_flattened,pyramid_2_flattened],axis=1)
            end_points['AvgPool3D'] = net
            net = tf.layers.dropout(inputs=net, rate=dropout_keep_prob)
            logits = tf.contrib.layers.fully_connected(inputs=net, num_outputs=num_outputs, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(0.0),
                                                       weights_regularizer=regularizers.l2_regularizer(0.0002),
                                                       biases_regularizer=regularizers.l2_regularizer(0.0002),
                                                       scope='InnerProduct')
            end_points['Logits'] = logits
            return logits, end_points

def stpnet_v3(inputs,
                    num_outputs=30,
                    dropout_keep_prob=0.8,
                    pooling_kernel_size=[[1,7,7],[2,14,14],[4,28,28]],
                    pooling_stride=[[1,7,7],[2,14,14],[4,28,28]],
                    scope='Inception3D'):
    # Use Avg Pooling
    with tf.variable_scope(scope, 'InceptionV1', [inputs]) as scope:
        net, end_points = stpnet_base_v1(inputs=inputs, final_endpoint='Inception_4a')
        with tf.variable_scope('Logits'):
            pyramid_0 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[0], stride=pooling_stride[0], scope='AvgPool3D_Pyramid_0')
            pyramid_0_flattened = tf.contrib.layers.flatten(pyramid_0)
            pyramid_1 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[1], stride=pooling_stride[1], scope='AvgPool3D_Pyramid_1')
            pyramid_1_flattened = tf.contrib.layers.flatten(pyramid_1)
            pyramid_2 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[2], stride=pooling_stride[2], scope='AvgPool3D_Pyramid_2')
            pyramid_2_flattened = tf.contrib.layers.flatten(pyramid_2)
            net = tf.concat([pyramid_0_flattened,pyramid_1_flattened,pyramid_2_flattened],axis=1)
            end_points['AvgPool3D'] = net
            net = tf.layers.dropout(inputs=net, rate=dropout_keep_prob)
            logits = tf.contrib.layers.fully_connected(inputs=net, num_outputs=num_outputs, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(0.0),
                                                       weights_regularizer=regularizers.l2_regularizer(0.0002),
                                                       biases_regularizer=regularizers.l2_regularizer(0.0002),
                                                       scope='InnerProduct')
            end_points['Logits'] = logits
            return logits, end_points
def stpnet_v4(inputs,
              num_outputs=30,
              dropout_keep_prob=0.8,
              pooling_kernel_size=[[1,2,2],[1,4,4],[1,8,8]],
              pooling_stride=[[1,2,2],[1,4,4],[1,8,8]],
              scope='Inception3D'):
    # Inception v1 based
    with tf.variable_scope(scope, 'InceptionV1', [inputs]) as scope:
        net, end_points = stpnet_base(inputs=inputs, final_endpoint='Inception_5b')
        with tf.variable_scope('Logits'):
            pyramid_0 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[0], stride=pooling_stride[0], scope='AvgPool3D_Pyramid_0')
            pyramid_0_flattened = tf.contrib.layers.flatten(pyramid_0)
            pyramid_1 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[1], stride=pooling_stride[1], scope='AvgPool3D_Pyramid_1')
            pyramid_1_flattened = tf.contrib.layers.flatten(pyramid_1)
            pyramid_2 = tf.contrib.layers.avg_pool3d(inputs=net, kernel_size=pooling_kernel_size[2], stride=pooling_stride[2], scope='AvgPool3D_Pyramid_2')
            pyramid_2_flattened = tf.contrib.layers.flatten(pyramid_2)
            net = tf.concat([pyramid_0_flattened,pyramid_1_flattened,pyramid_2_flattened],axis=1)
            end_points['AvgPool3D'] = net
            net = tf.layers.dropout(inputs=net, rate=dropout_keep_prob)
            logits = tf.contrib.layers.fully_connected(inputs=net, num_outputs=num_outputs, activation_fn=None,
                                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                       biases_initializer=tf.constant_initializer(0.0),
                                                       weights_regularizer=regularizers.l2_regularizer(0.0002),
                                                       biases_regularizer=regularizers.l2_regularizer(0.0002),
                                                       scope='InnerProduct')
            end_points['Logits'] = logits
            return logits, end_points
def stp_layer(inputs,
              num_pyramids,
              scope):
    blob_shape = tf.shape(inputs)
    depth = blob_shape[1]
    height = blob_shape[2]
    width = blob_shape[3]

    num_splits = []
    for i in range(num_pyramids):
        num_splits.append(2.0**i)
    kernel_sizes = []
    for i in range(num_pyramids):
        kernel_depth = tf.floor(tf.div(tf.cast(depth,tf.float32),num_splits[i]))
        kernel_depth = tf.cast(kernel_depth,tf.int32)
        kernel_height = tf.floor(tf.div(tf.cast(height,tf.float32),num_splits[i]))
        kernel_height = tf.cast(kernel_height,tf.int32)
        kernel_width = tf.floor(tf.div(tf.cast(width,tf.float32),num_splits[i]))
        kernel_width = tf.cast(kernel_width,tf.int32)
        kernel_size = [kernel_depth,kernel_height,kernel_width]
        kernel_sizes.append(kernel_size)
    end_point = scope
    pyramids = []
    with tf.variable_scope(scope):
        for p_idx in range(num_pyramids):
            pyramid_name = 'Pyramid_{0}'.format(p_idx)
            kernel_size = kernel_sizes[p_idx]
            with tf.variable_scope(pyramid_name):
                pyramid = tf.contrib.layers.max_pool3d(inputs=inputs,
                                                       kernel_size=kernel_size,
                                                       stride=kernel_size,
                                                       padding='valid',
                                                       scope='MaxPool3D')


