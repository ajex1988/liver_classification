import os

import numpy as np
import tensorflow as tf
import inception_2d_fields


import  sys

def train_data_input_fn(tfrecord_file,perform_shuffle=True,repeat_count=-1,batch_size=32):
    # use 2d tfrecord
    def decode_liver_data(sequence_proto):
        feature_description = {
            'height':tf.FixedLenFeature([],tf.int64,0),
            'width':tf.FixedLenFeature([],tf.int64,0),
            'scanning_sequence': tf.FixedLenFeature([],tf.float32,0),
            'repetition_time':  tf.FixedLenFeature([],tf.float32,0),
            'echo_time':  tf.FixedLenFeature([],tf.float32,0),
            'echo_train_length':  tf.FixedLenFeature([],tf.float32,0),
            'flip_angle':  tf.FixedLenFeature([],tf.float32,0),
            'label': tf.FixedLenFeature([], tf.int64, 0),
            'img_raw':tf.FixedLenFeature([],tf.string,'')
        }
        sequence = tf.parse_single_example(sequence_proto,feature_description)
        height = sequence['height']
        width = sequence['width']
        label = sequence['label']

        scanning_sequence = sequence['scanning_sequence']
        repetition_time = sequence['repetition_time']
        echo_time = sequence['echo_time']
        echo_train_length = sequence['echo_train_length']
        flip_angle = sequence['flip_angle']
        fields = tf.stack([scanning_sequence,repetition_time,echo_time,echo_train_length,flip_angle],axis=0)

        img = tf.decode_raw(sequence['img_raw'],tf.int16)
        img = tf.cast(img,tf.float32)
        # Normalize to -1,1
        img = tf.div(tf.subtract(img,
                                 tf.reduce_min(img)+1e-8),
                     tf.subtract(tf.reduce_max(img),
                                 tf.reduce_min(img))+1e-8)*255.0
        img = tf.subtract(img,37)
        dim = tf.stack([height, width])
        img = tf.reshape(img,shape=dim)

        img = tf.expand_dims(img,2)
        img = tf.image.resize_images(img,(256,256),align_corners=True)
        img = tf.random_crop(img,[224,224,1])
        tf.assert_equal(img.shape,(224,224,1))
        return {'MRI':img,'Fields':fields},label

    dataset = tf.data.TFRecordDataset(tfrecord_file)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(decode_liver_data)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)

    return dataset

def model_fn(features,labels,mode,params):
    global_step = tf.train.get_global_step()
    img = features['MRI']
    fields = features['Fields']
    logits, end_points= inception_2d_fields.inception_2d_fields(img=img,
                                                                fields=fields,
                                                                num_classes=params['n_classes'],
                                                                spatial_squeeze=False,
                                                                dropout_keep_prob=0.6)

    # predict mode
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class':predicted_classes,
            'prob':tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
        my_acc = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_classes), tf.float32))
        tf.summary.scalar('accuracy',my_acc)


    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    # hook
    train_hook_list = []
    train_tensors_log = {'accuracy':accuracy[1],
                         'my_acc': my_acc,
                         'loss':loss,
                         'global_step':global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log,every_n_iter=200))
    #train_hook_list.append(tf_debug.LocalCLIDebugHook(ui_type="readline"))
    # training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
        #optimizer = tf.train.AdagradOptimizer(0.001)
        #optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=0.001,weight_decay=0.0002,momentum=0.9)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks=train_hook_list)
    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy':tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)

def train_val(tfrecord_folder,logdir,steps):
    n_classes = 30
    tf.logging.set_verbosity(tf.logging.INFO)
    # Train
    tf_record_path = tfrecord_folder
    train_file_name = 'train.tfrecord'
    validation_file_name = 'validation.tfrecord'

    model_path = logdir
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_path,
                                        params={
                                            'n_classes':n_classes
                                        })
    #classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[1024,512,256],n_classes=30,model_dir=model_path)
    train_tf_file = os.path.join(tf_record_path,train_file_name)

    # train_val
    train_spec = tf.estimator.TrainSpec(input_fn=lambda :train_data_input_fn(tfrecord_file=train_tf_file,perform_shuffle=False,repeat_count=-1,batch_size=32),
                                        max_steps=steps)
    eva_tf_file = os.path.join(tf_record_path, validation_file_name)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda :train_data_input_fn(eva_tf_file,False,1,64),steps=None)
    tf.estimator.train_and_evaluate(classifier,train_spec=train_spec,eval_spec=eval_spec)

if __name__ == '__main__':
    tfrecord_folder = sys.argv[1]
    logdir = sys.argv[2]
    steps = int(sys.argv[3])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_val(tfrecord_folder,logdir,steps)