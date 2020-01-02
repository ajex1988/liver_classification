import tensorflow as tf
import stpnet
import os
import numpy as np
import sys

import config

def train_input_fn(tfrecord_file,batch_size,repeat_count,perform_shuffle=False,uniform_resize=False,
                   min_slice=32,resize_hw=256,crop_hw=224):
    def decode_liver_data(sequence_proto):
        feature_description = {'height': tf.FixedLenFeature((),tf.int64,0),
                               'width':tf.FixedLenFeature((),tf.int64,0),
                               'slice':tf.FixedLenFeature((),tf.int64,0),
                               'label':tf.FixedLenFeature((),tf.int64,0),
                               'volume_raw':tf.FixedLenFeature((),tf.string,'')}
        sequence = tf.parse_single_example(sequence_proto,feature_description)
        height = sequence['height']
        width = sequence['width']
        slice_num = sequence['slice']
        label = sequence['label']
        volume = tf.decode_raw(sequence['volume_raw'],tf.int16)
        volume = tf.cast(volume,tf.float32)

        # normalize to 0-255
        volume = tf.div(tf.subtract(volume,
                                    tf.reduce_min(volume) + 1e-8),
                        tf.subtract(tf.reduce_max(volume),
                                    tf.reduce_min(volume)) + 1e-8) * 255.0
        dim = tf.stack([height,width,slice_num])
        volume = tf.reshape(volume,dim)

        def func_h(height):
            return height
        def func_w(width):
            return width
        min_dim = tf.cond(height<width,lambda :func_h(height),lambda :func_w(width))

        def func_t(volume,resize_hw,min_dim,height,width):
            #uniform resize
            scale_rate = tf.div(float(resize_hw),tf.cast(min_dim,tf.float32))
            resized_height = tf.cast((tf.round(tf.cast(height,tf.float32)*scale_rate)),tf.int32)
            resized_width = tf.cast((tf.round(tf.cast(width,tf.float32)*scale_rate)),tf.int32)
            return tf.image.resize_images(volume,(resized_height,resized_width))
        def func_f(volume,resize_hw):
            #non-uniform resize
            return tf.image.resize_images(volume,(resize_hw,resize_hw))

        # resize the volume depending on the size
        if uniform_resize:
            tf_b_uniform = tf.constant(True,dtype=tf.bool)
        else:
            tf_b_uniform = tf.constant(False,dtype=tf.bool)
        volume = tf.cond(tf_b_uniform,lambda :func_t(volume,resize_hw,min_dim,height,width),lambda :func_f(volume,resize_hw))
        # subtract mean
        volume = tf.subtract(volume,37) # Caffe Style

        # subsampling
        volume = tf.random_crop(volume,(crop_hw,crop_hw,min_slice))

        # convert to conv3d formats NDHWC
        volume = tf.expand_dims(volume,3)
        volume = tf.transpose(volume,[2,0,1,3])

        return {'MRI':volume},label
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_liver_data)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=64)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size)

    return dataset

def model_fn(features,labels,mode,params):
    global_step = tf.train.get_global_step()
    inputs = features['MRI']
    logits, end_points = stpnet.stpnet_v3(inputs=inputs,
                                                num_outputs=params['n_classes'],
                                                pooling_kernel_size=params['stpl_kernel_size'],
                                                pooling_stride=params['stpl_stride'])
    # predict mode
    predicted_classes = tf.argmax(logits, 1)
    with tf.name_scope('accuracy'):
        accumulate_accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
        batch_acc = tf.reduce_mean(tf.cast(tf.equal(labels, predicted_classes), tf.float32))
        tf.summary.scalar('accumulate_accuracy', accumulate_accuracy[1])
        tf.summary.scalar('batch_accuracy', batch_acc)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # compute loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # hook
    train_hook_list = []
    train_tensors_log = {'accumulate_accuracy': accumulate_accuracy[1],
                         'batch_acc': batch_acc,
                         'loss': loss,
                         'global_step': global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(tensors=train_tensors_log, every_n_iter=100))

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=train_hook_list)
    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def single_size_trainval(train_config):
    tf.logging.set_verbosity(tf.logging.INFO)

    model_path = train_config.model_path
    batch_size = train_config.batch_size
    tfrecord_folder = train_config.tfrecord_folder
    resize_hw = train_config.resize_hw
    crop_hw = resize_hw - train_config.crop_diff
    max_steps = train_config.max_steps
    n_classes = train_config.n_classes
    min_slice = train_config.min_slice

    stpl_kernel_size = train_config.stpl_kernel_size
    stpl_stride = train_config.stpl_stride


    tfrecord_file_name_train = 'train.tfrecord'
    tfrecord_file_train = os.path.join(tfrecord_folder, tfrecord_file_name_train)
    tfrecord_file_name_val = 'validation.tfrecord'
    tfrecord_file_val = os.path.join(tfrecord_folder,tfrecord_file_name_val)
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=model_path,
                                        params={
                                            'stpl_kernel_size': stpl_kernel_size,
                                            'stpl_stride': stpl_stride,
                                            'n_classes': n_classes
                                        }
                                        )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda :train_input_fn(tfrecord_file=tfrecord_file_train,
                                                                        batch_size=batch_size,
                                                                        resize_hw=resize_hw,
                                                                        crop_hw=crop_hw,
                                                                        min_slice=min_slice,
                                                                        repeat_count=-1),
                                        max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda :train_input_fn(tfrecord_file=tfrecord_file_val,
                                                                      batch_size=batch_size,
                                                                      resize_hw=resize_hw,
                                                                      crop_hw=crop_hw,
                                                                      min_slice=min_slice,
                                                                      repeat_count=1))
    tf.estimator.train_and_evaluate(classifier,train_spec,eval_spec)

def two_sizes_train_val(train_config):
    tf.logging.set_verbosity(tf.logging.INFO)
    num_epochs = train_config.num_epochs
    max_steps = train_config.max_steps
    param_list = train_config.param_list
    model_path = train_config.model_path
    tfrecord_folder = train_config.tfrecord_folder
    n_classes = train_config.n_classes
    batch_size = train_config.batch_size
    for epoch in range(num_epochs):
        for param in param_list:
            min_slice = param['min_slice']
            resize_hw = param['resize_hw']
            crop_hw = resize_hw - train_config.crop_diff
            stpl_kernel_size = param['stpl_kernel_size']
            stpl_stride = param['stpl_stride']
            tfrecord_file_name_train = 'train.tfrecord'
            tfrecord_file_train = os.path.join(tfrecord_folder, tfrecord_file_name_train)
            tfrecord_file_name_val = 'validation.tfrecord'
            tfrecord_file_val = os.path.join(tfrecord_folder, tfrecord_file_name_val)
            classifier = tf.estimator.Estimator(model_fn=model_fn,
                                                model_dir=model_path,
                                                params={
                                                    'stpl_kernel_size': stpl_kernel_size,
                                                    'stpl_stride': stpl_stride,
                                                    'n_classes': n_classes
                                                })
            train_spec = tf.estimator.TrainSpec(input_fn=lambda :train_input_fn(tfrecord_file=tfrecord_file_train,
                                                                        batch_size=batch_size,
                                                                        resize_hw=resize_hw,
                                                                        crop_hw=crop_hw,
                                                                        min_slice=min_slice,
                                                                        repeat_count=25),
                                        max_steps=max_steps)
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(tfrecord_file=tfrecord_file_val,
                                                                              batch_size=batch_size,
                                                                              resize_hw=resize_hw,
                                                                              crop_hw=crop_hw,
                                                                              min_slice=min_slice,
                                                                              repeat_count=1))
            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
    train_config = config.STPNSingleSize
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = train_config.cuda_device_id
    single_size_trainval(train_config=train_config)
