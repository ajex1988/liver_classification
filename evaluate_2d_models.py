import tensorflow as tf
import os
import numpy as np
from tensorflow.python.data import Dataset

import inception_2d_train
import inception_2d_fields_train

import alexnet_train
import vgg16_train
import mobilenet_train
import resnet152_train

def dataset_3d_decode(tfrecord):
    features = {
        'height': tf.FixedLenFeature([], tf.int64, 0),
        'width': tf.FixedLenFeature([], tf.int64, 0),
        'slice':tf.FixedLenFeature([],tf.int64,0),
        'label': tf.FixedLenFeature([], tf.int64, 0),
        'volume_raw': tf.FixedLenFeature([], tf.string, '')

    }

    # Extract the data record
    sample = tf.parse_single_example(tfrecord, features)
    height = sample['height']
    width = sample['width']
    slice = sample['slice']
    label = sample['label']
    volume_raw = sample['volume_raw']
    volume = tf.decode_raw(volume_raw, tf.int16)
    volume = tf.cast(volume, tf.float32)

    # normalize to 0-255
    # volume = tf.div(tf.subtract(volume,
    #                             tf.reduce_min(volume) + 1e-8),
    #                 tf.subtract(tf.reduce_max(volume),
    #                             tf.reduce_min(volume)) + 1e-8) * 255.0
    dim = tf.stack([height, width, slice])
    # volume = tf.subtract(volume,37)
    volume = tf.reshape(volume, dim)
    volume = tf.image.resize_images(volume, (256, 256))
    volume = tf.random_crop(volume, (tf.constant(224), tf.constant(224),tf.cast(slice,tf.int32)))

    return [slice, label,volume]

def dataset_3d_fields_decode(sequence_proto):
    feature_description = {
        'height':tf.FixedLenFeature([],tf.int64,0),
        'width':tf.FixedLenFeature([],tf.int64,0),
        'slice': tf.FixedLenFeature([],tf.int64,0),
        'scanning_sequence': tf.FixedLenFeature([],tf.float32,0),
        'repetition_time':  tf.FixedLenFeature([],tf.float32,0),
        'echo_time':  tf.FixedLenFeature([],tf.float32,0),
        'echo_train_length':  tf.FixedLenFeature([],tf.float32,0),
        'flip_angle':  tf.FixedLenFeature([],tf.float32,0),
        'label': tf.FixedLenFeature([], tf.int64, 0),
        'volume_raw':tf.FixedLenFeature([],tf.string,'')
    }
    sequence = tf.parse_single_example(sequence_proto,feature_description)
    height = sequence['height']
    width = sequence['width']
    label = sequence['label']
    slice = sequence['slice']

    scanning_sequence = sequence['scanning_sequence']
    repetition_time = sequence['repetition_time']
    echo_time = sequence['echo_time']
    echo_train_length = sequence['echo_train_length']
    flip_angle = sequence['flip_angle']
    fields = tf.stack([scanning_sequence,repetition_time,echo_time,echo_train_length,flip_angle],axis=0)

    volume_raw = sequence['volume_raw']
    volume = tf.decode_raw(volume_raw, tf.int16)
    volume = tf.cast(volume, tf.float32)

    dim = tf.stack([height, width, slice])
    volume = tf.reshape(volume, dim)
    volume = tf.image.resize_images(volume, (256, 256))
    volume = tf.random_crop(volume, (tf.constant(224), tf.constant(224), tf.cast(slice, tf.int32)))

    return {'MRI':volume,'Fields':fields},label,slice

def pred_input_fn(features):
    dataset = tf.data.Dataset.from_generator(
        lambda: (features for _ in range(1)),
        output_types=tf.int32)
    iterator = dataset.batch(1).make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element, None

def evalute_2d_model(estimator,tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_3d_decode)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    total = 0
    correct = 0
    slice_wise_total = 0
    slice_wise_correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while (True):
                slice, label, volume = sess.run(next_elem)
                for i in range(slice):
                    img = volume[:,:,i]
                    img = (img - np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0 - 37
                    volume[:,:,i] = img
                print('Processing {0}...\n'.format(total))
                volume = np.transpose(volume,(2,0,1))
                volume = np.expand_dims(volume,3)

                pred_total = np.zeros((1,30))
                features = {}
                features['MRI'] = volume
                for pred in estimator.predict(tf.estimator.inputs.numpy_input_fn(x=features, shuffle=False),
                                              yield_single_examples=False):
                    pred_classes = pred['class']
                    slice_correct = np.sum(pred_classes==label)
                    slice_wise_correct += slice_correct
                    pred_total = np.sum(pred['prob'],axis=0)
                slice_wise_total += slice
                pred_total /= float(slice)
                pred_volume = np.argmax(pred_total,0)
                if pred_volume==label:
                    correct += 1
                    print('Correct')
                total += 1
        except Exception as e:
            print(e)

    print('Total: {0}'.format(total))
    print('Correct: {0}'.format(correct))
    print('Slice total: {0}'.format(slice_wise_total))
    print ('Slice correct: {0}'.format(slice_wise_correct))

def evalute_2d_model_ct_mri(estimator,tfrecord_file, result_file):
    '''
    Evaluate and save the results
    Zhe Zhu 12/02/2019
    '''
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_3d_decode)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    total = 0
    correct = 0
    slice_wise_total = 0
    slice_wise_correct = 0
    results = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while (True):
                slice, label, volume = sess.run(next_elem)
                for i in range(slice):
                    img = volume[:,:,i]
                    img = (img - np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0 - 37
                    volume[:,:,i] = img
                print('Processing {0}...\n'.format(total))
                volume = np.transpose(volume,(2,0,1))
                volume = np.expand_dims(volume,3)

                pred_total = np.zeros((1,30))
                features = {}
                features['MRI'] = volume
                for pred in estimator.predict(tf.estimator.inputs.numpy_input_fn(x=features, shuffle=False),
                                              yield_single_examples=False):
                    pred_classes = pred['class']
                    slice_correct = np.sum(pred_classes==label)
                    slice_wise_correct += slice_correct
                    pred_total = np.sum(pred['prob'],axis=0)
                slice_wise_total += slice
                pred_total /= float(slice)
                pred_volume = np.argmax(pred_total,0)
                result = []
                result.append(pred_volume)
                result.append(label)
                results.append(result)
                if pred_volume==label:
                    correct += 1
                    print('Correct')
                total += 1
        except Exception as e:
            print(e)

    print('Total: {0}'.format(total))
    print('Correct: {0}'.format(correct))
    print('Slice total: {0}'.format(slice_wise_total))
    print ('Slice correct: {0}'.format(slice_wise_correct))
    results_arr = np.array(results)
    np.savetxt(result_file,results_arr,fmt='%2d')

def evalute_2d_model_fields(estimator,tfrecord_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_3d_fields_decode)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    total = 0
    correct = 0
    slice_wise_total = 0
    slice_wise_correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while (True):
                feature,label,slice = sess.run(next_elem)
                volume = feature['MRI']
                for i in range(slice):
                    img = volume[:,:,i]
                    img = (img - np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0 - 37
                    volume[:,:,i] = img
                print('Processing {0}...\n'.format(total))
                volume = np.transpose(volume,(2,0,1))
                volume = np.expand_dims(volume,3)

                pred_total = np.zeros((1,30))
                feature['MRI'] = volume
                fields = feature['Fields']
                fields = np.expand_dims(fields, 0)
                feature['Fields'] = np.repeat(fields,slice,axis=0)
                for pred in estimator.predict(tf.estimator.inputs.numpy_input_fn(x=feature, shuffle=False),
                                              yield_single_examples=False):
                    pred_classes = pred['class']
                    slice_correct = np.sum(pred_classes==label)
                    slice_wise_correct += slice_correct
                    pred_total = np.sum(pred['prob'],axis=0)
                slice_wise_total += slice
                pred_total /= float(slice)
                pred_volume = np.argmax(pred_total,0)
                if pred_volume==label:
                    correct += 1
                    print('Correct')
                total += 1
        except Exception as e:
            print(e)

    print('Total: {0}'.format(total))
    print('Correct: {0}'.format(correct))
    print('Slice total: {0}'.format(slice_wise_total))
    print ('Slice correct: {0}'.format(slice_wise_correct))

def eva_inception_2d():
    model_dir = '/media/zzhu/Seagate Backup Plus Drive/data/Run/log/vgg_1'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_4/validation.tfrecord'
    feature_columns = [tf.feature_column.numeric_column(key='MRI', shape=(224, 224, 1))]
    n_classes = 30
    estimator = tf.estimator.Estimator(model_fn=resnet152_train.model_fn,
                                       model_dir=model_dir,
                                       params={
                                           'feature_columns': feature_columns,
                                           'n_classes': n_classes
                                       }
                                       )
    evalute_2d_model(estimator,tfrecord_file)

def eva_inception_2d_fields():
    model_dir = '/media/zzhu/Seagate Backup Plus Drive/data/Run/log/inception_2d_fields'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_3d_fields/validation.tfrecord'
    feature_columns = [tf.feature_column.numeric_column(key='MRI', shape=(224, 224, 1))]
    n_classes = 30
    estimator = tf.estimator.Estimator(model_fn=inception_2d_fields_train.model_fn,
                                       model_dir=model_dir,
                                       params={
                                           'feature_columns': feature_columns,
                                           'n_classes': n_classes
                                       }
                                       )
    evalute_2d_model_fields(estimator,tfrecord_file)


def eva_inception_2d_ct_mri():
    '''Evaluate the mew model, including both CT and MRI series
    With the updated data of patient 71
    Zhe Zhu 12/30/2019'''
    model_dir = '/mnt/sdc/Liver/CT_Data20191226/models/1227'
    tfrecord_file = '/mnt/sdc/Liver/CT_Data20191226/tfrecord/3d/validation.tfrecord'
    result_file = '/mnt/sdc/Liver/CT_Data20191226/1230_result.txt'
    feature_columns = [tf.feature_column.numeric_column(key='MRI', shape=(224, 224, 1))]
    n_classes = 34
    estimator = tf.estimator.Estimator(model_fn=inception_2d_train.model_fn,
                                       model_dir=model_dir,
                                       params={
                                           'feature_columns': feature_columns,
                                           'n_classes': n_classes
                                       }
                                       )
    evalute_2d_model_ct_mri(estimator,tfrecord_file,result_file)
if __name__=='__main__':
    #eva_inception_2d_fields()
    #eva_inception_2d()
    eva_inception_2d_ct_mri()