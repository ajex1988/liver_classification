# Construct 2d dataset from original dicom files
# The following fields were included as suggested by Mustafa
# ScanningSequence
# RepetitionTime
# EchoTime
# EchoTrainLength
# FlipAngle
from __future__ import print_function
import sys
from random import shuffle
sys.path.append('/home/zzhu/zzhu/Liver/python')
import parse_reader_csv

import os
import glob
import numpy as np
import tensorflow as tf
import csv
import pydicom
import cv2



def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


default_pixel_spacing = 0.78125

def construct_tfrecord_2d_mri_ct(file_info_list,output_file):
    '''construct a tfrecord file using file info list
        written to construct the dataset containing both ct and mri
        zhe zhu 11/24/2019
    '''
    count = 0
    pixel_spacing = default_pixel_spacing
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for file_info in file_info_list:
            count += 1
            dicom_file = file_info['dicom_file']
            vol_label = file_info['label']
            if count % 1000 == 0:
                print('{0} dicom files have been processed. Current: {1} {2}'.format(count, dicom_file, vol_label))
            ds = pydicom.dcmread(dicom_file)
            if ds.dir('PixelSpacing'):
                pixel_spacing = float(ds.PixelSpacing[0])
            img = ds.pixel_array
            if img.dtype != np.int16:
                # print('#1 Error! Not Int16 data type {}'.format(dicom_file))
                if img.dtype != np.uint16:
                    print('#1 Error! Data Type:{} type {}'.format(img.dtype, dicom_file))
                img = img.astype(np.int16)
            true_height = img.shape[0]
            true_width = img.shape[1]
            img_height = ds.Rows
            img_width = ds.Columns
            gray_scale = True
            if len(img.shape) != 2:
                gray_scale = False
            if true_height != img_height or true_width != img_width and gray_scale:
                print('#2 Error! {}: from dicom header:{} {} MRI data:{} {}'.format(dicom_file,
                                                                                       img_height, img_width,
                                                                                       true_height, true_width))
            if not gray_scale:
                print('#3 Error! Not 2 channel image: {}'.format(dicom_file))
                img = img[:, :, 0]
            img_raw = img.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(true_height),
                        'width': _int64_feature(true_width),
                        'label': _int64_feature(vol_label),
                        # 'filename': _bytes_feature(dicom_file),
                        'img_raw': _bytes_feature(img_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())
def construct_tfrecord_2d_shuffle(input_dicom_folder,info_dict_list,output_file):
    valid_patient_ids = set()
    info_map = {}
    for info_dict in info_dict_list:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict


    patient_folder_list = glob.glob(input_dicom_folder + '/*')
    all_file_list = []
    for patient_folder in patient_folder_list:
        patient_id = str(int(os.path.basename(patient_folder)[5:]))
        if patient_id in valid_patient_ids:
            volume_folder_list = glob.glob(patient_folder + '/*')
            for volume_folder in volume_folder_list:

                volume_name = os.path.basename(volume_folder).strip()

                if volume_name in info_map[patient_id]:
                    vol_info = info_map[patient_id][volume_name]
                    vol_label = int(vol_info['series_label'])
                else:
                    vol_label = 1  # anything else

                dicom_file_list = glob.glob(volume_folder + '/*')
                for i, dicom_file in enumerate(dicom_file_list):
                    info = {}
                    info['dicom_file'] = dicom_file
                    info['label'] = vol_label
                    all_file_list.append(info)
    file_num = len(all_file_list)
    print('There are in total {0} files. Begin shuffling the data...'.format(file_num))
    shuffle(all_file_list)
    print('Dataset shuffling finished')

    # write tfrecord file
    count = 0
    pixel_spacing = default_pixel_spacing
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for file_info in all_file_list:
            count += 1
            dicom_file = file_info['dicom_file']
            vol_label = file_info['label']
            if count % 1000 == 0:
                print('{0} dicom files have been processed. Current: {1} {2}'.format(count,dicom_file,vol_label))
            ds = pydicom.dcmread(dicom_file)
            if ds.dir('PixelSpacing'):
                pixel_spacing = float(ds.PixelSpacing[0])
            img = ds.pixel_array
            if img.dtype != np.int16:
                # print('#1 Error! Not Int16 data type {}'.format(dicom_file))
                if img.dtype != np.uint16:
                    print('#1 Error! Data Type:{} type {}'.format(img.dtype, dicom_file))
                img = img.astype(np.int16)
            true_height = img.shape[0]
            true_width = img.shape[1]
            img_height = ds.Rows
            img_width = ds.Columns
            gray_scale = True
            if len(img.shape) != 2:
                gray_scale = False
            if true_height != img_height or true_width != img_width and gray_scale:
                print('#2 Error! {}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                       volume_name,
                                                                                       img_height, img_width,
                                                                                       true_height, true_width))
            if not gray_scale:
                print('#3 Error! Not 2 channel image: {}_{}'.format(patient_id, volume_name))
                img = img[:, :, 0]
            img_raw = img.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(true_height),
                        'width': _int64_feature(true_width),
                        'label': _int64_feature(vol_label),
                        #'filename': _bytes_feature(dicom_file),
                        'img_raw': _bytes_feature(img_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def construct_tfrecord_2d_fields_shuffle(input_dicom_folder,info_dict_list,output_file):
    scanning_sequence_set_0 = set(["SE","['EP', 'SE']","['SE','IR']","['IR', 'SE']","['EP', 'SE', 'EP']","SE/IR"])
    scanning_sequence_set_1 = set(["GR","['GR', 'IR']"])

    valid_patient_ids = set()
    info_map = {}
    for info_dict in info_dict_list:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict


    patient_folder_list = glob.glob(input_dicom_folder + '/*')
    all_file_list = []
    for patient_folder in patient_folder_list:
        patient_id = str(int(os.path.basename(patient_folder)[5:]))
        if patient_id in valid_patient_ids:
            volume_folder_list = glob.glob(patient_folder + '/*')
            for volume_folder in volume_folder_list:

                volume_name = os.path.basename(volume_folder).strip()

                if volume_name in info_map[patient_id]:
                    vol_info = info_map[patient_id][volume_name]
                    vol_label = int(vol_info['series_label'])
                else:
                    vol_label = 1  # anything else

                dicom_file_list = glob.glob(volume_folder + '/*')
                for i, dicom_file in enumerate(dicom_file_list):
                    info = {}
                    info['dicom_file'] = dicom_file
                    info['label'] = vol_label
                    all_file_list.append(info)
    file_num = len(all_file_list)
    print('There are in total {0} files. Begin shuffling the data...'.format(file_num))
    shuffle(all_file_list)
    print('Dataset shuffling finished')

    # write tfrecord file
    count = 0
    pixel_spacing = default_pixel_spacing
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for file_info in all_file_list:
            count += 1
            dicom_file = file_info['dicom_file']
            vol_label = file_info['label']
            if count % 1000 == 0:
                print('{0} dicom files have been processed. Current: {1} {2}'.format(count,dicom_file,vol_label))
            ds = pydicom.dcmread(dicom_file)
            if ds.dir('PixelSpacing'):
                pixel_spacing = float(ds.PixelSpacing[0])
            img = ds.pixel_array
            if img.dtype != np.int16:
                # print('#1 Error! Not Int16 data type {}'.format(dicom_file))
                if img.dtype != np.uint16:
                    print('#1 Error! Data Type:{} type {}'.format(img.dtype, dicom_file))
                img = img.astype(np.int16)
            true_height = img.shape[0]
            true_width = img.shape[1]
            img_height = ds.Rows
            img_width = ds.Columns
            gray_scale = True
            if len(img.shape) != 2:
                gray_scale = False
            if true_height != img_height or true_width != img_width and gray_scale:
                print('#2 Error! {}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                       volume_name,
                                                                                       img_height, img_width,
                                                                                       true_height, true_width))
            if not gray_scale:
                print('#3 Error! Not 2 channel image: {}_{}'.format(patient_id, volume_name))
                img = img[:, :, 0]
            img_raw = img.tostring()
            if ds.dir('ScanningSequence'):
                scanning_sequence = ds.ScanningSequence
            else:
                scanning_sequence = 'NULL'
            if ds.dir('RepetitionTime'):
                repetition_time = ds.RepetitionTime
            else:
                repetition_time = 'NULL'
            if ds.dir('EchoTime'):
                echo_time = ds.EchoTime
            else:
                echo_time = 'NULL'
            if ds.dir('EchoTrainLength'):
                echo_train_length = ds.EchoTrainLength
            else:
                echo_train_length = 'NULL'
            if ds.dir('FlipAngle'):
                flip_angle = ds.FlipAngle
            else:
                flip_angle = 'NULL'

            # make those fileds into features
            scanning_sequence = str(scanning_sequence)
            if scanning_sequence in scanning_sequence_set_0:
                scanning_sequence = 0*128
            elif scanning_sequence in scanning_sequence_set_1:
                scanning_sequence = 1*128
            else:
                scanning_sequence = 2*128
            if repetition_time == 'NULL':
                repetition_time = -1
            else:
                repetition_time = repetition_time / 10.0
                if repetition_time>255.0:
                    repetition_time = 255.0
            if echo_time == 'NULL':
                echo_time = -1
            if echo_train_length == 'NULL':
                echo_train_length = -1
            if flip_angle == 'NULL':
                flip_angle = -1

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(true_height),
                        'width': _int64_feature(true_width),
                        'label': _int64_feature(vol_label),
                        'scanning_sequence':_float_feature(scanning_sequence),
                        'repetition_time':_float_feature(repetition_time),
                        'echo_time':_float_feature(echo_time),
                        'echo_train_length':_float_feature(echo_train_length),
                        'flip_angle':_float_feature(flip_angle),
                        #'filename': _bytes_feature(dicom_file),
                        'img_raw': _bytes_feature(img_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())



def construct_tfrecord_2d(input_dicom_folder,info_dict_list,output_folder,output_mode,use_fields):
    if use_fields:
        print('Generating tfrecord with fields information')
    else:
        print('Generating tfrecord using image data only')
    tfrecord_filename = ''
    if output_mode == 'train':
        tfrecord_filename = 'train.tfrecord'
    elif output_mode == 'validation':
        tfrecord_filename = 'validation.tfrecord'
    else:
        raise Exception('Output mode should be either train or test')
    tfrecord_file = os.path.join(output_folder,tfrecord_filename)

    valid_patient_ids = set()
    info_map = {}
    for info_dict in info_dict_list:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict

    count = 0
    patient_folder_list = glob.glob(input_dicom_folder+'/*')
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:

        for patient_folder in patient_folder_list:
            patient_id = str(int(os.path.basename(patient_folder)[5:]))
            if patient_id in valid_patient_ids:
                volume_folder_list = glob.glob(patient_folder+'/*')
                for volume_folder in volume_folder_list:
                    count += 1
                    volume_name = os.path.basename(volume_folder).strip()
                    if count % 50 == 0:
                        print('{}. Writting volume {}_{}...'.format(count, patient_id,volume_name))
                    if volume_name in info_map[patient_id]:
                        vol_info = info_map[patient_id][volume_name]
                        vol_label = int(vol_info['series_label'])
                    else:
                        vol_label = 1 # anything else

                    pixel_spacing = default_pixel_spacing

                    dicom_file_list = glob.glob(volume_folder + '/*')

                    for i, dicom_file in enumerate(dicom_file_list):
                        ds = pydicom.dcmread(dicom_file)
                        if ds.dir('PixelSpacing'):
                            pixel_spacing = float(ds.PixelSpacing[0])
                        if use_fields:
                            scanning_sequence = ds.ScanningSequence
                            repetition_time = ds.RepetitionTime
                            echo_time = ds.EchoTime
                            echo_train_length = ds.EchoTrainLength
                            flip_angle = ds.FlipAngle
                        img = ds.pixel_array
                        if img.dtype != np.int16:
                            #print('#1 Error! Not Int16 data type {}'.format(dicom_file))
                            if img.dtype != np.uint16:
                                print('#1 Error! Data Type:{} type {}'.format(img.dtype,dicom_file))
                            img = img.astype(np.int16)
                        true_height = img.shape[0]
                        true_width = img.shape[1]
                        img_height = ds.Rows
                        img_width = ds.Columns
                        gray_scale = True
                        if len(img.shape) != 2:
                            gray_scale = False
                        if true_height != img_height or true_width != img_width and gray_scale:
                            print('#2 Error! {}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                         volume_name,
                                                                                         img_height,img_width,
                                                                                         true_height,true_width))
                        if not gray_scale:
                            print('#3 Error! Not 2 channel image: {}_{}'.format(patient_id,volume_name))
                            img = img[:,:,0]
                        img_raw = img.tostring()
                        if use_fields:
                            example = tf.train.Example(
                                features=tf.train.Features(
                                    feature={
                                        'height': _int64_feature(true_height),
                                        'width': _int64_feature(true_width),
                                        'scanning_sequence': _bytes_feature(scanning_sequence),
                                        'repetition_time': _float_feature(repetition_time),
                                        'echo_time': _float_feature(echo_time),
                                        'echo_train_length': _float_feature(echo_train_length),
                                        'flip_angle': _float_feature(flip_angle),
                                        'label': _int64_feature(vol_label),
                                        'img_raw': _bytes_feature(img_raw)
                                    }
                                )
                            )
                        else:
                            example = tf.train.Example(
                                features=tf.train.Features(
                                    feature={
                                        'height': _int64_feature(true_height),
                                        'width': _int64_feature(true_width),
                                        'label': _int64_feature(vol_label),
                                        'patient_id':_bytes_feature(patient_id),
                                        'volume_id':_bytes_feature(volume_name),
                                        'filename':_bytes_feature(os.path.basename(dicom_file)),
                                        'img_raw': _bytes_feature(img_raw)
                                    }
                                )
                            )
                        writer.write(example.SerializeToString())





def main_1():
    # ai_vs_radiologist_2d no field validation
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_external'
    info_dict_file = '/home/zzhu/Data/Liver/Reader Data/LIRADSMachineLearnin_DATA_mustafa.csv'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d'
    output_mode = 'validation'
    col_begin = 1
    col_end = 28

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map_fatonly.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)

    construct_tfrecord_2d(input_dicom_folder,series_info,output_folder,output_mode,False)

def main_2():
    # ai_vs_radiologist_2d no field train
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d'
    output_mode = 'train'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)

    construct_tfrecord_2d(input_dicom_folder,series_info,output_folder,output_mode,False)

def main_3():
    # random shuffle
    l = [[i] for i in range(10)]
    shuffle(l)
    print(l)

def main_4():
    # generate shuffled dataset
    # train set
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d/train.tfrecord'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder,info_dict_list=series_info,output_file=output_file)

def main_5():
    # generate shuffled dataset
    # validation set
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_external'
    info_dict_file = '/home/zzhu/Data/Liver/Reader Data/LIRADSMachineLearnin_DATA_mustafa.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d/validation.tfrecord'
    col_begin = 1
    col_end = 28

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map_fatonly.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder,info_dict_list=series_info,output_file=output_file)

def main_6():
    # construct the 2d train dataset that can be published
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/publish_train_decompressed'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/tf_dataset/internal_2d/train.tfrecord'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)

def main_7():
    #construct 2d val dataset that can be published
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/publish_val_decompressed'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/tf_dataset/internal_2d/validation.tfrecord'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)
def main_8():
    # create 5-fold datasets
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord

    # fold 1 train
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_1/train'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_1/train.tfrecord'
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)
    # folder 1 val
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_1/val'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_1/validation.tfrecord'
    construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)

    # # fold 2 train
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_2/train'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_2/train.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)
    # # folder 2 val
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_2/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_2/validation.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)
    #
    # # fold 3 train
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_3/train'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_3/train.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)
    # # folder 3 val
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_3/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_3/validation.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)
    #
    # # fold 4 train
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_4/train'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_4/train.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)
    # # folder 4 val
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_4/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord/cv_4/validation.tfrecord'
    # construct_tfrecord_2d_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
    #                               output_file=output_file)

def main_9():
    # ai vs radiologist using fields train
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d_fields/train.tfrecord'
    col_begin = 5
    col_end = 43

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_fields_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)

def main_10():
    # ai vs radiologists val with fields
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_external'
    info_dict_file = '/home/zzhu/Data/Liver/Reader Data/LIRADSMachineLearnin_DATA_mustafa.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_2d_fields/validation.tfrecord'
    col_begin = 1
    col_end = 28

    map_txt = '/home/zzhu/Data/data/ai_vs_radiologist/map_fatonly.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            if series_name not in name_label_dict:
                name_label_dict[series_name] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse(info_dict_file, name_label_dict)
    # construct tfrecord
    construct_tfrecord_2d_fields_shuffle(input_dicom_folder=input_dicom_folder, info_dict_list=series_info,
                                  output_file=output_file)

def main_11():
    '''
    Construct the tfrecord file containing the CT data
    :return:
    '''
    dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/LIRADSMachineLearnin_DATA_2019-11-24_1657.csv'
    col_begin = 5
    col_end = 42 # precontrast_ct has been deleted manually for consistency

    output_file_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/train.tfrecord'
    output_file_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/val.tfrecord'

    map_txt = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/map_ct_mri.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['mri_dwi_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_hepatocyte'] = name_label_dict['mri_hepa_trans']
    name_label_dict['mri_transitional'] = name_label_dict['mri_hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            series_name_mri = 'mri_' + series_name
            series_name_ct = 'ct_' + series_name
            if series_name_mri not in name_label_dict:
                name_label_dict[series_name_mri] = name_label_dict['anythingelse']
            if series_name_ct not in name_label_dict:
                name_label_dict[series_name_ct] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse_v2(info_dict_file, name_label_dict)

    # seperate train val
    valid_patient_ids = set()
    info_map = {}
    for info_dict in series_info:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict

    patient_folder_list = glob.glob(dicom_folder + '/*')

    train_file_list = []
    val_file_list = []
    all_file_list = []
    for patient_folder in patient_folder_list:
        patient_id = str(int(os.path.basename(patient_folder)[5:]))
        if patient_id in valid_patient_ids:
            volume_folder_list = glob.glob(patient_folder + '/*')
            for volume_folder in volume_folder_list:

                volume_name = os.path.basename(volume_folder).strip()

                if volume_name in info_map[patient_id]:
                    vol_info = info_map[patient_id][volume_name]
                    vol_label = int(vol_info['series_label'])
                else:
                    vol_label = 1  # anything else

                dicom_file_list = glob.glob(volume_folder + '/*')
                for i, dicom_file in enumerate(dicom_file_list):
                    info = {}
                    info['dicom_file'] = dicom_file
                    info['label'] = vol_label
                    if int(patient_id)%5 == 0:
                        val_file_list.append(info)
                    else:
                        train_file_list.append(info)
                    all_file_list.append(info)
    file_num = len(all_file_list)
    print('There are in total {0} files. Begin shuffling the data...'.format(file_num))
    #shuffle(all_file_list)
    shuffle(train_file_list)
    print('Dataset shuffling finished')

    print('Constructing train tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=train_file_list,output_file=output_file_train)
    print('Train file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=val_file_list,output_file=output_file_val)
    print('Val file generated')

def main_12():
    '''Construct the dataset containing both the ct and mri series
    Zhe Zhu 2019/12/24
    '''
    dicom_folder_mri = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    dicom_folder_ct = '/mnt/sdc/Liver/CT_Data/CT_Final'
    info_dict_file = '/mnt/sdc/Liver/CT_Data/LIRADSMachineLearnin_DATA_2019-12-20_1645.csv'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

    output_file_train = '/mnt/sdc/Liver/CT_Data/tfrecord/2d/train.tfrecord'
    output_file_val = '/mnt/sdc/Liver/CT_Data/tfrecord/2d/val.tfrecord'

    map_txt = '/mnt/sdc/Liver/CT_Data/map_ct_mri.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['mri_dwi_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_hepatocyte'] = name_label_dict['mri_hepa_trans']
    name_label_dict['mri_transitional'] = name_label_dict['mri_hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            series_name_mri = 'mri_' + series_name
            series_name_ct = 'ct_' + series_name
            if series_name_mri not in name_label_dict:
                name_label_dict[series_name_mri] = name_label_dict['anythingelse']
            if series_name_ct not in name_label_dict:
                name_label_dict[series_name_ct] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse_v2(info_dict_file, name_label_dict)

    # seperate train val
    valid_patient_ids = set()
    info_map = {}
    for info_dict in series_info:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict

    patient_folder_list_mri = glob.glob(dicom_folder_mri + '/*')

    train_file_list = []
    val_file_list = []
    # process ct
    ct_patient_list = glob.glob(dicom_folder_ct + '/*')
    for ct_patient_folder in ct_patient_list:
        ct_series_folder_list = glob.glob(ct_patient_folder + '/*')
        ct_patient_id = os.path.basename(ct_patient_folder)[5:9]
        for ct_series_folder in ct_series_folder_list:
            ct_series_name = os.path.basename(ct_series_folder)
            ct_series_label = name_label_dict[ct_series_name]
            ct_dicom_file_list = glob.glob(ct_series_folder + '/*')
            for ct_dicom_file in ct_dicom_file_list:
                info = {}
                info['dicom_file'] = ct_dicom_file
                info['label'] = ct_series_label
                if int(ct_patient_id) % 5 == 0:
                    val_file_list.append(info)
                else:
                    train_file_list.append(info)
    #process mri
    for patient_folder in patient_folder_list_mri:
        patient_id = str(int(os.path.basename(patient_folder)[5:]))
        if patient_id in valid_patient_ids:
            volume_folder_list = glob.glob(patient_folder + '/*')
            for volume_folder in volume_folder_list:

                volume_name = os.path.basename(volume_folder).strip()

                if volume_name in info_map[patient_id]:
                    vol_info = info_map[patient_id][volume_name]
                    vol_label = int(vol_info['series_label'])
                else:
                    vol_label = 1  # anything else

                dicom_file_list = glob.glob(volume_folder + '/*')
                for i, dicom_file in enumerate(dicom_file_list):
                    info = {}
                    info['dicom_file'] = dicom_file
                    info['label'] = vol_label
                    if int(patient_id) % 5 == 0:
                        val_file_list.append(info)
                    else:
                        train_file_list.append(info)
                    #all_file_list.append(info)


    file_num = len(train_file_list)+len(val_file_list)
    print('There are in total {0} files. Begin shuffling the data...'.format(file_num))
    # shuffle(all_file_list)
    shuffle(train_file_list)
    print('Dataset shuffling finished')

    print('Constructing train tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=train_file_list, output_file=output_file_train)
    print('Train file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=val_file_list, output_file=output_file_val)
    print('Val file generated')

def main_13():
    '''Construct the dataset containing both the ct and mri series
    With the updated data of patient 71
    Zhe Zhu 2019/12/27
    '''
    dicom_folder_mri = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    dicom_folder_ct = '/mnt/sdc/Liver/CT_Data20191226/CT_Final'
    info_dict_file = '/mnt/sdc/Liver/CT_Data20191226/LIRADSMachineLearnin_DATA_2019-12-26_2215.csv'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

    output_file_train = '/mnt/sdc/Liver/CT_Data20191226/tfrecord/2d/train.tfrecord'
    output_file_val = '/mnt/sdc/Liver/CT_Data20191226/tfrecord/2d/validation.tfrecord'

    map_txt = '/mnt/sdc/Liver/CT_Data/map_ct_mri.txt'

    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['mri_dwi_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_hepatocyte'] = name_label_dict['mri_hepa_trans']
    name_label_dict['mri_transitional'] = name_label_dict['mri_hepa_trans']

    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            series_name_mri = 'mri_' + series_name
            series_name_ct = 'ct_' + series_name
            if series_name_mri not in name_label_dict:
                name_label_dict[series_name_mri] = name_label_dict['anythingelse']
            if series_name_ct not in name_label_dict:
                name_label_dict[series_name_ct] = name_label_dict['anythingelse']

    # load info_dict
    series_info = parse_reader_csv.parse_v2(info_dict_file, name_label_dict)

    # seperate train val
    valid_patient_ids = set()
    info_map = {}
    for info_dict in series_info:
        valid_patient_ids.add(info_dict['patient_id'])
        if info_dict['patient_id'] in info_map:
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict
        else:
            info_map[info_dict['patient_id']] = {}
            info_map[info_dict['patient_id']][info_dict['str'].strip()] = info_dict

    patient_folder_list_mri = glob.glob(dicom_folder_mri + '/*')

    train_file_list = []
    val_file_list = []
    # process ct
    ct_patient_list = glob.glob(dicom_folder_ct + '/*')
    for ct_patient_folder in ct_patient_list:
        ct_series_folder_list = glob.glob(ct_patient_folder + '/*')
        ct_patient_id = os.path.basename(ct_patient_folder)[5:9]
        for ct_series_folder in ct_series_folder_list:
            ct_series_name = os.path.basename(ct_series_folder)
            ct_series_label = name_label_dict[ct_series_name]
            ct_dicom_file_list = glob.glob(ct_series_folder + '/*')
            for ct_dicom_file in ct_dicom_file_list:
                info = {}
                info['dicom_file'] = ct_dicom_file
                info['label'] = ct_series_label
                if int(ct_patient_id) % 5 == 0:
                    val_file_list.append(info)
                else:
                    train_file_list.append(info)
    #process mri
    for patient_folder in patient_folder_list_mri:
        patient_id = str(int(os.path.basename(patient_folder)[5:]))
        if patient_id in valid_patient_ids:
            volume_folder_list = glob.glob(patient_folder + '/*')
            for volume_folder in volume_folder_list:

                volume_name = os.path.basename(volume_folder).strip()

                if volume_name in info_map[patient_id]:
                    vol_info = info_map[patient_id][volume_name]
                    vol_label = int(vol_info['series_label'])
                else:
                    vol_label = 1  # anything else

                dicom_file_list = glob.glob(volume_folder + '/*')
                for i, dicom_file in enumerate(dicom_file_list):
                    info = {}
                    info['dicom_file'] = dicom_file
                    info['label'] = vol_label
                    if int(patient_id) % 5 == 0:
                        val_file_list.append(info)
                    else:
                        train_file_list.append(info)
                    #all_file_list.append(info)


    file_num = len(train_file_list)+len(val_file_list)
    print('There are in total {0} files. Begin shuffling the data...'.format(file_num))
    # shuffle(all_file_list)
    shuffle(train_file_list)
    print('Dataset shuffling finished')

    print('Constructing train tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=train_file_list, output_file=output_file_train)
    print('Train file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_2d_mri_ct(file_info_list=val_file_list, output_file=output_file_val)
    print('Val file generated')
if __name__ == '__main__':
    #main_1()
    #main_2()
    #main_6()
    #main_7()
    #main_8()
    #main_9()
    #main_10()
    #main_11()
    #main_12()
    main_13()


