# Construct 3d dataset from original dicom files
# The created dataset is in tfrecord format
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

def construct_tfrecord_3d_mri_ct(series_info_list,output_file):
    '''
    construct 3d dataset for evaluation. The dataset contains both mri and ct data, including 36 series types in total.
    written by zhe zhu 11/26/2019
    :param series_info_list: series-wise information listr. used for constructing the dataset
    :param output_file: output tfrecord file
    :return: none
    '''
    count = 0
    with tf.python_io.TFRecordWriter(output_file) as writer:
        for series_info in series_info_list:
            volume_folder = series_info['dicom_folder']
            vol_label = series_info['label']
            count += 1
            volume_name = os.path.basename(volume_folder).strip()
            if count % 50 == 0:
                print('{}. Writting volume {}...'.format(count, volume_folder))

            dicom_file_list = glob.glob(volume_folder + '/*')
            if len(dicom_file_list) == 0:
                print('Empty Folder! '+volume_folder)
                continue
            anchor_dicom_file = dicom_file_list[0]
            ds_anchor = pydicom.dcmread(anchor_dicom_file)
            if ds_anchor.dir('PixelSpacing'):
                pixel_spacing = float(ds_anchor.PixelSpacing[0])
            img_height = ds_anchor.Rows
            img_width = ds_anchor.Columns
            img_slice = len(dicom_file_list)
            vol_data = np.zeros((img_height, img_width, img_slice), dtype=np.int16)

            for i, dicom_file in enumerate(dicom_file_list):
                ds = pydicom.dcmread(dicom_file)
                img = ds.pixel_array
                true_height = img.shape[0]
                true_width = img.shape[1]
                gray_scale = True
                if len(img.shape) != 2:
                    gray_scale = False
                if true_height != img_height or true_width != img_width and gray_scale:
                    print('{}: from dicom header:{} {} MRI data:{} {}'.format(volume_folder,
                                                                                 img_height, img_width,
                                                                                 true_height, true_width))
                    img = cv2.resize(img, (img_width, img_height))
                if not gray_scale:
                    print('Error! Not 2 channel image: {}'.format(volume_folder))
                    img = img[:, :, 0]
                vol_data[:, :, i] = img
            volume_raw = vol_data.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(img_height),
                        'width': _int64_feature(img_width),
                        'slice': _int64_feature(img_slice),
                        'label': _int64_feature(vol_label),
                        'volume_raw': _bytes_feature(volume_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def construct_tfrecord_3d_shuffle(input_dicom_folder,info_dict_list,output_file):
    tfrecord_file = output_file
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
    all_sequence_list = []
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
                info = {}
                info['dicom_folder'] = volume_folder
                info['label'] = vol_label
                all_sequence_list.append(info)

    file_num = len(all_sequence_list)
    print('There are in total {0} sequences. Begin shuffling the data...'.format(file_num))
    shuffle(all_sequence_list)
    print('Dataset shuffling finished')

    count = 0
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for sequence_info in all_sequence_list:
            volume_folder = sequence_info['dicom_folder']
            vol_label = sequence_info['label']
            count += 1
            volume_name = os.path.basename(volume_folder).strip()
            if count % 50 == 0:
                print('{}. Writting volume {}_{}...'.format(count, patient_id, volume_name))

            dicom_file_list = glob.glob(volume_folder + '/*')

            anchor_dicom_file = dicom_file_list[0]
            ds_anchor = pydicom.dcmread(anchor_dicom_file)
            if ds_anchor.dir('PixelSpacing'):
                pixel_spacing = float(ds_anchor.PixelSpacing[0])
            img_height = ds_anchor.Rows
            img_width = ds_anchor.Columns
            img_slice = len(dicom_file_list)
            vol_data = np.zeros((img_height, img_width, img_slice), dtype=np.int16)

            for i, dicom_file in enumerate(dicom_file_list):
                ds = pydicom.dcmread(dicom_file)
                img = ds.pixel_array
                true_height = img.shape[0]
                true_width = img.shape[1]
                gray_scale = True
                if len(img.shape) != 2:
                    gray_scale = False
                if true_height != img_height or true_width != img_width and gray_scale:
                    print('{}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                 volume_name,
                                                                                 img_height, img_width,
                                                                                 true_height, true_width))
                    img = cv2.resize(img, (img_width, img_height))
                if not gray_scale:
                    print('Error! Not 2 channel image: {}_{}'.format(patient_id, volume_name))
                    img = img[:, :, 0]
                vol_data[:, :, i] = img
            volume_raw = vol_data.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(img_height),
                        'width': _int64_feature(img_width),
                        'slice': _int64_feature(img_slice),
                        'label': _int64_feature(vol_label),
                        'volume_raw': _bytes_feature(volume_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def construct_tfrecord_3d_fields_shuffle(input_dicom_folder,info_dict_list,output_file):
    scanning_sequence_set_0 = set(["SE", "['EP', 'SE']", "['SE','IR']", "['IR', 'SE']", "['EP', 'SE', 'EP']", "SE/IR"])
    scanning_sequence_set_1 = set(["GR", "['GR', 'IR']"])

    tfrecord_file = output_file
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
    all_sequence_list = []
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
                info = {}
                info['dicom_folder'] = volume_folder
                info['label'] = vol_label
                all_sequence_list.append(info)

    file_num = len(all_sequence_list)
    print('There are in total {0} sequences. Begin shuffling the data...'.format(file_num))
    shuffle(all_sequence_list)
    print('Dataset shuffling finished')

    count = 0
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for sequence_info in all_sequence_list:
            volume_folder = sequence_info['dicom_folder']
            vol_label = sequence_info['label']
            count += 1
            volume_name = os.path.basename(volume_folder).strip()
            if count % 50 == 0:
                print('{}. Writting volume {}_{}...'.format(count, patient_id, volume_name))

            dicom_file_list = glob.glob(volume_folder + '/*')

            anchor_dicom_file = dicom_file_list[0]
            ds_anchor = pydicom.dcmread(anchor_dicom_file)
            if ds_anchor.dir('PixelSpacing'):
                pixel_spacing = float(ds_anchor.PixelSpacing[0])
            img_height = ds_anchor.Rows
            img_width = ds_anchor.Columns
            img_slice = len(dicom_file_list)
            vol_data = np.zeros((img_height, img_width, img_slice), dtype=np.int16)

            for i, dicom_file in enumerate(dicom_file_list):
                ds = pydicom.dcmread(dicom_file)
                img = ds.pixel_array
                true_height = img.shape[0]
                true_width = img.shape[1]
                gray_scale = True
                if len(img.shape) != 2:
                    gray_scale = False
                if true_height != img_height or true_width != img_width and gray_scale:
                    print('{}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                 volume_name,
                                                                                 img_height, img_width,
                                                                                 true_height, true_width))
                    img = cv2.resize(img, (img_width, img_height))
                if not gray_scale:
                    print('Error! Not 2 channel image: {}_{}'.format(patient_id, volume_name))
                    img = img[:, :, 0]
                vol_data[:, :, i] = img
            volume_raw = vol_data.tostring()
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
                        'height': _int64_feature(img_height),
                        'width': _int64_feature(img_width),
                        'slice': _int64_feature(img_slice),
                        'label': _int64_feature(vol_label),
                        'scanning_sequence': _float_feature(scanning_sequence),
                        'repetition_time': _float_feature(repetition_time),
                        'echo_time': _float_feature(echo_time),
                        'echo_train_length': _float_feature(echo_train_length),
                        'flip_angle': _float_feature(flip_angle),
                        'volume_raw': _bytes_feature(volume_raw)
                    }
                )
            )
            writer.write(example.SerializeToString())

def construct_tfrecord_3d(input_dicom_folder,info_dict_list,output_folder,output_file):
    tfrecord_file = output_file

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

                    anchor_dicom_file = dicom_file_list[0]
                    ds_anchor = pydicom.dcmread(anchor_dicom_file)
                    if ds_anchor.dir('PixelSpacing'):
                        pixel_spacing = float(ds_anchor.PixelSpacing[0])
                    img_height = ds_anchor.Rows
                    img_width = ds_anchor.Columns
                    img_slice = len(dicom_file_list)
                    vol_data = np.zeros((img_height, img_width, img_slice),dtype=np.int16)



                    for i, dicom_file in enumerate(dicom_file_list):
                        ds = pydicom.dcmread(dicom_file)
                        img = ds.pixel_array
                        true_height = img.shape[0]
                        true_width = img.shape[1]
                        gray_scale = True
                        if len(img.shape) != 2:
                            gray_scale = False
                        if true_height != img_height or true_width != img_width and gray_scale:
                            print('{}_{}: from dicom header:{} {} MRI data:{} {}'.format(patient_id,
                                                                                         volume_name,
                                                                                         img_height,img_width,
                                                                                         true_height,true_width))
                            img = cv2.resize(img,(img_width,img_height))
                        if not gray_scale:
                            print('Error! Not 2 channel image: {}_{}'.format(patient_id,volume_name))
                            img = img[:,:,0]
                        vol_data[:, :, i] = img
                    volume_raw = vol_data.tostring()
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'height': _int64_feature(img_height),
                                'width': _int64_feature(img_width),
                                'slice': _int64_feature(img_slice),
                                'label': _int64_feature(vol_label),
                                'volume_raw': _bytes_feature(volume_raw)
                            }
                        )
                    )
                    writer.write(example.SerializeToString())

def main_1():
    # ai_vs_radiologist
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_external'
    info_dict_file = '/home/zzhu/Data/Liver/Reader Data/LIRADSMachineLearnin_DATA_mustafa.csv'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_3d'
    output_mode = 'validation'
    col_begin = 1
    col_end = 28

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

    construct_tfrecord_3d(input_dicom_folder,series_info,output_folder,output_mode)
def main_2():
    # ai_vs_radiologist_2d no field train
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_3d'
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
    construct_tfrecord_3d(input_dicom_folder,series_info,output_folder,output_mode)

def main_3():
    # internal 3d train
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/publish_train_decompressed'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/tf_dataset/internal_3d/train.tfrecord'
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
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)
def main_4():
    # internal 3d val
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/publish_val_decompressed'
    info_dict_file = '/home/zzhu/Data/Liver/LIRADSMachineLearnin_DATA_latest.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/tf_dataset/internal_3d/validation.tfrecord'
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
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

def main_5():
    # fields 3d for test
    input_dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_external'
    info_dict_file = '/home/zzhu/Data/Liver/Reader Data/LIRADSMachineLearnin_DATA_mustafa.csv'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/tf_dataset/ai_vs_radiologist_3d_fields/validation.tfrecord'
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

    construct_tfrecord_3d_fields_shuffle(input_dicom_folder, series_info, output_file)

def main_6():
    # cv 4 folds validation

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

    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_1/val'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_1/validation.tfrecord'
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_2/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_2/validation.tfrecord'
    # construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)
    #
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_3/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_3/validation.tfrecord'
    # construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)
    #
    # input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_4/val'
    # output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_4/validation.tfrecord'
    # construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)
def main_7():
    # cv 5 3d train
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

    print('Processing Fold 1')
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_1/train'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_1/train.tfrecord'
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

    print('Processing Fold 2')
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_2/train'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_2/train.tfrecord'
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

    print('Processing Fold 3')
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_3/train'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_3/train.tfrecord'
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

    print('Processing Fold 4')
    input_dicom_folder = '/home/zzhu/Data/Liver/dataset_split_test/duke_liver_4/train'
    output_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/dataset_split_test/tfrecord_3d/cv_4/train.tfrecord'
    construct_tfrecord_3d_shuffle(input_dicom_folder, series_info, output_file)

def main_8():
    # construct mri&ct dataset for evaluation
    dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    info_dict_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/LIRADSMachineLearnin_DATA_2019-11-24_1657.csv'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

    output_file_train = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/train_3d.tfrecord'
    output_file_val = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/validation_3d.tfrecord'

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

    train_volume_list = []
    val_volume_list = []

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
                info = {}
                info['dicom_folder'] = volume_folder
                info['label'] = vol_label
                if int(patient_id) % 5 == 0:
                    val_volume_list.append(info)
                else:
                    train_volume_list.append(info)
    file_num = len(train_volume_list) + len(val_volume_list)
    print('There are in total {0} volumes.'.format(file_num))

    print('Constructing train tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=train_volume_list, output_file=output_file_train)
    print('Train 3D file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=val_volume_list, output_file=output_file_val)
    print('Val 3D file generated')

def main_9():
    '''Construct the true ct&mri dataset
    Zhe Zhu 2019/12/24'''
    dicom_folder_mri = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    dicom_folder_ct = '/mnt/sdc/Liver/CT_Data/CT_Final'
    info_dict_file = '/mnt/sdc/Liver/CT_Data/LIRADSMachineLearnin_DATA_2019-12-20_1645.csv'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

    output_file_train = '/mnt/sdc/Liver/CT_Data/tfrecord/3d/train.tfrecord'
    output_file_val = '/mnt/sdc/Liver/CT_Data/tfrecord/3d/validation.tfrecord'

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

    train_volume_list = []
    val_volume_list = []
    # process ct
    ct_patient_list = glob.glob(dicom_folder_ct + '/*')
    for ct_patient_folder in ct_patient_list:
        ct_series_folder_list = glob.glob(ct_patient_folder + '/*')
        ct_patient_id = os.path.basename(ct_patient_folder)[5:9]
        for ct_series_folder in ct_series_folder_list:
            ct_series_name = os.path.basename(ct_series_folder)
            ct_series_label = name_label_dict[ct_series_name]
            info = {}
            info['dicom_folder'] = ct_series_folder
            info['label'] = ct_series_label
            if int(ct_patient_id) % 5 == 0:
                val_volume_list.append(info)
            else:
                train_volume_list.append(info)

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
                info = {}
                info['dicom_folder'] = volume_folder
                info['label'] = vol_label
                if int(patient_id) % 5 == 0:
                    val_volume_list.append(info)
                else:
                    train_volume_list.append(info)
    file_num = len(train_volume_list) + len(val_volume_list)
    print('There are in total {0} volumes.'.format(file_num))

    print('Constructing train tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=train_volume_list, output_file=output_file_train)
    print('Train 3D file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=val_volume_list, output_file=output_file_val)
    print('Val 3D file generated')

def main_10():
    '''Construct the true ct&mri dataset
    With the update of patient 71
    Zhe Zhu 2019/12/28'''
    dicom_folder_mri = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/decompressed_dicom'
    dicom_folder_ct = '/mnt/sdc/Liver/CT_Data20191226/CT_Final'
    info_dict_file = '/mnt/sdc/Liver/CT_Data20191226/LIRADSMachineLearnin_DATA_2019-12-26_2215.csv'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

    output_file_train = '/mnt/sdc/Liver/CT_Data20191226/tfrecord/3d/train.tfrecord'
    output_file_val = '/mnt/sdc/Liver/CT_Data20191226/tfrecord/3d/validation.tfrecord'

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

    train_volume_list = []
    val_volume_list = []
    # process ct
    ct_patient_list = glob.glob(dicom_folder_ct + '/*')
    for ct_patient_folder in ct_patient_list:
        ct_series_folder_list = glob.glob(ct_patient_folder + '/*')
        ct_patient_id = os.path.basename(ct_patient_folder)[5:9]
        for ct_series_folder in ct_series_folder_list:
            ct_series_name = os.path.basename(ct_series_folder)
            ct_series_label = name_label_dict[ct_series_name]
            info = {}
            info['dicom_folder'] = ct_series_folder
            info['label'] = ct_series_label
            if int(ct_patient_id) % 5 == 0:
                val_volume_list.append(info)
            else:
                train_volume_list.append(info)

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
                info = {}
                info['dicom_folder'] = volume_folder
                info['label'] = vol_label
                if int(patient_id) % 5 == 0:
                    val_volume_list.append(info)
                else:
                    train_volume_list.append(info)
    file_num = len(train_volume_list) + len(val_volume_list)
    print('There are in total {0} volumes.'.format(file_num))

    print('Constructing train tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=train_volume_list, output_file=output_file_train)
    print('Train 3D file generated')

    print('Constructing val tfrecord file')
    construct_tfrecord_3d_mri_ct(series_info_list=val_volume_list, output_file=output_file_val)
    print('Val 3D file generated')
if __name__ == '__main__':
    # main_3()

    # main_4()
    #main_7()
    #main_8()
    #main_9()
    main_10()


