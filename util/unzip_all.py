import os
import glob
import zipfile
import subprocess
import sys
import numpy as np
import csv

import tensorflow as tf
import pydicom
import cv2

import decompress_dicom

from tensorflow.python.tools import freeze_graph,optimize_for_inference_lib
from shutil import copyfile

sys.path.append('/home/zzhu/zzhu/Liver/python')
sys.path.append('/home/zzhu/Data/Liver/code/python/src')
import parse_reader_csv as parse

import inception_2d

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def dataset_3d_decode(sequence_proto):
    feature_description = {
        'height': tf.FixedLenFeature([],tf.int64,0),
        'width': tf.FixedLenFeature([],tf.int64,0),
        'slice': tf.FixedLenFeature([],tf.int64,0),
        'label': tf.FixedLenFeature([],tf.int64,0),
        'series_str': tf.FixedLenFeature([],tf.string,''),
        'patient_id': tf.FixedLenFeature([],tf.string,''),
        'volume_raw': tf.FixedLenFeature([],tf.string,'')
    }
    sequence = tf.parse_single_example(sequence_proto,feature_description)
    height = sequence['height']
    width = sequence['width']
    label = sequence['label']
    slice = sequence['slice']

    patient_id = sequence['patient_id']
    series_str = sequence['series_str']

    volume_raw = sequence['volume_raw']
    volume = tf.decode_raw(volume_raw, tf.uint8)
    volume = tf.cast(volume, tf.float32)

    dim = tf.stack([height, width, slice])
    volume = tf.reshape(volume, dim)
    volume = tf.image.resize_images(volume, (256, 256))
    volume = tf.random_crop(volume, (tf.constant(224), tf.constant(224), tf.cast(slice, tf.int32)))

    return volume,label,patient_id,series_str

def model_fn(features,labels,mode,params):
    global_step = tf.train.get_global_step()
    input_layer = features['feature']
    logits, end_points= inception_2d.inception_v1_classifier1_caffe(input_layer,
                                                     num_classes=30,
                                                     spatial_squeeze=True,
                                                     dropout_keep_prob=1.0)

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
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9)
        optimizer = tf.train.AdagradOptimizer(0.001)
        #optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=0.001,weight_decay=0.0002,momentum=0.9)
        train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op,training_hooks=train_hook_list)
    # compute evaluation metrics
    eval_metric_ops = {
        'accuracy':tf.metrics.accuracy(labels=labels,predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


def unzip_all(zip_folder,output_folder):
    zip_file_list = glob.glob(zip_folder+'/*')
    for zip_file in zip_file_list:
        with zipfile.ZipFile(zip_file,'r') as zip:
            zip.extractall(output_folder)
    print('Finished')



def decompress(input_folder,output_folder):
    patient_folder_list = glob.glob(input_folder + '/*')
    for patient_folder in patient_folder_list:
        patient_folder_name = os.path.basename(patient_folder)
        output_patient_folder = os.path.join(output_folder, patient_folder_name)
        if not os.path.exists(output_patient_folder):
            os.makedirs(output_patient_folder)
        sub_folder_list = glob.glob(patient_folder + '/*')
        sub_folder = sub_folder_list[0]
        subsub_folder_list = glob.glob(sub_folder+'/*')
        subsub_folder = subsub_folder_list[0]
        sequence_folder_list = glob.glob(subsub_folder + '/*')
        print('Processing {0}'.format(patient_folder_name))
        for sequence_folder in sequence_folder_list:
            sequence_folder_name = os.path.basename(sequence_folder)
            output_sequence_folder = os.path.join(output_patient_folder, sequence_folder_name)
            if not os.path.exists(output_sequence_folder):
                os.makedirs(output_sequence_folder)
            dicom_file_list = glob.glob(sequence_folder + '/*')
            for dicom_file in dicom_file_list:
                dicom_file_name = os.path.basename(dicom_file)
                output_dicom_file = os.path.join(output_sequence_folder, dicom_file_name)
                subprocess.check_output(['gdcmconv', '-w', dicom_file, output_dicom_file])

def load_map_dict(info_dict_file,col_begin,col_end):
    map_txt = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/map.txt' # This is fixed
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

    return name_label_dict

def load_map_dict_v2(info_dict_file,col_begin,col_end):
    '''
    34 series types in total, including all the ct series
    Zhe Zhu, 2019/12/20
    '''
    map_txt = '/mnt/sdc/Liver/CT_Data/map_ct_mri.txt'  # This is fixed
    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['mri_dwi_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_t2'] = name_label_dict['mri_dwi_and_t2']
    name_label_dict['mri_hepatocyte'] = name_label_dict['mri_hepa_trans']
    name_label_dict['mri_transitional'] = name_label_dict['mri_hepa_trans']

    # ct series
    name_label_dict['ct_arterial_early'] = name_label_dict['ct_arterial']
    name_label_dict['ct_arterial_late'] = name_label_dict['ct_arterial']


    # deal with anything else classes
    with open(info_dict_file, 'r') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        header = next(rows, None)
        for i in range(col_begin, col_end):
            series_name = header[i]
            series_name_ct = 'ct_' + series_name
            series_name_mri = 'mri_' + series_name
            if series_name_ct not in name_label_dict:
                name_label_dict[series_name_ct] = name_label_dict['anythingelse']
            if series_name_mri not in name_label_dict:
                name_label_dict[series_name_mri] = name_label_dict['anythingelse']

    return name_label_dict

def compare_two_readings(reading1_file,reading2_file,map_dict):
    print('Comparing {0} and {1}'.format(reading1_file,reading2_file))
    reading1 = parse.parse(reading1_file,map_dict)
    reading2 = parse.parse(reading2_file,map_dict)

    rs1 = set()
    for r in reading1:
        if r['series_id'] in rs1:
            print('Repeated item:{0}'.format(r))
        rs1.add(r['series_id'])
    rs2 = set()
    for r in reading2:
        if r['series_id'] in rs2:
            print ('Repeated item:{0}'.format(r))
        rs2.add(r['series_id'])

    # Check repeated items first
    if len(reading1) != len(rs1):
        print('reading 1 may contain repeated items')
        print('from list:{0}'.format(len(reading1)))
    if len(reading2) != len(rs2):
        print('reading 2 may contain repeated items')
        print('from list:{0}'.format(len(reading2)))

    print('{0} reading contains {1} items'.format(os.path.basename(reading1_file),len(rs1)))
    print('{0} reading contains {1} items'.format(os.path.basename(reading2_file),len(rs2)))
    diff1 = rs1-rs2
    if diff1:
        print('In {0} but not in {1}:'.format(os.path.basename(reading1_file),os.path.basename(reading2_file)))
        for item in diff1:
            print(item)
    diff2 = rs2-rs1
    if diff2:
        print('In {0} but not in {1}:'.format(os.path.basename(reading2_file),os.path.basename(reading1_file)))
        for item in diff2:
            print(item)

def dicom2png(dicom_folder,png_folder):
    # extract slices as png and rename the series by series number
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)
    patient_folder_list = glob.glob(dicom_folder+'/*')
    for patient_folder in patient_folder_list:
        series_folder_list = glob.glob(patient_folder+'/*')
        patient_id = os.path.basename(patient_folder)
        png_patient_folder = os.path.join(png_folder,patient_id)
        if not os.path.exists(png_patient_folder):
            os.makedirs(png_patient_folder)
        for series_folder in series_folder_list:
            series_folder_name = os.path.basename(series_folder)
            series_num_str = series_folder_name[series_folder_name.rfind('-')+1:].strip()
            png_series_folder = os.path.join(png_patient_folder,series_num_str)
            if not os.path.exists(png_series_folder):
                os.makedirs(png_series_folder)
            dicom_file_list = glob.glob(series_folder+'/*.dcm')
            for dicom_file in dicom_file_list:
                ds = pydicom.dcmread(dicom_file)
                img = ds.pixel_array
                img = img.astype(np.float32)
                img = (img-np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0
                img = img.astype(np.uint8)
                img_name = os.path.basename(dicom_file)[:-3]+'png'
                img_output_path = os.path.join(png_series_folder,img_name)
                cv2.imwrite(img_output_path,img)

def separate_series(input_folder,output_folder):
    # use mustafa's file as reference
    map_dict = load_map_dict('/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',3,29)
    reading = parse.parse('/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',map_dict)
    count = 0
    for info in reading:
        count += 1
        if count%25==0:
            print(count)
        patient_id = info['patient_id']
        patient_folder_source = os.path.join(input_folder, 'LRML_{:04d}'.format(int(patient_id)))
        patient_folder_target = os.path.join(output_folder, 'LRML_{:04d}'.format(int(patient_id)))
        if not os.path.exists(patient_folder_target):
            os.makedirs(patient_folder_target)
        scan_subfolder = info['str'].strip()
        scan_folder_target = os.path.join(patient_folder_target, scan_subfolder)
        if not os.path.exists(scan_folder_target):
            os.makedirs(scan_folder_target)

        scan_type = info['type']
        if scan_type == "number":
            # Directly copy
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.png')
            for img_file in img_file_list:
                source_file = img_file
                target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                copyfile(source_file, target_file)
        elif scan_type == "slice":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            slice_begin = int(info['slice_begin'])
            slice_end = int(info['slice_end'])
            for idx in range(slice_begin, slice_end + 1):
                source_file = os.path.join(scan_folder_source, 'IM-0001-{:04d}-0001.png'.format(idx))
                target_file = os.path.join(scan_folder_target, 'IM-0001-{:04d}-0001.png'.format(idx))
                copyfile(source_file, target_file)
        elif scan_type == "odd":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.png')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 1:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)
        elif scan_type == "even":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.png')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 0:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)

def inference_volume(tfrecord_file,model_file):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(dataset_3d_decode)
    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    reading_dict = {}
    from tensorflow.contrib import predictor
    predict_fn = predictor.from_saved_model(model_file)
    total = 0
    correct = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            while (True):
                volume,label,patient_id,series_str = sess.run(next_elem)
                slice = volume.shape[2]
                for i in range(slice):
                    img = volume[:,:,i]
                    img = (img - np.amin(img)+1e-8)/(np.amax(img)-np.amin(img)+1e-8)*255.0 - 37
                    volume[:,:,i] = img
                print('Processing {0}...\n'.format(total))
                volume = np.transpose(volume,(2,0,1))
                volume = np.expand_dims(volume,3)
                pred = predict_fn({'image': volume})
                scores = pred['prob']
                score = np.mean(scores,axis=0)
                pred_label = np.argmax(score)
                if pred_label == label:
                    correct += 1
                total += 1
                if patient_id not in reading_dict:
                    reading_dict[patient_id] = {}
                reading_dict[patient_id][series_str] = pred_label

        except Exception as e:
            print(e)
    print('Total: {0}'.format(total))
    print('Correct: {0}'.format(correct))
    return reading_dict

def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    image = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,1], name='image')
    receiver_tensors = {'image': image}
    return tf.estimator.export.ServingInputReceiver(image, receiver_tensors)

def write_csv(reading_list, file_name):
    # file_names = ['subject_id',	'mrn', 'scan_date', 'locs', 'precontrast', 'arterial_single', 'portalvenous','latedynamic',
    #               'coronallatedynamic',	'transitional',	'hepatocyte', 'hepatocyte_coronal',	'coronalt2', 't2', 'mrcp','ssfp',
    #               'ssfp_coronal', 'opposedphase', 'in_phase', 'opposedphase_coronal', 'inphase_coronal', 'fat',	'preconstrast_coronal',
    #               'dwi_t2',	'dwi',	'adc',	'dwi_coronal_t2',	'dwi_coronal',	'adc_coronal',	'notes', 'series_type_abstraction_tool_complete',
    #               'age','sex',	'fieldstrength',	'manufacturer',	'model','name_facility', 'exam_and_subject_info_complete']
    file_names = ['subject_id', 'mrn', 'scan_date', 'locs', 'precontrast', 'arterial_single', 'portalvenous',
                  'latedynamic',
                  'coronallatedynamic', 'transitional', 'hepatocyte', 'hepatocyte_coronal', 'coronalt2', 'dwi_and_t2', 'mrcp',
                  'ssfp',
                  'ssfp_coronal', 'opposedphase', 'in_phase', 'opposedphase_coronal', 'inphase_coronal', 'fat',
                  'preconstrast_coronal',
                  'dwi_t2','dwi', 'adc', 'dwi_coronal_t2', 'dwi_coronal', 'adc_coronal', 'notes',
                  'series_type_abstraction_tool_complete',
                  'age', 'sex', 'fieldstrength', 'manufacturer', 'model', 'name_facility',
                  'exam_and_subject_info_complete']
    with open(file_name,mode='w') as csv_file:
        writer = csv.DictWriter(csv_file,fieldnames=file_names)
        writer.writeheader()
        for reading in reading_list:
            writer.writerow(reading)

def compare_readings(reading_gt_file, reading_pred_file):
    map_dict = load_map_dict('/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',
                             3, 29)
    reading_gt = parse.parse(reading_gt_file,
                          map_dict)
    reading_pred = parse.parse(reading_pred_file,
                          map_dict)
    count = 0
    correct = 0
    print('Comparing {0} and {1}'.format(os.path.basename(reading_gt_file), os.path.basename(reading_pred_file)))

    reading_map_gt = {}
    for read in reading_gt:
        reading_map_gt[read['patient_id'].strip() + read['str'].strip()] = read
    reading_map_pred = {}
    for read in reading_pred:
        reading_map_pred[read['patient_id'].strip() + read['str'].strip()] = read

    for key in reading_map_gt:
        pred_label = reading_map_pred[key]['series_label']
        gt_label = reading_map_gt[key]['series_label']
        if gt_label==pred_label:
            correct += 1
        count += 1

    print('Total:{0} Correct:{1} Acc:{2}'.format(count,correct,float(correct)/float(count)))

def test_1():
    zip_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/50_new_cases'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/data'
    unzip_all(zip_folder=zip_folder,output_folder=output_folder)

def test_2():
    # decompress all the sequences
    input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/data'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/data_decompressed'
    decompress(input_folder,output_folder)

def test_3():
    # pair-wise comparison, to check the splitting is correct
    info_dict_file = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv'
    col_begin = 3
    col_end = 29
    map_dict = load_map_dict(info_dict_file,col_begin,col_end)

    reading_file_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_brian.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_chad.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_erin.csv'
    ]
    # check if the script can read the readings

    compare_two_readings(reading_file_list[0],reading_file_list[1],map_dict)
    compare_two_readings(reading_file_list[0], reading_file_list[2], map_dict)
    compare_two_readings(reading_file_list[0], reading_file_list[3], map_dict)
    compare_two_readings(reading_file_list[1], reading_file_list[2], map_dict)
    compare_two_readings(reading_file_list[1], reading_file_list[3], map_dict)
    compare_two_readings(reading_file_list[2], reading_file_list[3], map_dict)

def test_4():
    # cvt dicom to png and rename the series folder
    dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/data_decompressed'
    png_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/png'
    dicom2png(dicom_folder,png_folder)

def test_5():
    # separate the series
    input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/png'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/png_series'
    separate_series(input_folder,output_folder)

def test_6():
    # construct dataset. Each instance is a 3D volume.
    reference_file = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv'
    patient_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/png_series'
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/tfrecord/external_validation.tfrecord'

    map_dict = load_map_dict('/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',
                             3, 29)
    reading = parse.parse('/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv',
                          map_dict)
    reading_map = {}
    for read in reading:
        reading_map[read['patient_id']+read['str'].strip()] = read
    patient_folder_list = glob.glob(patient_folder+'/*')
    series_num = 0
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for patient_folder in patient_folder_list:
            patient_id = os.path.basename(patient_folder)[-3:]
            series_folder_list = glob.glob(patient_folder+'/*')
            for series_folder in series_folder_list:
                folder_name = os.path.basename(series_folder).strip()
                read = reading_map[patient_id+folder_name]
                series_str = read['str']
                patient_id = read['patient_id']
                series_label = read['series_label']
                slice_file_list = glob.glob(series_folder+'/*')
                #print(slice_file_list[0])
                if len(slice_file_list) == 0:
                    print(series_folder)
                series_num += 1
                if series_num%50 == 0:
                    print(series_num)
                anchor_slice_file = slice_file_list[0]
                anchor_slice = cv2.imread(anchor_slice_file,0)
                slice_height = anchor_slice.shape[0]
                slice_width = anchor_slice.shape[1]
                slice_depth = len(slice_file_list)
                volume = np.zeros((slice_height,slice_width,slice_depth),dtype=np.uint8)
                slice_idx = 0
                for slice_file in slice_file_list:
                    slice = cv2.imread(slice_file,0)
                    if slice.shape[0] != volume.shape[0]:
                        print(slice_file_list[0])
                        slice = cv2.resize(slice,(volume.shape[1],volume.shape[0]))
                    volume[:,:,slice_idx] = slice
                    slice_idx += 1
                volume_raw = volume.tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'height': _int64_feature(slice_height),
                            'width': _int64_feature(slice_width),
                            'slice': _int64_feature(slice_depth),
                            'label': _int64_feature(series_label),
                            'series_str': _bytes_feature(series_str),
                            'patient_id': _bytes_feature(patient_id),
                            'volume_raw': _bytes_feature(volume_raw)
                        }
                    )
                )
                writer.write(example.SerializeToString())
    print('In total {0} series'.format(series_num))
def test_7():
    # correct 645
    input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/ori'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/decompressed_645'
    decompress(input_folder, output_folder)

def test_8():
    dicom_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/decompressed_645'
    png_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/png'
    dicom2png(dicom_folder, png_folder)

def test_9():
    input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/png'
    output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/Correct_645/png_series'
    separate_series(input_folder, output_folder)

def test_10():
    # freeze the pre-trained model
    location = freeze_graph.freeze_graph(input_graph='/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception/graph.pbtxt',
                                         input_saver='',
                                         input_binary=False,
                                         input_checkpoint='/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception/model.ckpt-50000',
                                         output_node_names='InceptionV1_Classifier1/Logits/Predictions/Softmax',
                                         restore_op_name='save/restore_all',
                                         filename_tensor_name='save/Const:0',
                                         output_graph='/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception/freezed/frozentensorflowModel.pb',
                                         clear_devices=True,
                                         initializer_nodes='')
    print('Freezed graph save in {0}'.format(location))

def test_10_1():
    # Exporting the estimator as a tf.saved_model
    estimator = tf.estimator.Estimator(model_fn, '/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception', params={})
    estimator.export_savedmodel('/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception/freezed/saved_model', serving_input_receiver_fn)


def test_11():
    # inference
    tfrecord_file = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/tfrecord/external_validation.tfrecord'
    model_file = '/media/zzhu/Seagate Backup Plus Drive/data/Run/liver/trained_models/2d_inception/freezed/saved_model/1567801772'
    reading_dict = inference_volume(tfrecord_file,model_file)
    np.save('/media/zzhu/Seagate Backup Plus Drive/data/Amber/ai.npy',reading_dict)

def test_12():
    # file_names = ['subject_id', 'mrn', 'scan_date', 'locs', 'precontrast', 'arterial_single', 'portalvenous',
    #               'latedynamic',
    #               'coronallatedynamic', 'transitional', 'hepatocyte', 'hepatocyte_coronal', 'coronalt2', 't2', 'mrcp',
    #               'ssfp',
    #               'ssfp_coronal', 'opposedphase', 'in_phase', 'opposedphase_coronal', 'inphase_coronal', 'fat',
    #               'preconstrast_coronal',
    #               'dwi_t2', 'dwi', 'adc', 'dwi_coronal_t2', 'dwi_coronal', 'adc_coronal', 'notes',
    #               'series_type_abstraction_tool_complete',
    #               'age', 'sex', 'fieldstrength', 'manufacturer', 'model', 'name_facility',
    #               'exam_and_subject_info_complete']
    file_names = ['subject_id', 'mrn', 'scan_date', 'locs', 'precontrast', 'arterial_single', 'portalvenous',
                  'latedynamic',
                  'coronallatedynamic', 'transitional', 'hepatocyte', 'hepatocyte_coronal', 'coronalt2', 'dwi_and_t2', 'mrcp',
                  'ssfp',
                  'ssfp_coronal', 'opposedphase', 'in_phase', 'opposedphase_coronal', 'inphase_coronal', 'fat',
                  'preconstrast_coronal',
                  'dwi_t2','dwi', 'adc', 'dwi_coronal_t2', 'dwi_coronal', 'adc_coronal', 'notes',
                  'series_type_abstraction_tool_complete',
                  'age', 'sex', 'fieldstrength', 'manufacturer', 'model', 'name_facility',
                  'exam_and_subject_info_complete']
    old_new_map = {'hepa_trans':'transitional',
                   'arterial_early':'arterial_single',
                   'arterial':'arterial_single',
                   'arterial_late':'arterial_single'}
    # load label2series list
    map_txt = '/media/zzhu/Seagate Backup Plus Drive/data/Amber/map.txt'  # This is fixed
    dt = {'names': ('series_name', 'label'), 'formats': ('S20', 'i2')}
    name_label_list = np.loadtxt(map_txt, dtype=dt)
    label_name_list = [None]*30
    for i in range(len(name_label_list)):
        label_name_list[name_label_list[i][1]] = name_label_list[i][0]
    # save to csv format
    reading_dict = np.load('/media/zzhu/Seagate Backup Plus Drive/data/Amber/ai.npy')
    reading_list = [None]*50
    for i in range(50):
        reading_list[i] = {}
        reading_list[i]['subject_id'] = str(i+601)
    reading_dict = dict(np.ndenumerate(reading_dict))[()]
    for key in reading_dict:
        idx = int(key)-601
        sub_reading = reading_dict[key]
        reading_list[idx]['notes'] = ''
        for skey in sub_reading:
            reading_label = sub_reading[skey]
            reading_name = label_name_list[reading_label]
            if reading_name in old_new_map:
                reading_name = old_new_map[reading_name] # replace with new one
            if reading_name not in file_names:
                if reading_list[idx]['notes'] == '':
                    reading_list[idx]['notes'] = skey+':'+reading_name
                else:
                    reading_list[idx]['notes'] = reading_list[idx]['notes'] + ',' + skey+':'+reading_name
            else:
                if reading_name not in reading_list[idx]:
                    reading_list[idx][reading_name] = skey
                else:
                    reading_list[idx][reading_name] = reading_list[idx][reading_name] + ',' + skey
    write_csv(reading_list,'/media/zzhu/Seagate Backup Plus Drive/data/Amber/ai.csv')

def test_13():
    # compare readings
    reading_file_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Amber/ai.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_brian.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_chad.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_erin.csv',
                         '/media/zzhu/Seagate Backup Plus Drive/data/Amber/LIRADSMachineLearnin_DATA_mustafa.csv']
    for i in range(len(reading_file_list)):
        for j in range(len(reading_file_list)):
            if i > j:
                compare_readings(reading_file_list[i],reading_file_list[j])

def test_14():
    '''Rename the 1-150 CT series into:
    Folder
    --Patient_XXXX
      --4
      --3
    ...
    Zhe Zhu 2019/12/23
    '''
    input_folder = '/home/zzhu/Data/Liver/LRML'
    output_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step1'

    ct_patient_num_set = set(['0006','0037','0038','0041','0046','0060','0062','0071','0077','0078',
                              '0079','0080','0085','0087','0090','0091','0094','0095','0098','0103',
                              '0107','0115','0118','0120','0121','0127','0128','0130','0133','0134',
                              '0136','0138','0143','0144','0147','0148','0149','0150'])

    patient_list = glob.glob(input_folder+'/*')
    for patient_folder in patient_list:
        patient_id = os.path.basename(patient_folder)
        patient_num_4digit = patient_id[5:9]
        if patient_num_4digit not in ct_patient_num_set:
            continue
        patient_folder_tgt = os.path.join(output_folder,patient_id[:9])
        if not os.path.exists(patient_folder_tgt):
            os.makedirs(patient_folder_tgt)
        sub_folder_list = glob.glob(patient_folder+'/*')
        sub_folder = sub_folder_list[0]
        series_folder_list = glob.glob(sub_folder+'/*')
        for series_folder in series_folder_list:
            dicom_file_list = glob.glob(series_folder+'/*')
            if len(dicom_file_list) == 0:
                print('{0} empty.'.format(series_folder))
                continue
            series_num_idx = series_folder.rfind('- ')
            series_num = series_folder[series_num_idx+2:]
            series_folder_tgt = os.path.join(patient_folder_tgt,series_num)
            if not os.path.exists(series_folder_tgt):
                os.makedirs(series_folder_tgt)

            for dicom_file in dicom_file_list:
                source_file = dicom_file
                dicome_file_name = os.path.basename(source_file)
                target_file = os.path.join(series_folder_tgt,dicome_file_name)
                copyfile(source_file,target_file)


def test_15():
    '''Decompress the ct series of patient1-150
    Zhe Zhu 2019/12/23
    '''
    input_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step1'
    output_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step2'
    decompress_dicom.decompress_v1(input_folder, output_folder)

def test_16():
    '''
    Rename CT patient 1-150
    Zhe Zhu, 2019.12.20
    :return:
    '''
    input_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step2'
    output_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step3'
    redcap_file = '/mnt/sdc/Liver/CT_Data/LIRADSMachineLearnin_DATA_2019-12-20_1645.csv'

    col_begin = 5
    col_end = 42
    map_dict = load_map_dict_v2(redcap_file,
                             col_begin, col_end)
    reading = parse.parse_v2(redcap_file,
                          map_dict)
    count = 0
    for info in reading:
        if info['series_name'][0] == 'm':
            continue
        if int(info['patient_id']) > 150:
            continue
        count += 1
        if count % 25 == 0:
            print(count)
        patient_id = info['patient_id']
        patient_folder_source = os.path.join(input_folder, 'LRML_{:04d}'.format(int(patient_id)))
        patient_folder_target = os.path.join(output_folder, 'LRML_{:04d}'.format(int(patient_id)))
        if not os.path.exists(patient_folder_target):
            os.makedirs(patient_folder_target)
        scan_subfolder = info['str'].strip()
        scan_folder_target = os.path.join(patient_folder_target, scan_subfolder)
        if not os.path.exists(scan_folder_target):
            os.makedirs(scan_folder_target)

        scan_type = info['type']
        if scan_type == "number":
            # Directly copy
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                source_file = img_file
                target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                copyfile(source_file, target_file)
        elif scan_type == "slice":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            slice_begin = int(info['slice_begin'])
            slice_end = int(info['slice_end'])
            for idx in range(slice_begin, slice_end + 1):
                source_file = os.path.join(scan_folder_source, 'IM-0001-{:04d}-0001.dcm'.format(idx))
                target_file = os.path.join(scan_folder_target, 'IM-0001-{:04d}-0001.dcm'.format(idx))
                copyfile(source_file, target_file)
        elif scan_type == "odd":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 1:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)
        elif scan_type == "even":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 0:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)
def test_17():
    '''Convert the data to ct_arterial, ct_pre, etc format
    Zhe Zhu, 2019/12/23'''
    input_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step3'
    output_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_1_150_step4'
    col_begin = 5
    col_end = 42
    valid_series_set = set(['ct_precontrast','ct_portalvenous',
                            'ct_latedynamic','ct_arterial_early',
                            'ct_arterial_late','ct_arterial'])
    redcap_file = '/mnt/sdc/Liver/CT_Data/LIRADSMachineLearnin_DATA_2019-12-20_1645.csv'
    map_dict = load_map_dict_v2(redcap_file,
                                col_begin, col_end)
    reading = parse.parse_v2(redcap_file,
                             map_dict)
    count = 0
    for info in reading:
        if info['series_name'][0] == 'm':
            continue
        if int(info['patient_id']) > 150:
            continue
        count += 1
        if count % 25 == 0:
            print(count)
        series_name = info['series_name']
        source_dicom_folder = os.path.join(input_folder, 'LRML_{:04d}'.format(int(info['patient_id'])), info['str'])
        target_patient_folder = os.path.join(output_folder, 'LRML_{:04d}'.format(int(info['patient_id'])))
        if not os.path.exists(target_patient_folder):
            os.makedirs(target_patient_folder)
        if series_name in valid_series_set:
            if series_name == 'ct_arterial_early' or series_name == 'ct_arterial_late':
                target_series_folder = os.path.join(target_patient_folder, 'ct_arterial')
            else:
                target_series_folder = os.path.join(target_patient_folder, series_name)
            if not os.path.exists(target_series_folder):
                os.makedirs(target_series_folder)
            dicom_file_list = glob.glob(source_dicom_folder+'/*')
            for dicome_file in dicom_file_list:
                dicom_file_name = os.path.basename(dicome_file)
                target_dicom_file_name = dicom_file_name[:-4]+'_'+info['series_id']+'.dcm'
                target_dicom_file = os.path.join(target_series_folder, target_dicom_file_name)
                copyfile(dicome_file, target_dicom_file)

def test_18():
    '''Rename the 1-150 CT series into:
    Folder
    --Patient_XXXX
      --4
      --3
    ...
    With the updated data of patient 71
    Zhe Zhu 2019/12/26
    '''
    input_folder = '/home/zzhu/Data/Liver/LRML'
    output_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step1'

    ct_patient_num_set = set(['0006','0037','0038','0041','0046','0060','0062','0071','0077','0078',
                              '0079','0080','0085','0087','0090','0091','0094','0095','0098','0103',
                              '0107','0115','0118','0120','0121','0127','0128','0130','0133','0134',
                              '0136','0138','0143','0144','0147','0148','0149','0150'])

    patient_list = glob.glob(input_folder+'/*')
    for patient_folder in patient_list:
        patient_id = os.path.basename(patient_folder)
        patient_num_4digit = patient_id[5:9]
        if patient_num_4digit not in ct_patient_num_set:
            continue
        patient_folder_tgt = os.path.join(output_folder,patient_id[:9])
        if not os.path.exists(patient_folder_tgt):
            os.makedirs(patient_folder_tgt)
        sub_folder_list = glob.glob(patient_folder+'/*')
        sub_folder = sub_folder_list[0]
        series_folder_list = glob.glob(sub_folder+'/*')
        for series_folder in series_folder_list:
            dicom_file_list = glob.glob(series_folder+'/*')
            if len(dicom_file_list) == 0:
                print('{0} empty.'.format(series_folder))
                continue
            series_num_idx = series_folder.rfind('- ')
            series_num = series_folder[series_num_idx+2:]
            series_folder_tgt = os.path.join(patient_folder_tgt,series_num)
            if not os.path.exists(series_folder_tgt):
                os.makedirs(series_folder_tgt)

            for dicom_file in dicom_file_list:
                source_file = dicom_file
                dicome_file_name = os.path.basename(source_file)
                target_file = os.path.join(series_folder_tgt,dicome_file_name)
                copyfile(source_file,target_file)

def test_19():
    '''Decompress the ct series of patient1-150
    With the patient 71 data updated
    Zhe Zhu 2019/12/26
    '''
    input_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step1'
    output_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step2'
    decompress_dicom.decompress_v1(input_folder, output_folder)

def test_20():
    '''
    Rename CT patient 1-150
    With the updated data of patient 71
    Zhe Zhu, 2019.12.27
    '''
    input_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step2'
    output_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step3'
    redcap_file = '/mnt/sdc/Liver/CT_Data20191226/LIRADSMachineLearnin_DATA_2019-12-26_2215.csv'

    col_begin = 5
    col_end = 42
    map_dict = load_map_dict_v2(redcap_file,
                                col_begin, col_end)
    reading = parse.parse_v2(redcap_file,
                             map_dict)
    count = 0
    for info in reading:
        if info['series_name'][0] == 'm':
            continue
        if int(info['patient_id']) > 150:
            continue
        count += 1
        if count % 25 == 0:
            print(count)
        patient_id = info['patient_id']
        patient_folder_source = os.path.join(input_folder, 'LRML_{:04d}'.format(int(patient_id)))
        patient_folder_target = os.path.join(output_folder, 'LRML_{:04d}'.format(int(patient_id)))
        if not os.path.exists(patient_folder_target):
            os.makedirs(patient_folder_target)
        scan_subfolder = info['str'].strip()
        scan_folder_target = os.path.join(patient_folder_target, scan_subfolder)
        if not os.path.exists(scan_folder_target):
            os.makedirs(scan_folder_target)

        scan_type = info['type']
        if scan_type == "number":
            # Directly copy
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                source_file = img_file
                target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                copyfile(source_file, target_file)
        elif scan_type == "slice":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            slice_begin = int(info['slice_begin'])
            slice_end = int(info['slice_end'])
            for idx in range(slice_begin, slice_end + 1):
                source_file = os.path.join(scan_folder_source, 'IM-0001-{:04d}-0001.dcm'.format(idx))
                target_file = os.path.join(scan_folder_target, 'IM-0001-{:04d}-0001.dcm'.format(idx))
                copyfile(source_file, target_file)
        elif scan_type == "odd":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 1:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)
        elif scan_type == "even":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            img_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for img_file in img_file_list:
                digit = int(img_file[-10])
                if digit % 2 == 0:
                    source_file = img_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(img_file))
                    copyfile(source_file, target_file)
def test_21():
    '''Convert the data to ct_arterial, ct_pre, etc format
    With the updated data of patient 71
    Zhe Zhu, 2019/12/27'''
    input_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step3'
    output_folder = '/mnt/sdc/Liver/CT_Data20191226/CT_Patient_1_150_step4'
    col_begin = 5
    col_end = 42
    valid_series_set = set(['ct_precontrast','ct_portalvenous',
                            'ct_latedynamic','ct_arterial_early',
                            'ct_arterial_late','ct_arterial'])
    redcap_file = '/mnt/sdc/Liver/CT_Data20191226/LIRADSMachineLearnin_DATA_2019-12-26_2215.csv'
    map_dict = load_map_dict_v2(redcap_file,
                                col_begin, col_end)
    reading = parse.parse_v2(redcap_file,
                             map_dict)
    count = 0
    for info in reading:
        if info['series_name'][0] == 'm':
            continue
        if int(info['patient_id']) > 150:
            continue
        count += 1
        if count % 25 == 0:
            print(count)
        series_name = info['series_name']
        source_dicom_folder = os.path.join(input_folder, 'LRML_{:04d}'.format(int(info['patient_id'])), info['str'])
        target_patient_folder = os.path.join(output_folder, 'LRML_{:04d}'.format(int(info['patient_id'])))
        if not os.path.exists(target_patient_folder):
            os.makedirs(target_patient_folder)
        if series_name in valid_series_set:
            if series_name == 'ct_arterial_early' or series_name == 'ct_arterial_late':
                target_series_folder = os.path.join(target_patient_folder, 'ct_arterial')
            else:
                target_series_folder = os.path.join(target_patient_folder, series_name)
            if not os.path.exists(target_series_folder):
                os.makedirs(target_series_folder)
            dicom_file_list = glob.glob(source_dicom_folder+'/*')
            for dicome_file in dicom_file_list:
                dicom_file_name = os.path.basename(dicome_file)
                target_dicom_file_name = dicom_file_name[:-4]+'_'+info['series_id']+'.dcm'
                target_dicom_file = os.path.join(target_series_folder, target_dicom_file_name)
                copyfile(dicome_file, target_dicom_file)
if __name__ == "__main__":
    #test_1()
    #test_2()
    #test_3()
    #test_4()
    #test_5()
    #test_6()
    #test_7()
    #test_8()
    #test_9()
    #test_10()
    #test_10_1()
    #test_11()
    #test_12()
    #test_13()
    #test_14()
    #test_15()
    #test_16()
    #test_17()
    #test_18()
    #test_19()
    #test_20()
    test_21()