import os
import numpy as np
import glob
import logging
import pydicom

def check_dataset(folder,log_path):
    logging.basicConfig(filename=log_path,level=logging.INFO)
    exam_folder_list = glob.glob(folder+'/*')

    scanning_sequence_list = []
    repetition_time_list = []
    echo_time_list = []
    echo_train_length_list = []
    flip_angle_list = []

    for exam_folder in exam_folder_list:
        sequence_folder_list = glob.glob(exam_folder+'/*')
        for sequence_folder in sequence_folder_list:
            dicom_list = glob.glob(sequence_folder+'/*')
            for dicom_file in dicom_list:
                ds = pydicom.dcmread(dicom_file)
                if ds.dir('ScanningSequence'):
                    scanning_sequence = ds.ScanningSequence
                    type_ = type(scanning_sequence)
                    if not isinstance(scanning_sequence,str):
                        logging.info(dicom_file+' '+scanning_sequence)
                        if isinstance(scanning_sequence,list):
                            for sub_ss in scanning_sequence:
                                if isinstance(sub_ss,str):
                                    scanning_sequence_list.append(sub_ss)
                                else:
                                    logging.error('Unkown Type'+sub_ss)
                        else:
                            logging.error('Unkown Type')
                    else:
                        scanning_sequence_list.append(scanning_sequence)
                else:
                    logging.error(dicom_file+' no scanning sequence')

                if ds.dir('RepetitionTime'):
                    repetition_time = ds.RepetitionTime
                    repetition_time_list.append(repetition_time)
                else:
                    logging.error(dicom_file+' no repetition time')

                if ds.dir('EchoTime'):
                    echo_time = ds.EchoTime
                    echo_time_list.append(echo_time)
                else:
                    logging.error(dicom_file+' no echo time')

                if ds.dir('EchoTrainLength'):
                    echo_train_length = ds.EchoTrainLength
                    echo_train_length_list.append(echo_train_length)
                else:
                    logging.error(dicom_file+' no echo train length')

                if ds.dir('FlipAngle'):
                    flip_angle = ds.FlipAngle
                    flip_angle_list.append(flip_angle)
                else:
                    logging.error(dicom_file+' no flip angle')





if __name__ == '__main__':
    dataset_folder = '/home/zzhu/Data/Liver/cvpr2019/liver_dataset_dicom'
    log_path = '/home/zzhu/Data/Liver/log/0301.log'
    check_dataset(dataset_folder,log_path)

