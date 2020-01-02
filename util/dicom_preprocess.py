'''
This file contains scripts to pre-process the original dicom files
Note that some scripts are copies from previous code
Organized by Zhe Zhu, 2019/12/10
'''
import os
import numpy as np
import sys
import zipfile
sys.path.append('/home/zzhu/zzhu/Liver/python')
import parse_reader_csv
import pydicom
import glob
import csv
import decompress_dicom
from shutil import copyfile

def check_data(redcap_csv,dicom_folder):
    '''
    Check if the original dicom files in a particular folder are consistent with the redcap info
    Zhe Zhu 2019/12/10
    :param redcap_csv:
    :param dicom_folder:
    :return:
    '''
def organize_dataset(input_folder,info_csv,map_txt,output_folder):
    '''
    Separate the original dicome folder into subfolders that are consistent to redcap info.
    This function should be run after the data have been checked.
    Zhe Zhu 2019/12/10
    :param input_folder:
    :param info_csv:
    :param map_txt:
    :param output_folder:
    :return:
    '''
    dt = {'names':('series_name','label'),'formats':('S20','i2')}
    name_label_list = np.loadtxt(map_txt,dtype = dt)
    name_label_dict = {}
    for i in range(len(name_label_list)):
        name_label_dict[name_label_list[i][0]] = name_label_list[i][1]
    name_label_dict['dwi_t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['t2'] = name_label_dict['dwi_and_t2']
    name_label_dict['hepatocyte'] = name_label_dict['hepa_trans']
    name_label_dict['transitional'] = name_label_dict['hepa_trans']

    count = 0
    series_info = parse_reader_csv.parse(info_csv,name_label_dict)
    for info in series_info:
        count += 1
        print count
        patient_id = info['patient_id']
        patient_folder_source = os.path.join(input_folder,'LRML_{:04d}'.format(int(patient_id)))
        patient_folder_target = os.path.join(output_folder,'LRML_{:04d}'.format(int(patient_id)))
        if not os.path.exists(patient_folder_target):
            os.makedirs(patient_folder_target)
        scan_subfolder = info['str'].strip()
        scan_folder_target = os.path.join(patient_folder_target,scan_subfolder)
        if not os.path.exists(scan_folder_target):
            os.makedirs(scan_folder_target)

        scan_type = info['type']
        if scan_type == "number":
            # Directly copy
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            dicom_file_list = glob.glob(scan_folder_source+'/*.dcm')
            for dicom_file in dicom_file_list:
                source_file = dicom_file
                target_file = os.path.join(scan_folder_target,os.path.basename(dicom_file))
                copyfile(source_file,target_file)
        elif scan_type == "slice":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source,scan_subfolder)
            slice_begin = int(info['slice_begin'])
            slice_end = int(info['slice_end'])
            for idx in range(slice_begin,slice_end+1):
                source_file = os.path.join(scan_folder_source,'{:04d}.dcm'.format(idx))
                target_file = os.path.join(scan_folder_target,'{:04d}.dcm'.format(idx))
                copyfile(source_file,target_file)
        elif scan_type == "odd":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            dicom_file_list = glob.glob(scan_folder_source+'/*.dcm')
            for dicom_file in dicom_file_list:
                digit = int(dicom_file[-5])
                if digit%2 == 1:
                    source_file = dicom_file
                    target_file = os.path.join(scan_folder_target,os.path.basename(dicom_file))
                    copyfile(source_file,target_file)
        elif scan_type == "even":
            scan_subfolder = scan_subfolder[:scan_subfolder.find('(')].strip()
            scan_folder_source = os.path.join(patient_folder_source, scan_subfolder)
            dicom_file_list = glob.glob(scan_folder_source + '/*.dcm')
            for dicom_file in dicom_file_list:
                digit = int(dicom_file[-5])
                if digit % 2 == 0:
                    source_file = dicom_file
                    target_file = os.path.join(scan_folder_target, os.path.basename(dicom_file))
                    copyfile(source_file, target_file)


def test_1():
    '''
    Organize the new CT data info the following hierachy:
    dicom_folder
    |-Patient_001
      |-series_1
      |-series_2
      ...
    |-Patient_002
    This function is the 1st step, which only extract the dicoms
    '''
    series_organization_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Series_Organization'
    patient_series_organization_tmp_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization'

    series_folder_list = glob.glob(series_organization_folder+'/*')
    for series_folder in series_folder_list:
        series_name_folder = glob.glob(series_folder+'/*')
        series_name = os.path.basename(series_name_folder[0])
        zip_file_list = glob.glob(series_name_folder[0]+'/*')
        for zip_file in zip_file_list:
            patient_id = os.path.basename(zip_file)[:9]
            target_patient_folder = os.path.join(patient_series_organization_tmp_folder,patient_id)
            if not os.path.exists(target_patient_folder):
                os.makedirs(target_patient_folder)
            target_series_folder = os.path.join(target_patient_folder,series_name)
            if not os.path.exists(target_series_folder):
                os.makedirs(target_series_folder)
            with zipfile.ZipFile(zip_file, 'r') as zip:
                zip.extractall(target_series_folder)

def test_2():
    '''
    Organize the CT data, do the step 2, following step 1.
    This script delete unnecessary folders generated in step1
    Zhe Zhu 2019/12/11
    '''
    source_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp'
    target_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp_'

    patient_folder_list = glob.glob(source_folder+'/*')

    for patient_folder in patient_folder_list:
        patient_id = os.path.basename(patient_folder)
        target_patient_folder = os.path.join(target_folder,patient_id)
        if not os.path.exists(target_patient_folder):
            os.makedirs(target_patient_folder)
        series_folder_list = glob.glob(patient_folder+'/*')
        for series_folder in series_folder_list:
            series_name = os.path.basename(series_folder)
            target_series_folder = os.path.join(target_patient_folder,series_name)
            if not os.path.exists(target_series_folder):
                os.makedirs(target_series_folder)
            for root, dirs, files in os.walk(series_folder):
                if files:
                    for file in files:
                        source_file = os.path.join(root,file)
                        target_file = os.path.join(target_series_folder,file)
                        copyfile(source_file,target_file)

def test_2_1():
    # correct the mistake, rename ct_atrerial-> ct_arterial
    folder_list = ['/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp_',
                   '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp']
    for folder in folder_list:
        patient_folder_list = glob.glob(folder+'/*')
        for patient_folder in patient_folder_list:
            series_folder_list = glob.glob(patient_folder+'/*')
            for series_folder in series_folder_list:
                series_file_name_src = os.path.basename(series_folder)
                if series_file_name_src == 'ct_atrerial':
                    os.rename(series_folder,os.path.join(patient_folder,'ct_arterial'))


def test_3():
    '''
    Organize the CT data, step 3
    Check the data, find mismatches with REDCap info table
    Zhe Zhu 2019/12/11
    '''
    info_dict_file = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/CT_included/LIRADSMachineLearnin_DATA_2019-11-24_1657.csv'
    dicome_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp_'
    col_begin = 5
    col_end = 42  # precontrast_ct has been deleted manually for consistency

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

    for info_dict in series_info:
        if int(info_dict['patient_id'])>300:
            series_name = info_dict['series_name']
            modality = series_name[:2]
            if modality == 'ct' and info_dict['series_label'] != 1: # exclude the anything else
                series_type = info_dict['type']
                patient_folder = os.path.join(dicome_folder, 'LRML_{:04d}'.format(int(info_dict['patient_id'])))
                series_folder = os.path.join(patient_folder, series_name)
                dicom_list = glob.glob(series_folder + '/*')
                if series_type == 'number':
                    # just check if has at least one dicom
                    if len(dicom_list) == 0:
                        print('!!! Empty dicom folder: '+series_folder)
                if series_type == 'even':
                    if len(dicom_list) == 0:
                        print('!!! Empty dicom folder: ' + series_folder)
                if series_type == 'odd':
                    if len(dicom_list) == 0:
                        print('!!! Empty dicom folder: ' + series_folder)
                if series_type == 'slice':
                    slice_str = info_dict['str']
                    slice_begin = int(slice_str[slice_str.find('(')+1:slice_str.find('-')])
                    slice_end = info_dict['slice_end']
                    slice_num = slice_end-slice_begin+1
                    if slice_num != len(dicom_list):
                        print('### REDCap and data mismatch: {0}, {1}'.format(slice_num,len(dicom_list))+series_folder)
def test_4():
    '''
    DEcompress all the CT series
    Zhe Zhu 1029/12/19
    :return:
    '''
    input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/Additional_CT/CT_Patient_Series_Organization_tmp_'
    output_folder = '/mnt/sdc/Liver/CT_Data/CT_Patient_Series_Decompressed'
    decompress_dicom.decompress(input_folder,output_folder)
if __name__ == '__main__':
    #test_1()
    #test_2()
    #test_2_1()
    #test_3()
    test_4()