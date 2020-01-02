# Batch decompress dicoms
import subprocess
import os
import glob
import sys

def decompress(input_folder,output_folder):
    patient_folder_list = glob.glob(input_folder + '/*')
    for patient_folder in patient_folder_list:
        patient_folder_name = os.path.basename(patient_folder)
        output_patient_folder = os.path.join(output_folder, patient_folder_name)
        if not os.path.exists(output_patient_folder):
            os.makedirs(output_patient_folder)
        sequence_folder_list = glob.glob(patient_folder + '/*')
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

def decompress_v1(input_folder,output_folder):
    '''If target file exists, skip
    Zhe Zhu, 2019/12/23'''
    patient_folder_list = glob.glob(input_folder + '/*')
    for patient_folder in patient_folder_list:
        patient_folder_name = os.path.basename(patient_folder)
        output_patient_folder = os.path.join(output_folder, patient_folder_name)
        if not os.path.exists(output_patient_folder):
            os.makedirs(output_patient_folder)
        sequence_folder_list = glob.glob(patient_folder + '/*')
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
                if not os.path.exists(output_dicom_file):
                    subprocess.check_output(['gdcmconv', '-w', dicom_file, output_dicom_file])
if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

# input_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/data_to_publish'
# output_folder = '/media/zzhu/Seagate Backup Plus Drive/data/Liver/duke_liverset/publish_all_decompressed'


