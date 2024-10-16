import os
import re
import sys
import glob
import json
import copy
import time
import random
import shutil
import pydicom
import argparse
import numpy as np
import SimpleITK as sitk
from Libs.run_mqi import run_mqi
from Libs.CT import construct_CT_object
from Libs.construct_dose_dcm_mqi import construct_dose_dcm_mqi
from Libs.construct_dict_mqi_input_parameters import construct_dict_mqi_input_parameters
from Libs.resample_and_override_CT import resample_and_override_CT, get_structures_with_override
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))


# Hardcoded parameters
raw_data_dir = '/data/sama/TEST_MOQUI/Raw_data'
reg_dir = '/data/sama/TEST_MOQUI/pat_reg'
main_path = '/data/sama/Gabriel_inf/computeDoseMoqui'
ct_output_path = '/data/sama/TEST_MOQUI/ct_output'
moqui_output_main_path = '/data/sama/TEST_MOQUI/moqui_output'
root_mqi_binary = '/data/sama/Gabriel_inf/computeDoseMoqui/Moqui'
name_mqi_binary = 'moqui'
particle_history = 10000
experiment_name = 'single_run'
target_rct = 'rCTp2'  # Specific rCT to process
trans_values = {'x': 0.2, 'y': 0.5, 'z': -0.3}  # Explicit translocation values
trans_type = {'type': 'conventional', 'range': (-1, 1)}  # Translocation type




import os
import glob

def make_data_dict(raw_data_dir, patient_id, rct_part):
    config_dict = {}

    # Initialize the ct_dirs dictionary
    config_dict[patient_id] = {'ct_dirs': {}}
    
    # Now include the pCTp0 information
    pCTp0_path = os.path.join(raw_data_dir, patient_id, 'pCTp0')
    
    if os.path.exists(pCTp0_path):
        pCT_struct_dir = glob.glob(os.path.join(pCTp0_path, "RS*.dcm"))
        
        if pCT_struct_dir:  # Check if there are any structure files
            config_dict[patient_id]['ct_dirs']['pCTp0'] = {
                'ct_dir': pCTp0_path,
                'struct_dir': pCT_struct_dir[0]  # Use the first matching structure file
            }
        else:
            print(f"No structure files found in {pCTp0_path}")
    else:
        print(f"Path does not exist: {pCTp0_path}")

    # Construct the path for the rCT
    rct_path = os.path.join(raw_data_dir, patient_id, rct_part)

    # Check if the constructed path exists
    if os.path.exists(rct_path):
        struct_dir = glob.glob(os.path.join(rct_path, "RS*.dcm"))

        if struct_dir:  # Check if there are any structure files for rCT
            dose_dir = glob.glob(os.path.join(raw_data_dir, patient_id, 'A1PHH', "RD*.dcm"))
            plan_dir = glob.glob(os.path.join(raw_data_dir, patient_id, 'A1PHH', "RP*.dcm"))
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # print(plan_dir)
            config_dict[patient_id]['ct_dirs'][rct_part] = {
                'ct_dir': rct_path,
                'struct_dir': struct_dir[0],  # Use the first matching structure file
                'dose_dir': dose_dir[0] if dose_dir else None,  # Safely access the first dose dir
                'plan_dir': plan_dir[0] if plan_dir else None  # Safely access the first plan dir
            }
        else:
            print(f"No structure files found in {rct_path}")
    else:
        print(f"{rct_path} does not exist!!!!")

    return config_dict



def add_reg_dir(config_dict, reg_dir):
    
    conf_copy = config_dict.copy()
    reg_dir_list = glob.glob(os.path.join(reg_dir, "*.json"))

    for reg_file in reg_dir_list:
        string_list = reg_file.split('/')
        pat_id = string_list[-1][:9]
        
        try:
            conf_copy[pat_id]['reg_dir'] = reg_file
        except KeyError:
            pass
    
    return conf_copy

def make_ct_obj(ct_struct_dirs, main_path):
    externalROI, overrideROIs = get_structures_with_override(main_path, ct_struct_dirs['struct_dir'])
    roi_names = [externalROI['ROIObservationLabel']] + [struct['ROIObservationLabel'] for struct in overrideROIs] + ["CTV_7000", "CTV_5425"]
    roi_names = list(set(roi_names))
    
    if 'Vullingen-implan' in roi_names:
        roi_names = roi_names.remove('Vullingen-implan')


    CT_object = construct_CT_object('CT', ct_struct_dirs['ct_dir'], ct_struct_dirs['struct_dir'], roi_names = roi_names)
    
    return CT_object, externalROI, overrideROIs 


def make_trans_vector(trans_type, trans_values=None):
    if trans_values:  # If explicit translocation values are provided, use them
        return trans_values
    else:  # Otherwise, fall back to default behavior
        if trans_type['type'] == 'conventional':
            rand_num = random.uniform(trans_type['range'][0], trans_type['range'][1])
        elif trans_type['type'] == 'discrete':
            if random.choice([True, False]): 
                rand_num = random.uniform(trans_type['range'][0], trans_type['range'][1])
            else:
                rand_num = random.uniform(-trans_type['range'][0], -trans_type['range'][1])
        return rand_num



def new_transfer_param(rig_matrix, trans_type, trans_values):
    matrix = rig_matrix.copy()
    final_translation_coordinate = {'x': 0, 'y': 0, 'z': 0}  # Initialize the dictionary

    for idx, key in zip([3, 7, 11], ['x', 'y', 'z']): 
        random_translation = make_trans_vector(trans_type, trans_values[key] if trans_values else None)
        matrix[idx] = (matrix[idx] + random_translation) * 10  # Make the new translation
        final_translation_coordinate[key] = random_translation * 10  # Store the translation

    return matrix, final_translation_coordinate

    
    
def register_ct_struct(rct, inf, CT_object, externalROI, overrideROIs, ct_out_patient_dir, final_dict, trans_type):
    ref_sitk = sitk.ReadImage(inf['dose_dir'])
    
    # Print dimensions before registration
    print(f"Before Registration: Reference image (dose) dimensions: {ref_sitk.GetSize()}")

    for roi_name, mask in CT_object.masks_structures.items():
        print(f"Before Registration: Mask '{roi_name}' dimensions: {mask.shape}")
    
    # Check if 'reg_dir' exists
    if 'reg_dir' in inf:
        with open(inf['reg_dir']) as f:
            reg_file = json.load(f)
        
        registration_matrix = np.array(reg_file['examinations'][rct]['registration_to_planning_examinations']['A1PHH']['rigid_transformation_matrix'])
        frame_of_uid = reg_file['examinations'][rct]['equipment_info']['frame_of_reference']

        new_matrix, final_translation_coordinate = new_transfer_param(registration_matrix, trans_type)
        
        final_dict[pat_id]['experiments'][rct][f'{experiment_name}']['old_reg_mat'] = registration_matrix
        final_dict[pat_id]['experiments'][rct][f'{experiment_name}']['new_reg_mat'] = new_matrix
        final_dict[pat_id]['experiments'][rct][f'{experiment_name}']['trans_coord'] = final_translation_coordinate

        # Apply transformation and resample the CT object
        CT_object.transform_and_resample(transformation_matrix=new_matrix, reference_sitk=ref_sitk, new_FoR_UID=frame_of_uid)

        # Print dimensions after transformation and resampling
        for roi_name, mask in CT_object.masks_structures.items():
            print(f"After Resampling: Mask '{roi_name}' dimensions: {mask.shape}")
    else:
        print(f"No registration file found for {rct}, skipping registration.")
        CT_object.resample(reference_sitk=ref_sitk, square_slices=False)

        # Print dimensions after resampling (without transformation)
        for roi_name, mask in CT_object.masks_structures.items():
            print(f"After Resampling: Mask '{roi_name}' dimensions: {mask.shape}")
    
    CT_object.override(externalROI, overrideROIs)

    # Print CT image dimensions after override
    print(f"After Override: CT image dimensions: {sitk.GetArrayFromImage(CT_object.image).shape}")
    
    CT_object.save(ct_out_patient_dir, save_struct_file=True)
    shutil.copy2(inf['plan_dir'], ct_out_patient_dir)

    return final_dict




def run_moqui(CT_object, ct_output_path, mq_dose_dir, particle_history, root_mqi_binary, name_mqi_binary):
    mqi_input_parameters = construct_dict_mqi_input_parameters(ct_output_path, "", 
                                                                output_dir = mq_dose_dir, 
                                                                GPUID = 0, 
                                                                particles_per_history = particle_history)

    # Debugging prints for Moqui inputs
    print("=== Moqui Run Debug Information ===")
    print(f"CT Output Path: {ct_output_path}")
    print(f"MQI Dose Directory: {mq_dose_dir}")
    print(f"MQI Input Parameters: {mqi_input_parameters}")

    try:
        run_mqi(root_mqi_binary, name_mqi_binary, mqi_input_parameters, rerun=True)
    except Exception as e:
        print(f"Error during Moqui run: {e}")


    run_mqi(root_mqi_binary, name_mqi_binary, mqi_input_parameters, rerun = True)

    filename_dose_DICOM = construct_dose_dcm_mqi(mq_dose_dir, CT_object.reference_dcm, frame_of_reference_uid = CT_object.frame_of_reference_UID)
    
def convert_ndarray_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_ndarray_to_list(i) for i in obj]
        return obj

def find_available_patients(path):
    json_files = glob.glob(f'{path}/*.json')
    return [re.split(r'[/.]', file_name)[-2] for file_name in json_files]


if __name__ == '__main__':
    config_dict = make_data_dict(raw_data_dir)

    for pat_id, inf in config_dict.items(): 
        start_time = time.time()
        final_dict = {}
        final_dict[pat_id] = inf
        final_dict[pat_id]['experiments'] = {}


        if target_rct in inf['ct_dirs']:
            ct_struct_dirs = inf['ct_dirs'][target_rct]
            final_dict[pat_id]['experiments'][target_rct] = {}

            ct_out_patient_dir = os.path.join(ct_output_path, pat_id, target_rct, experiment_name)
            os.makedirs(ct_out_patient_dir, exist_ok=True)
            final_dict[pat_id]['experiments'][target_rct][experiment_name] = {'reg_ct': ct_out_patient_dir}

            mq_dose_dir = os.path.join(moqui_output_main_path, pat_id, target_rct, experiment_name)
            os.makedirs(mq_dose_dir, exist_ok=True)
            final_dict[pat_id]['experiments'][target_rct][experiment_name]['reg_dose'] = mq_dose_dir

            CT_object, externalROI, overrideROIs = make_ct_obj(ct_struct_dirs, main_path)

            final_dict = register_ct_struct(target_rct, inf, CT_object, externalROI, overrideROIs, ct_out_patient_dir, final_dict, trans_type)

            run_moqui(CT_object, ct_out_patient_dir, mq_dose_dir, particle_history, root_mqi_binary, name_mqi_binary)


            final_dict = convert_ndarray_to_list(final_dict)
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'{pat_id}.json')
            with open(file_path, 'w') as json_file:
                json.dump(final_dict, json_file, indent=4)

        else:
            print(f"rCT '{target_rct}' not found in the configuration.")