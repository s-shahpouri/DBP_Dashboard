
import os
import sys
import json
import copy
import time
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


def construct_config_file(root):
    print("root", root)
    config = {}
    config['outputPath'] = root + "output/"
    os.makedirs(config['outputPath'], exist_ok=True)
    config['dataPath'] = root 
    config['plan'] = root + [filename for filename in os.listdir(root) if filename.startswith("RP")][0]
    config['struct'] = root + [filename for filename in os.listdir(root) if filename.startswith("RS")][0]
    config['CT'] = root + "CT/"
    config['referenceDose'] = root + [filename for filename in os.listdir(root) if filename.startswith("RD")][0]
    config['reg'] = root + [filename for filename in os.listdir(root) if filename.startswith("REG")][0]

    config['particlesPerHistory'] = 10000

    with open(root + "config.json", 'w') as f:
        json.dump(config, f)

##################################################################


def read_config(args):

    with open(args.configfilepath) as f:
        config = json.load(f)

    path_input_data = os.path.join(config['dataPath'], "input_data/")
    os.makedirs(path_input_data, exist_ok=True)

    os.makedirs(config['outputPath'], exist_ok=True)
    
    return config, path_input_data


def make_ct_obj(config):
    externalROI, overrideROIs = get_structures_with_override(os.path.dirname(os.path.realpath(__file__)), config['struct'])
    print(config['struct'])
    roi_names = [externalROI['ROIObservationLabel']] + [struct['ROIObservationLabel'] for struct in overrideROIs] + ["CTV_7000", "CTV_5425"]
    roi_names = list(set(roi_names))
    CT_object = construct_CT_object('CT', config['CT'], config['struct'], roi_names = roi_names)
    
    return CT_object, externalROI, overrideROIs 


def register_ct_struct(config, CT_object, externalROI, overrideROIs):

    ref_sitk = sitk.ReadImage(config['referenceDose'])
    if 'reg' in config.keys():
        
        with open(config['reg']) as f:
            reg_file = json.load(f)
        
        registration_matrix = np.array(reg_file['examinations']['rCTp3']['registration_to_planning_examinations']['A1PHH']['rigid_transformation_matrix']).reshape(4, 4)
        frame_of_uid = reg_file['examinations']['rCTp3']['equipment_info']['frame_of_reference']

        registration_matrix[0,3], registration_matrix[1,3], registration_matrix[2,3] = 10 * registration_matrix[0,3], 10 * registration_matrix[1,3], 10 * registration_matrix[2,3]

        CT_object.transform_and_resample(transformation_matrix = registration_matrix, reference_sitk = ref_sitk, new_FoR_UID = frame_of_uid)#, square_slices = True)
    else:
        CT_object.resample(reference_sitk = ref_sitk, square_slices=False)#, square_slices = True)

    CT_object.override(externalROI, overrideROIs)

    CT_object.save(path_input_data, save_struct_file=False)


    shutil.copy2(config['plan'], path_input_data)


def run_moqui(args, path_input_data, config):
    mqi_input_parameters = construct_dict_mqi_input_parameters(path_input_data, "", 
                                                                output_dir = config['outputPath'], 
                                                                GPUID = 0, 
                                                                particles_per_history = config['particlesPerHistory'])


    run_mqi(args.root_mqi_binary, args.name_mqi_binary, mqi_input_parameters, rerun = True)

    filename_dose_DICOM = construct_dose_dcm_mqi(config['outputPath'], CT_object.reference_dcm, frame_of_reference_uid = CT_object.frame_of_reference_UID)



if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Computes dose with moqui")
    parser.add_argument('configfilepath')
    # parser.add_argument('--root', default=None)
                        # default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--root_mqi_binary', nargs='?',
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "Moqui"))
    parser.add_argument('--name_mqi_binary', nargs='?', default="moqui")

    args = parser.parse_args()    
    
    # Make the config file
    config, path_input_data = read_config(args)
    
    # Make CT object
    CT_object, externalROI, overrideROIs = make_ct_obj(config)
    
    # Register CT
    register_ct_struct(config, CT_object, externalROI, overrideROIs)
    
    # Run moqui
    run_moqui(args, path_input_data, config)
    
    print("dose calc finished: ", time.time() - start_time)
    
    
    
    