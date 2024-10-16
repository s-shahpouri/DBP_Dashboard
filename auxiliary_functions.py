
import os
from Libs.Coordinate import Coordinate
# from transform import Transform
import numpy as np
import time
import re
from pydicom.uid import UID
import random
import datetime
from random import randrange
import copy
import SimpleITK as sitk

def get_unique_identifier():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    identifier = f"{timestamp}_{randrange(1000, 10000)}"
    return identifier

    
def add_folder_to_root(location_root, *folder_names):
    # update location_root to the patient and plan folder
    # updated_location_root = location_root
    
    # if not location_root.endswith('/'):
    #     updated_location_root = location_root + '/' 

    # for name in folder_names:
    #     updated_location_root += name + '/'

    updated_location_root = os.path.join(location_root, *folder_names, "")

    if not os.path.exists(updated_location_root):
        os.makedirs(updated_location_root)

    return updated_location_root

def create_path_with_folders(root, *folder_names):
    # update location_root to the patient and plan folder
    path = os.path.join(*folder_names, "")




    complete_path = os.path.join(root, path)

    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    return path








def compute_dose_on_additional_set(case, plan, beam_set, examination_name):
    beam_set.ComputeDoseOnAdditionalSets(
        OnlyOneDosePerImageSet=False, 
        AllowGridExpansion=True, 
        ExaminationNames=[examination_name], 
        FractionNumbers=[0], 
        ComputeBeamDoses=True
        )
    
    fraction_eval = [frac_eval for frac_eval
                        in case.TreatmentDelivery.FractionEvaluations
                        if frac_eval.FractionNumber == 0][0]

    dose_on_exam = [doe for doe in fraction_eval.DoseOnExaminations
                            if doe.OnExamination.Name == examination_name][0]
    
    dose = [dose_eval for dose_eval in dose_on_exam.DoseEvaluations if dose_eval.ForBeamSet.DicomPlanLabel == plan.Name][0]
    return dose



def get_roi_statistics(dose, statistic):
    
    # nr_of_fractions = dose.ForBeamSet.FractionationPattern.NumberOfFractions

    # if 'RelativeVolumeAtDose' in statistic['types']:
    #     relative_volumes = dose.GetRelativeVolumeAtDoseValues( 
    #             RoiName = statistic['contour']['name'],
    #             DoseValues = statistic['contour']['dose'] * np.array(statistic['relative_doses']) / nr_of_fractions
    #         )
    #     result = {f"v_{statistic['relative_doses'][idx]}": relative_volumes[idx] for idx in range(len(statistic['relative_doses']))}
    # else:
    #     result = {}
    # if 'DoseStatistics' in statistic['types']:
    #     for stat_type in statistic['dose_statistic_types']:
    #         result[stat_type] = dose.GetDoseStatistic( 
    #             RoiName = statistic['contour']['name'],
    #             DoseType = stat_type
    #         )
    #         result[stat_type] *= nr_of_fractions


    # return result

        
    if 'RelativeVolumeAtDose' in statistic['types']:
        relative_volumes = dose.get_relative_volumes_at_dose_values(statistic['contour']['name'], statistic['contour']['dose'] * np.array(statistic['relative_doses']))
        result = {f"v_{statistic['relative_doses'][idx]}": relative_volumes[idx] for idx in range(len(statistic['relative_doses']))}
    else:
        result = {}
    if 'DoseStatistics' in statistic['types']:
        for stat_type in statistic['dose_statistic_types']:
            result[stat_type] = dose.get_dose_statistics(statistic['contour']['name'], [stat_type])
    
    return result




def get_coordinate_from_transformation_matrix(transformation_matrix):
    # check and adjust!!!!!
    if type(transformation_matrix) == list:
        transformation_matrix = np.array(transformation_matrix)
    transform_matrix = transformation_matrix.reshape(4,4)
    # transformation_matrix_transform = Transform.from_list(transformation_matrix)
    
    x = transform_matrix[0,3]
    y = transform_matrix[1,3]
    z = transform_matrix[2,3]
    pitch = np.arcsin(transform_matrix[2,1])
    roll = -np.arcsin(transform_matrix[0,1] / np.cos(pitch))
    yaw = np.arcsin(transform_matrix[2,0] / np.cos(pitch))
    
    # x = transformation_matrix_transform.translation['x']
    # y = transformation_matrix_transform.translation['y']
    # z = transformation_matrix_transform.translation['z']
    # roll = transformation_matrix_transform.roll
    # pitch = transformation_matrix_transform.pitch
    # yaw = transformation_matrix_transform.yaw
    return Coordinate(x, y, z, np.degrees(roll), np.degrees(pitch), np.degrees(yaw))



def FoR_exists(case, from_exam_name, to_exam_name):
    exists = False
    registrations = [regist for regist in case.Registrations 
                            if regist.StructureRegistrations]
    source_registrations = [regist for regist in registrations if regist.StructureRegistrations['Source registration'].ToExamination.Name == from_exam_name]
    if len(source_registrations):
        if source_registrations[0].StructureRegistrations['Source registration'].FromExamination.Name == to_exam_name:
            exists =True
    return exists

def set_registration(case, from_examination_name, to_examination_name, registration_coordinate):
   
    case.SetFoRRegistrationRigidTransformation(
                                    FromExaminationName = from_examination_name,
                                    ToExaminationName = to_examination_name,
                                    RigidTransformation = registration_coordinate.rigid_trafo_dict()
    )
    return registration_coordinate

            


def get_registration_coordinate(case, FromExamination, ToExamination):
    transformation_matrix = case.GetTransformForExaminations(FromExamination = FromExamination, ToExamination = ToExamination)
    return get_coordinate_from_transformation_matrix(transformation_matrix)


        
def get_path_structures_DICOM_file(path_CT_DICOM_series):
    filenames = [filename for filename in os.listdir(path_CT_DICOM_series) if filename.startswith('RS')]

    assert len(filenames) > 0, "no structures DICOM file found"
    assert len(filenames) <= 1, "more than one structures DICOM files found"

    return os.path.join(path_CT_DICOM_series, filenames[0])

def get_filename_plan_DICOM_file(root_plan_DICOM_file):
    filenames = [filename for filename in os.listdir(root_plan_DICOM_file) if filename.startswith('RP')]

    assert len(filenames) > 0, "no plan DICOM file found"
    assert len(filenames) <= 1, "more than one plan DICOM files found"

    return filenames[0]

def get_path_dose_DICOM_file(root_dose_DICOM_file, startswith = 'RD'):

    filenames = [filename for filename in os.listdir(root_dose_DICOM_file) if filename.startswith(startswith)]

    assert len(filenames) > 0, "no dose DICOM file found"
    assert len(filenames) <= 1, "more than one dose DICOM files found"

    return os.path.join(root_dose_DICOM_file, filenames[0])





def generate_date_time_uid(prefix):
    
    uid = UID(f'{prefix}.{datetime.datetime.now():%Y%m%d%H%M%S}'
                                   f'{random.randrange(int(1e2), int(1e3))}.'
                                   f'{random.randrange(int(1e3), int(1e4))}.'
                                   f'{random.randrange(int(1e4), int(1e5))}') 
    return uid
    

def set_isocenter_plan_dcm(plan_dcm, isocenter_position):
    for beam in plan_dcm.IonBeamSequence:
        beam.IonControlPointSequence[0].IsocenterPosition = isocenter_position



def resample_CT(CT_object, path_reference_dcm = None, spacing = [3,3,3]):
    CT_resampled = copy.deepcopy(CT_object)
    if path_reference_dcm:
        reference_sitk = sitk.ReadImage(path_reference_dcm)
        CT_resampled.resample(reference_sitk = reference_sitk, new_spacing = spacing, default_pixel_value_CT =-1000)
    else:
       CT_resampled.resample(new_spacing = spacing, default_pixel_value_CT = -1024)

    return CT_resampled
    
