import SimpleITK as sitk
import numpy as np
import os
import pydicom
import datetime
from pydicom.uid import UID
import random
from Libs.create_dose_dcm import create_dose_dcm


def generate_date_time_uid(prefix):
    
    uid = UID(f'{prefix}.{datetime.datetime.now():%Y%m%d%H%M%S}'
                                   f'{random.randrange(int(1e2), int(1e3))}.'
                                   f'{random.randrange(int(1e3), int(1e4))}.'
                                   f'{random.randrange(int(1e4), int(1e5))}') 
    return uid

def construct_sitk_dose_image_from_DICOM_file(path_dose_DICOM_file, nr_fractions):
    # include direction
    dose_dcm = pydicom.dcmread(path_dose_DICOM_file)

    dose_array = dose_dcm.pixel_array
    dose_grid_scaling = dose_dcm.DoseGridScaling
    dose_array = dose_array * dose_grid_scaling * nr_fractions

    dose_image = sitk.GetImageFromArray(dose_array)
    dose_image.SetOrigin(list(dose_dcm.ImagePositionPatient))
    dose_image.SetSpacing(list(dose_dcm.PixelSpacing) + [dose_dcm.SliceThickness])

    return dose_array, dose_image, dose_dcm

def get_resample_dose_image(dose_image, ref_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_dose_image =  resampler.Execute(dose_image)
    return resample_dose_image


def construct_dose_dcm_mqi(dir_mqi_doses, reference_dcm, frame_of_reference_uid = None, export_path = None, nr_fractions = 1):

    # load moqui-beam-doses (mhd) in itk
    moqui_beamdose_paths = [os.path.join(dir_mqi_doses, f) for f in os.listdir(dir_mqi_doses) if f[-4:] == '.mhd']
    moqui_beamdose_itks = [sitk.ReadImage(f) for f in moqui_beamdose_paths]
    moqui_dose = np.sum([sitk.GetArrayFromImage(i) for i in moqui_beamdose_itks], axis=0)
    moqui_dose_sitk = sitk.GetImageFromArray(moqui_dose)
    moqui_dose_sitk.SetSpacing(moqui_beamdose_itks[0].GetSpacing())
    moqui_dose_sitk.SetOrigin(np.array(moqui_beamdose_itks[0].GetOrigin()) + np.array(moqui_beamdose_itks[0].GetSpacing()) / 2)
    moqui_dose_sitk.SetDirection(moqui_beamdose_itks[0].GetDirection())
    
    # moqui_dose_dcm = create_dose_dcm2(moqui_dose_sitk, tps_dose_dcm)
    if not export_path:
        export_path = dir_mqi_doses

    filename_dose_dcm = create_dose_dcm(moqui_dose_sitk, reference_dcm, export_path, frame_of_reference_uid, nr_fractions)

    return filename_dose_dcm