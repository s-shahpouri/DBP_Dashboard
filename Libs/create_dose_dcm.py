from pydicom.uid import UID, generate_uid
from pydicom.dataset import FileDataset, ImplicitVRLittleEndian, FileMetaDataset
import datetime
import numpy as np
import SimpleITK as sitk
import os
from Libs.generate_date_time_uid import generate_date_time_uid






patient_name = ""
patient_id = ""
patient_date_of_birth = ""





def create_dose_dcm(moqui_dose_sitk, reference_dcm, export_path, frame_of_reference_uid = None, nr_fractions = 1):


    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.481.2')  # RT Dose Storage
    # Needs valid UID

    prefix_uid = "1.2.752.243.1.1"

    file_meta.MediaStorageSOPInstanceUID = generate_date_time_uid(prefix_uid)
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = ''
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.PrivateInformationCreatorUID = file_meta.ImplementationClassUID
    file_meta.PrivateInformation = b"pydicomScripts"
    file_meta.TransferSyntaxUID = ImplicitVRLittleEndian  # Implicit VR Little Endian Transfer Syntax(
    # 1.2.840.10008.1.2)and Explicit VR Little Endian Transfer Syntax (1.2.840.10008.1.2.1)

    # create DICOM RT-Dose object.





    rtdose = FileDataset(None, {}, file_meta=file_meta, preamble=b'\0' * 128)

    # No DICOM object standard. Use only required to avoid errors with viewers.
    rtdose.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    rtdose.SOPClassUID = file_meta.MediaStorageSOPClassUID

    rtdose.PatientName = reference_dcm.PatientName # getDicomTag(0x010, 0x010)["Patient's Name"]
    rtdose.PatientID = reference_dcm.PatientID # getDicomTag(0x010, 0x020)["Patient ID"]
    rtdose.PatientBirthDate = reference_dcm.PatientBirthDate # str(getDicomTag(0x010, 0x030)["Patient's Birth Date"]).replace("-", "")
    rtdose.PatientSex = reference_dcm.PatientSex



    rtdose.SeriesDate = f"{datetime.datetime.now():%Y%m%d}"
    rtdose.SeriesTime = f"{datetime.datetime.now():%H%M%S.%f}"
    rtdose.StudyDate = ""
    rtdose.StudyTime = ""
    rtdose.ContentDate = rtdose.SeriesDate
    rtdose.ContentTime = rtdose.SeriesTime
    rtdose.InstanceCreationDate = rtdose.SeriesDate
    rtdose.InstanceCreationTime = rtdose.SeriesTime
    rtdose.SpecificCharacterSet = 'ISO_IR 100'
    rtdose.AccessionNumber = ""
    rtdose.OperatorsName = ""
    rtdose.Manufacturer = "UMCG"
    rtdose.ManufacturerModelName = "PythonScriptsMoquiDose"
    rtdose.SoftwareVersions = "Evaluation Dose"

    rtdose.StudyInstanceUID = reference_dcm.StudyInstanceUID
    rtdose.SeriesInstanceUID = generate_date_time_uid(prefix_uid)
    rtdose.SeriesDescription = ""
        

    rtdose.StudyID = ""
    rtdose.SeriesNumber = 1 # getDicomTag(0x020, 0x011)["Series Number"]
    rtdose.InstanceNumber = 1 #getDicomTag(0x020, 0x013)["Instance Number"]
    rtdose.Modality = 'RTDOSE'

    moqui_dose_array = sitk.GetArrayFromImage(moqui_dose_sitk)
    scaling = moqui_dose_array.max() / (2**16 - 1)
    moqui_dose_scaled = (moqui_dose_array / scaling).astype(np.uint16) 




    rtdose.DoseGridScaling = scaling * nr_fractions

    # copy the gamma matrix on top of the dose map
    # np.copyto(rtdose.pixel_array, moqui_dose_scaled)
    rtdose.PixelData = moqui_dose_scaled.tobytes()
    rtdose.ImagePositionPatient = list(moqui_dose_sitk.GetOrigin())
    spacing  = moqui_dose_sitk.GetSpacing()
    size = moqui_dose_sitk.GetSize()
    rtdose.PixelSpacing = [spacing[1], spacing[0]]     #????????????????????
    rtdose.SliceThickness = spacing[2]
    if frame_of_reference_uid:
        rtdose.FrameOfReferenceUID = frame_of_reference_uid
    else:
        rtdose.FrameOfReferenceUID = reference_dcm.FrameOfReferenceUID
    rtdose.Rows = moqui_dose_array.shape[1]
    rtdose.Columns = moqui_dose_array.shape[2]
    rtdose.NumberOfFrames = moqui_dose_array.shape[0]
    rtdose.GridFrameOffsetVector = list(np.arange(0, moqui_dose_array.shape[0] * spacing[2], spacing[2]).astype(int))
    rtdose.ImageOrientationPatient = list(moqui_dose_sitk.GetDirection())
    rtdose.PositionReferenceIndicator = ""

    rtdose.SamplesPerPixel = 1 
    rtdose.PhotometricInterpretation = "MONOCHROME2"
    rtdose.BitsAllocated = 16  
    rtdose.BitsStored = 16  
    rtdose.HighBit = 15  
    rtdose.PixelRepresentation = 0 
    rtdose.DoseUnits = 'GY'
    rtdose.DoseType = 'EFFECTIVE'
    rtdose.DoseSummationType = 'EVALUATION'
    rtdose.TissueHeterogeneityCorrection = ["IMAGE"]  # , "ROI_OVERRID"]

    filename_rtdose = 'RD-moqui.dcm'



    # finally, save the DICOM object
    rtdose.save_as(os.path.join(export_path, filename_rtdose))

    return filename_rtdose

    








    



