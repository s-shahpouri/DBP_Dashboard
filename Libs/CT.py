
from rt_utils import RTStructBuilder
import numpy as np
import SimpleITK as sitk
from auxiliary_functions import generate_date_time_uid
import copy
import os
import shutil
from rt_utils import RTStructBuilder
import os
import numpy as np
import pydicom
import glob

def get_rotation_matrix(theta_pitch, theta_yaw, theta_roll):
    roll = np.array([[np.cos(theta_roll), -np.sin(theta_roll), 0],
                      [np.sin(theta_roll), np.cos(theta_roll), 0],
                      [0, 0, 1]])
    yaw = np.array([[np.cos(theta_yaw), 0, -np.sin(theta_yaw)],
                      [0, 1, 0],
                      [np.sin(theta_yaw), 0, np.cos(theta_yaw)]])
    pitch = np.array([[1, 0, 0],
                      [0, np.cos(theta_pitch), -np.sin(theta_pitch)],
                      [0, np.sin(theta_pitch), np.cos(theta_pitch)]])
    rotation_matrix = np.matmul(roll, np.matmul(pitch, yaw))

    return rotation_matrix


def construct_sitk_transform_object_from_transformation_matrix(transformation_matrix):
    # note: when applying the transform sitk object to a CT somewhere the inverse needs to be taken

    assert transformation_matrix.shape == (4,4), "transformation matrix is not 4 by 4"





    theta_pitch = np.arcsin(transformation_matrix[2,1])
    theta_roll = -np.arcsin(transformation_matrix[0,1] / np.cos(theta_pitch))
    theta_yaw = np.arcsin(transformation_matrix[2,0] / np.cos(theta_pitch))



    rotation_matrix = get_rotation_matrix(theta_pitch, theta_yaw, theta_roll)


    transform = sitk.VersorRigid3DTransform()
    transform.SetTranslation(transformation_matrix[:3, 3])
    transform.SetMatrix(rotation_matrix.reshape(9))

    return transform


class CT(object):
    
    def __init__(self,
                 CT_name,
                 CT_image,
                 masks_structures,
                 reference_dcm,
                 struct_dcm = None,
                 frame_of_reference_UID = None,
                 KVP = None):
        self.name = CT_name
        self.image = CT_image
        self.masks_structures = masks_structures
        self.reference_dcm = reference_dcm
        self.struct_dcm = struct_dcm
        self.frame_of_reference_UID = frame_of_reference_UID
        self.KVP = KVP


    # def transform_and_resample(self, transformation_matrix, reference_sitk, new_FoR_UID, interpolator=sitk.sitkNearestNeighbor, default_pixel_value_CT=-1000):
    #     # Print dimensions before transformation and resampling
    #     print(f"Before Resample: CT image dimensions: {sitk.GetArrayFromImage(self.image).shape}")
        
    #     transform = construct_sitk_transform_object_from_transformation_matrix(transformation_matrix)
    #     resampler = sitk.ResampleImageFilter()
    #     resampler.SetReferenceImage(reference_sitk)
    #     resampler.SetInterpolator(interpolator)
    #     resampler.SetTransform(transform.GetInverse())
        
    #     # Resample CT image
    #     resampler.SetDefaultPixelValue(default_pixel_value_CT)
    #     self.image = resampler.Execute(self.image)

    #     # Resample each mask in the RTStruct (stored in masks_structures)
    #     for roi_name, mask in self.masks_structures.items():
    #         mask_image = sitk.GetImageFromArray(mask.astype(int))
    #         mask_image.SetOrigin(self.image.GetOrigin())
    #         mask_image.SetSpacing(self.image.GetSpacing())
    #         resampled_mask_image = resampler.Execute(mask_image)
    #         self.masks_structures[roi_name] = sitk.GetArrayFromImage(resampled_mask_image).astype(bool)

    #     self.frame_of_reference_UID = new_FoR_UID
    #     self.image.SetOrigin(reference_sitk.GetOrigin())
    #     self.image.SetSpacing(reference_sitk.GetSpacing())
        
    #     # Print dimensions after resampling
    #     print(f"After Resample: CT image dimensions: {sitk.GetArrayFromImage(self.image).shape}")
    #     for roi_name, mask in self.masks_structures.items():
    #         print(f"After Resample: Mask '{roi_name}' dimensions: {mask.shape}")


    def transform_and_resample(self, transformation_matrix, reference_sitk, new_FoR_UID, interpolator = sitk.sitkNearestNeighbor, default_pixel_value_CT = -1000):

        transform = construct_sitk_transform_object_from_transformation_matrix(transformation_matrix)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_sitk)
        resampler.SetInterpolator(interpolator)
        resampler.SetTransform(transform.GetInverse())

        resampler.SetDefaultPixelValue(0)
        for roi_name, mask in self.masks_structures.items():
            mask_image = sitk.GetImageFromArray(mask.astype(int))
            mask_image.SetOrigin(self.image.GetOrigin())
            mask_image.SetSpacing(self.image.GetSpacing())
            resampled_mask_image = resampler.Execute(mask_image)
            self.masks_structures[roi_name] = sitk.GetArrayFromImage(resampled_mask_image).astype(bool)

        resampler.SetDefaultPixelValue(default_pixel_value_CT)
        self.image = resampler.Execute(self.image)
        self.frame_of_reference_UID = new_FoR_UID
        self.image.SetOrigin(reference_sitk.GetOrigin())
        self.image.SetSpacing(reference_sitk.GetSpacing())

        
    def resample(self, reference_sitk=None, new_spacing=[3, 3, 3], interpolator=sitk.sitkNearestNeighbor, default_pixel_value_CT=-1000, square_slices=True):
        # Print dimensions before resampling
        print(f"Before Resample: CT image dimensions: {sitk.GetArrayFromImage(self.image).shape}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        if reference_sitk:
            resampler.SetOutputOrigin(reference_sitk.GetOrigin())
            resampler.SetOutputSpacing(reference_sitk.GetSpacing())
            new_size = list(reference_sitk.GetSize())
            if square_slices:
                new_size[0] = max(new_size[0], new_size[1])
                new_size[1] = max(new_size[0], new_size[1])
            resampler.SetSize(new_size)
        else:
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(self.image.GetOrigin())
            orig_size = np.array(self.image.GetSize(), dtype=np.int16)
            orig_spacing = self.image.GetSpacing()
            new_size = orig_size * (np.array(orig_spacing) / np.array(new_spacing))
            new_size = np.ceil(new_size).astype(np.int16)
            resampler.SetSize([int(s) for s in new_size])

        resampler.SetDefaultPixelValue(0)
        for roi_name, mask in self.masks_structures.items():
            mask_image = sitk.GetImageFromArray(mask.astype(int))
            mask_image.SetOrigin(self.image.GetOrigin())
            mask_image.SetSpacing(self.image.GetSpacing())
            resampled_mask_image = resampler.Execute(mask_image)
            self.masks_structures[roi_name] = sitk.GetArrayFromImage(resampled_mask_image).astype(bool)

        resampler.SetDefaultPixelValue(default_pixel_value_CT)
        self.image = resampler.Execute(self.image)

        # Print dimensions after resampling
        print(f"After Resample: CT image dimensions: {sitk.GetArrayFromImage(self.image).shape}")
        for roi_name, mask in self.masks_structures.items():
            print(f"After Resample: Mask '{roi_name}' dimensions: {mask.shape}")


    def override(self, externalROI, overrideROIs):
       

        desired_masks = [['CTV_7000', 'CTV_5425', 'External']]
        
        new_CT_array = sitk.GetArrayFromImage(self.image)
        external = self.masks_structures[externalROI['ROIObservationLabel']]
        new_CT_array[external == False] = externalROI['HU']


        print("overridesROIs: ", overrideROIs)
        for override in overrideROIs:

            if override in desired_masks: # Change this part to limit the overrides and masks into CTVs and external
                mask = self.masks_structures[override["ROIObservationLabel"]]
                new_CT_array[mask] = override['HU']



        new_CT_image = sitk.GetImageFromArray(new_CT_array)

        new_CT_image.SetOrigin(self.image.GetOrigin())
        new_CT_image.SetSpacing(self.image.GetSpacing())
        self.image = new_CT_image



        
    def override_air_outside_external(self,  name_external = 'External'):

        external = self.masks_structures[name_external]
        new_CT_array = sitk.GetArrayFromImage(self.image)
        new_CT_array[external == False] = -1000
        new_CT_image = sitk.GetImageFromArray(new_CT_array)

        new_CT_image.SetOrigin(self.image.GetOrigin())
        new_CT_image.SetSpacing(self.image.GetSpacing())

        self.image = new_CT_image




    def save(self, root, save_struct_file=True):
        # Clear the folder where resampled series are saved to avoid accumulation
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)

        series_instance_uid = generate_date_time_uid(self.reference_dcm.SeriesInstanceUID[:15])
        CT_array = sitk.GetArrayFromImage(self.image)
        nr_slice_idxs = CT_array.shape[0]
        origin = self.image.GetOrigin()
        spacing = self.image.GetSpacing()
        CT_array_max = CT_array.max()

        # Debugging: Print the shape of the CT_array
        print(f"CT_array shape: {CT_array.shape}")

        # Save each slice of the CT image
        for slice_idx in range(nr_slice_idxs):
            dcm = copy.deepcopy(self.reference_dcm)
            dcm.file_meta.MediaStorageSOPInstanceUID = generate_date_time_uid(dcm.file_meta.MediaStorageSOPInstanceUID[:15])
            dcm.SOPInstanceUID = dcm.file_meta.MediaStorageSOPInstanceUID
            dcm.SeriesInstanceUID = series_instance_uid
            dcm.FrameOfReferenceUID = self.frame_of_reference_UID
            dcm.InstanceNumber = nr_slice_idxs - slice_idx
            dcm.SliceLocation = origin[2] + slice_idx * spacing[2]
            dcm.PixelSpacing = [spacing[1], spacing[0]]
            dcm.ImagePositionPatient = list(origin[:2]) + [dcm.SliceLocation]
            dcm.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            dcm.Rows = CT_array.shape[1]
            dcm.Columns = CT_array.shape[2]
            dcm.SliceThickness = spacing[2]

            if dcm.file_meta.TransferSyntaxUID == "1.2.840.10008.1.2.1":
                rescale_intercept = -1000
                rescale_slope = (CT_array_max - rescale_intercept) / (2**16 - 1)
                dcm.RescaleSlope = rescale_slope
                dcm.RescaleIntercept = rescale_intercept
                CT_array[slice_idx][np.where(CT_array[slice_idx] < rescale_intercept)] = rescale_intercept
                pixel_array = ((CT_array[slice_idx] - dcm.RescaleIntercept) / dcm.RescaleSlope).astype(self.reference_dcm.pixel_array.dtype)
            else:
                pixel_array = ((CT_array[slice_idx] - dcm.RescaleIntercept) / dcm.RescaleSlope).astype(self.reference_dcm.pixel_array.dtype)
                while pixel_array.max() > 2**dcm.BitsStored - 1:
                    dcm.BitsStored += 1
                    dcm.HighBit += 1
                dcm.SmallestImagePixelValue = pixel_array.min()
                dcm.LargestImagePixelValue = pixel_array.max()

            dcm.PixelData = pixel_array.tobytes()
            filename = f"CT{dcm.file_meta.MediaStorageSOPInstanceUID}.dcm"
            dcm.save_as(os.path.join(root, filename))

        # Saving RTStruct if necessary
        if save_struct_file:
            # Initialize the RTStructBuilder with the resampled DICOM series
            rtstruct_new = RTStructBuilder.create_new(dicom_series_path=root)

            # Debugging: Print the expected number of DICOM slices in the resampled series
            dicom_files = glob.glob(os.path.join(root, "*.dcm"))
            print(f"Saving RTStruct: Expected number of slices in reference_dcm: {len(dicom_files)}")
            
            # Add ROIs to the RTStruct
            for roi_name in self.masks_structures.keys():
                mask = np.moveaxis(self.masks_structures[roi_name], 0, 2)
                # Debugging: Print mask and slice info
                print(f"Adding ROI '{roi_name}' with mask shape: {mask.shape}")
                
                # Ensure that the mask has the correct number of slices
                if mask.shape[2] != len(dicom_files):
                    raise ValueError(f"Mask for '{roi_name}' has incorrect number of slices. Expected {len(dicom_files)}, but got {mask.shape[2]}")
                
                rtstruct_new.add_roi(mask=mask, name=roi_name)

            filename = f"RS{rtstruct_new.ds.file_meta.MediaStorageSOPInstanceUID}.dcm"
            rtstruct_new.save(os.path.join(root, filename))
    



def construct_sitk_image_from_CT_dicom_series(CT_dicom_series):
    # include direction!

    ref_ds = CT_dicom_series[0] # series_data is ordered according to patient position
    CT_array_dimensions = (len(CT_dicom_series), ref_ds.Rows, ref_ds.Columns)
    CT_array = np.zeros(CT_array_dimensions, dtype = float)
    for idx, slice in enumerate(CT_dicom_series):
        CT_array[idx ,:, :] = slice.RescaleIntercept + slice.pixel_array * slice.RescaleSlope

    CT_image = sitk.GetImageFromArray(CT_array)
    CT_image.SetOrigin(list(ref_ds.ImagePositionPatient))
    CT_image.SetSpacing(list(ref_ds.PixelSpacing) + [ref_ds.SliceThickness])

    frame_of_reference_UID = ref_ds.FrameOfReferenceUID
    KVP = ref_ds.get_item('KVP').value

    return (CT_image, frame_of_reference_UID, KVP)

def construct_CT_object(CT_name, path_CT_DICOM_series, path_structures_DICOM_file, roi_names = []):
    rtstruct = RTStructBuilder.create_from(dicom_series_path = path_CT_DICOM_series, rt_struct_path = path_structures_DICOM_file)
    CT_image, frame_of_reference_UID, KVP = construct_sitk_image_from_CT_dicom_series(rtstruct.series_data)

    masks_structures = {}
    if not roi_names:
        for roi_name in rtstruct.get_roi_names():
            try:
                masks_structures[roi_name] = np.moveaxis(rtstruct.get_roi_mask_by_name(roi_name),2, 0)
            except:
                pass
    else:
        desired_rois = ['CTV_5425', 'CTV_7000', 'External']
        for roi_name in roi_names:
            if roi_name in desired_rois:
                masks_structures[roi_name] = np.moveaxis(rtstruct.get_roi_mask_by_name(roi_name),2, 0)


    reference_dcm = rtstruct.series_data[0]
    struct_dcm = rtstruct.ds

    return CT(CT_name, CT_image, masks_structures, reference_dcm, struct_dcm, frame_of_reference_UID, KVP)


