
from rt_utils import RTStructBuilder
import numpy as np
import SimpleITK as sitk
from DBP_Dashboard.auxiliary_functions import generate_date_time_uid
import copy
import os



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


    def transform(self, transformation_matrix, reference_examination, interpolator = sitk.sitkNearestNeighbor, default_pixel_value_CT = -1000):

        transform = construct_sitk_transform_object_from_transformation_matrix(transformation_matrix)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_examination.image)
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
        self.frame_of_reference_UID = reference_examination.frame_of_reference_UID
        self.image.SetOrigin(reference_examination.image.GetOrigin())
        self.image.SetSpacing(reference_examination.image.GetSpacing())

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



    def resample(self, reference_sitk = None, new_spacing = [3,3,3], interpolator = sitk.sitkNearestNeighbor, default_pixel_value_CT = -1000, square_slices = True):

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator = interpolator
        if reference_sitk:
            resampler.SetOutputOrigin(reference_sitk.GetOrigin())
            resampler.SetOutputSpacing(reference_sitk.GetSpacing())
            new_size = list(reference_sitk.GetSize())
            if square_slices:
                if new_size[1] >= new_size[0]:
                    new_size[0] = new_size[1]
                else:
                    new_size[1] = new_size[0]
            resampler.SetSize(new_size)
        else:
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(self.image.GetOrigin())
            orig_size = np.array(self.image.GetSize(), dtype=np.int16)
            orig_spacing = self.image.GetSpacing()
            new_size = orig_size*(np.array(orig_spacing)/np.array(new_spacing))
            new_size = np.ceil(new_size).astype(np.int16) #  Image dimensions are in integers
            new_size = [int(s) for s in new_size]
            resampler.SetSize(new_size)



        resampler.SetDefaultPixelValue(0)
        for roi_name, mask in self.masks_structures.items():
            mask_image = sitk.GetImageFromArray(mask.astype(int))
            mask_image.SetOrigin(self.image.GetOrigin())
            mask_image.SetSpacing(self.image.GetSpacing())
            resampled_mask_image = resampler.Execute(mask_image)
            self.masks_structures[roi_name] = sitk.GetArrayFromImage(resampled_mask_image).astype(bool)

        resampler.SetDefaultPixelValue(default_pixel_value_CT)
        self.image = resampler.Execute(self.image)
        # self.image.SetOrigin()
        # self.image.SetSpacing(reference_examination.image.GetSpacing())


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





    def save(self, root, save_struct_file = True):
        #assert len(os.listdir(root)) <= 1
        series_instance_uid = generate_date_time_uid(self.reference_dcm.SeriesInstanceUID[:15])
        CT_array = sitk.GetArrayFromImage(self.image)
        nr_slice_idxs = CT_array.shape[0]
        origin = self.image.GetOrigin()
        spacing = self.image.GetSpacing()
        CT_array_max = CT_array.max()
        for slice_idx in range(nr_slice_idxs):
            dcm = copy.deepcopy(self.reference_dcm)
            dcm.pixel_array.dtype
            dcm.file_meta.MediaStorageSOPInstanceUID = generate_date_time_uid(dcm.file_meta.MediaStorageSOPInstanceUID[:15])
            dcm.SOPInstanceUID = dcm.file_meta.MediaStorageSOPInstanceUID
            dcm.SeriesInstanceUID = series_instance_uid
            dcm.FrameOfReferenceUID = self.frame_of_reference_UID
            dcm.InstanceNumber = nr_slice_idxs - slice_idx
            dcm.SliceLocation = origin[2] + slice_idx * spacing[2]
            dcm.PixelSpacing = [spacing[1], spacing[0]] #list(spacing[:2])
            dcm.ImagePositionPatient = list(origin[:2]) + [dcm.SliceLocation]
            dcm.ImageOrientationPatient = [int(1), int(0), int(0), int(0), int(1), int(0)]
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
                while (pixel_array.max() > 2**dcm.BitsStored - 1):
                    dcm.BitsStored = dcm.BitsStored + 1
                    dcm.HighBit = dcm.HighBit + 1
                dcm.SmallestImagePixelValue = pixel_array.min()
                dcm.LargestImagePixelValue = pixel_array.max()
            
            dcm.PixelData = pixel_array.tobytes()  
            filename = "CT" + dcm.file_meta.MediaStorageSOPInstanceUID + ".dcm"
            dcm.save_as(os.path.join(root, filename))

        if save_struct_file:
            rtstruct_new = RTStructBuilder.create_new(dicom_series_path=root)
            for roi_name in self.masks_structures.keys():
                rtstruct_new.add_roi(mask = np.moveaxis(self.masks_structures[roi_name],0,2), 
                                    name = roi_name)
                
            filename = "RS" + rtstruct_new.ds.file_meta.MediaStorageSOPInstanceUID + ".dcm"
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


