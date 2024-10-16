from Libs.CT import construct_CT_object
from scipy import interpolate
import copy
import pydicom
import SimpleITK as sitk
import re
import numpy as np
import os

def get_structures_with_override(scriptdir, structFileName):

    # open density CT conversion file - USED FOR MATERIALS NOT FOUND IN OVERRIDE
    CTdensity = [[], []]
    CTdensityFileName =  os.path.join(scriptdir, 'Scanners/Groningen/HU_Density_Conversion.txt')
    with open(CTdensityFileName) as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            if l[0] != '#':
                data = re.split('\t\t|\t| ', l.strip())
                if data[-1] != 'OVERRIDE':
                    CTdensity[0].append(float(data[1]))
                    CTdensity[1].append(float(data[0]))

    # open material conversion file - USED FOR MATERIALS FOUND IN OVERRIDE
    CTmaterial = dict()
    CTmaterialFileName = os.path.join(scriptdir, 'Scanners/Groningen/HU_Material_Conversion.txt')
    with open(CTmaterialFileName) as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            if l[0] != '#':
                data = re.split('\t\t|\t| ', l.strip())
                if data[-1] == 'OVERRIDE':
                    CTmaterial[data[1]] = data[0]

    # open material list
    materialList = dict()
    materialListFileName = os.path.join(scriptdir, 'Materials/list.dat')
    with open(materialListFileName) as f:
        lines = f.read().strip().split('\n')
        for l in lines:
            if len(l) and l[0] != '#':
                data = re.split('\t\t|\t| ', l.strip())
                materialList[data[1]] = data[0]

    print(CTdensity, CTmaterial, materialList)

    # read the structure file
    dcm = pydicom.read_file(structFileName)

    #
    # search for external and override ROIs
    #

    # create a list of override ROIs
    overrideROIs = []
    externalROI = {}

    # for each roi
    ROIs = dcm.RTROIObservationsSequence # This tag finds all the available ROIs
    for roi in ROIs:
        # there is only one EXTERNAL, so load after finding it
        if roi.RTROIInterpretedType == 'EXTERNAL': # This label find the type of the ROI
            newOverride = {}
            newOverride['ROIObservationLabel'] = roi.ROIObservationLabel
            newOverride['REL_MASS_DENSITY'] = 0.0012

            externalROI = newOverride

        # try to find a material ID for this structure, this marks a override
        try:
            roi.MaterialID

            # if exist, then proceed to save it in the override ROIs list (These ROIs can be fillings etc.)
            newOverride = {}
            newOverride['MaterialID'] = roi.MaterialID # Is a number 
            newOverride['ROIObservationLabel'] = roi.ROIObservationLabel # If there is a MaterialID, it should be sth that we want to get rid of.
            for prop in roi.ROIPhysicalPropertiesSequence: # It iterates through physical properties of the ROI that we want to override like dental filling
                if prop.ROIPhysicalProperty == 'REL_MASS_DENSITY':
                    newOverride['REL_MASS_DENSITY'] = float(prop.ROIPhysicalPropertyValue) # Save the value of the physical properties
                    break

            overrideROIs.append(newOverride)
        except:
            pass

    #
    # convert from REL_DENSITY to HU
    #

    CTdensityInterpolation = interpolate.interp1d(CTdensity[0], CTdensity[1])

    if externalROI:
        externalROI['HU'] = str(CTdensityInterpolation(externalROI['REL_MASS_DENSITY']))

    for i in range(len(overrideROIs)):
        # if the material is found in the mcsquare material list and
        # is found in the material override list, then use it for override.
        if overrideROIs[i]['MaterialID'] in materialList and materialList[overrideROIs[i]['MaterialID']] in CTmaterial:
            print(overrideROIs[i]['MaterialID'], materialList[overrideROIs[i]['MaterialID']], CTmaterial[materialList[overrideROIs[i]['MaterialID']]])
            overrideROIs[i]['HU'] = CTmaterial[materialList[overrideROIs[i]['MaterialID']]]
            
        # otherwise, continue to use the density override only
        else:
            if overrideROIs[i]['REL_MASS_DENSITY'] > CTdensity[0][-1]:
                overrideROIs[i]['REL_MASS_DENSITY'] = CTdensity[0][-1]
            if overrideROIs[i]['REL_MASS_DENSITY'] < CTdensity[0][0]:
                overrideROIs[i]['REL_MASS_DENSITY'] = CTdensity[0][0]
            overrideROIs[i]['HU'] = str(CTdensityInterpolation(overrideROIs[i]['REL_MASS_DENSITY']))

    print('external:', externalROI, '\noverrides:', overrideROIs)

    return externalROI, overrideROIs


def resample_and_override_CT(scriptdir, path_CT_DICOM_series, path_structures_DICOM_file, export_root, path_dose_DICOM_file = None, spacing = None, default_pixel_value_CT = -1000):
    externalROI, overrideROIs = get_structures_with_override(scriptdir, path_structures_DICOM_file)


    CT_obj = construct_CT_object('pCTp0', path_CT_DICOM_series, path_structures_DICOM_file, roi_names = [externalROI['ROIObservationLabel']] + [struct['ROIObservationLabel'] for struct in overrideROIs])

    if path_dose_DICOM_file:
        reference_dose_image = sitk.ReadImage(path_dose_DICOM_file)
        CT_obj.resample(reference_sitk = reference_dose_image, default_pixel_value_CT = default_pixel_value_CT, square_slices = True)
    elif spacing:
       CT_obj.resample(new_spacing = spacing, default_pixel_value_CT = default_pixel_value_CT)

    CT_obj_override = copy.deepcopy(CT_obj)


    new_CT_array = sitk.GetArrayFromImage(CT_obj_override.image)
    external = CT_obj_override.masks_structures[externalROI['ROIObservationLabel']]
    new_CT_array[external == False] = externalROI['HU']


    print("overridesROIs: ", overrideROIs)
    for override in overrideROIs:


        mask = CT_obj_override.masks_structures[override["ROIObservationLabel"]]
        new_CT_array[mask] = override['HU']



    new_CT_image = sitk.GetImageFromArray(new_CT_array)

    new_CT_image.SetOrigin(CT_obj_override.image.GetOrigin())
    new_CT_image.SetSpacing(CT_obj_override.image.GetSpacing())
    CT_obj_override.image = new_CT_image

    CT_obj_override.save(export_root, save_struct_file=False)

    print(sitk.GetArrayFromImage(CT_obj_override.image).max())

    return CT_obj_override