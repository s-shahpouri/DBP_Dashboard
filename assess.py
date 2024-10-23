import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import re
import pandas as pd
import os
import shutil
from Libs.run_mqi import run_mqi
from Libs.construct_dict_mqi_input_parameters import construct_dict_mqi_input_parameters
from Libs.construct_dose_dcm_mqi import construct_dose_dcm_mqi
import json
import SimpleITK as sitk
from Libs.CT import construct_CT_object
from Libs.resample_and_override_CT import get_structures_with_override
import subprocess
import glob
import time
import pydicom

        # trans_values = {'z':(float(self.row['true_0']) - float (self.row['pred_0'])),
        #                 'y':(float(self.row['true_1']) - float(self.row['pred_1'])),
        #                 'x': (float(self.row['true_2']) - float(self.row['pred_2']))}


class DoseGenerator:
    def __init__(self, filtered_data_row, moqui_path='/data/sama/Gabriel_inf/computeDoseMoqui', raw_data_dir='/data/bahrdoh/Datasets/Raw_data', reg_dir='/data/bahrdoh/Datasets/pat_reg'):
        self.row = filtered_data_row
        self.reg_dir = reg_dir
        self.moqui_path = moqui_path
        self.raw_data_dir = raw_data_dir
        self.root_mqi_binary = os.path.join(moqui_path, "Moqui")
        self.name_mqi_binary = 'moqui'
        self.patient_id = f"DBP_{self.row['PatientID']}"

    def generate_dose_paths(self):
        patient_id_with_diffs = self.row['PatientID_with_diffs']

        parts = patient_id_with_diffs.split('_')
        if len(parts) >= 3:
            self.rct_part = parts[2]  # e.g., 'rCTp17'
            iter_part = parts[1]  # e.g., 'Iter8'
            sanitized_value = f"X_{parts[-1].split('(')[-1].replace(')', '')}"
        else:
            raise ValueError("Invalid format for PatientID_with_diffs")

        main_dose_path = '/data/sama/Datasets/Dash_data/Dose_outlier'
        ct_moving_path_pred = os.path.join(main_dose_path, self.patient_id, iter_part, self.rct_part, sanitized_value, 'CT_pred')
        dose_moving_path_pred = os.path.join(main_dose_path, self.patient_id, iter_part, self.rct_part, sanitized_value, 'Dose_pred')
        ct_moving_path_true = os.path.join(main_dose_path, self.patient_id, iter_part, self.rct_part, sanitized_value, 'CT_true')
        dose_moving_path_true = os.path.join(main_dose_path, self.patient_id, iter_part, self.rct_part, sanitized_value, 'Dose_true')

        # Ensure directories exist
        os.makedirs(ct_moving_path_pred, exist_ok=True)
        os.makedirs(dose_moving_path_pred, exist_ok=True)
        os.makedirs(ct_moving_path_true, exist_ok=True)
        os.makedirs(dose_moving_path_true, exist_ok=True)



        self.row['dose_moving_pred'] = dose_moving_path_pred
        self.row['ct_moving_pred'] = ct_moving_path_pred
        self.row['dose_moving_true'] = dose_moving_path_true
        self.row['ct_moving_true'] = ct_moving_path_true


        return ct_moving_path_pred, dose_moving_path_pred, ct_moving_path_true, dose_moving_path_true


    def sort_dicom_by_instance_number(self, dicom_directory):
        # Get all DICOM file paths in the directory
        dicom_filenames = glob.glob(os.path.join(dicom_directory, 'CT*.dcm'))

        # Read and sort DICOM files by the InstanceNumber tag
        file_metadata = []
        for filename in dicom_filenames:
            try:
                ds = pydicom.dcmread(filename)
                instance_number = int(ds.InstanceNumber)
                file_metadata.append((filename, instance_number))
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

        # Sort the files by InstanceNumber
        sorted_files = sorted(file_metadata, key=lambda x: x[1])

        # Return the sorted file paths
        sorted_filenames = [file[0] for file in sorted_files]
        return sorted_filenames
    

    def generate_ct_objects(self):
        # Construct the data dictionary for patient and rCT

        
        # Extract CT structure directories for pCT and rCT
        pCT_struct_dirs = self.data_dict.get('pCTp0')
        rCT_struct_dirs = self.data_dict.get(self.rct_part)

        if not pCT_struct_dirs or not rCT_struct_dirs:
            raise ValueError("Missing data for pCT or rCT in data_dict.")

        # Construct CT objects for pCT and rCT
        
        CTobject, externalROI_moving, overrideROIs_moving = self.construct_ct_obj(rCT_struct_dirs)

        if CTobject is None:
            raise ValueError("Failed to create CT object for rCT.")
    
        self.CTobject = CTobject
        self.externalROI_moving = externalROI_moving
        self.overrideROIs_moving = overrideROIs_moving

    def run_dose_calculation(self, ct_moving_path, dose_moving_path):
        
        # Construct the input parameters for MQI
        mqi_input_parameters = construct_dict_mqi_input_parameters(
            ct_moving_path, "", 
            output_dir=dose_moving_path, 
            GPUID=1, 
            particles_per_history=10000
        )

        # Run MQI using the correct call and handle potential failures gracefully
        try:
            run_mqi(
                self.root_mqi_binary,
                self.name_mqi_binary,
                mqi_input_parameters,
                rerun=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Run MQI failed after several attempts: {e}")
            raise RuntimeError("MQI run failed. Aborting further processing.")

        # Debugging: Check input file contents
        input_file_path = os.path.join(dose_moving_path, "input_file.in")
        if os.path.exists(input_file_path):
            with open(input_file_path, "r") as infile:
                input_content = infile.read()
                print("Input File Contents:")
                print(input_content)

        # Save dose DICOM
        try:
            construct_dose_dcm_mqi(
                dose_moving_path,
                self.CTobject.reference_dcm,
                frame_of_reference_uid=self.CTobject.frame_of_reference_UID
            )
        except Exception as e:
            print(f"Error saving dose DICOM: {e}")
            raise

        print(f"Dose image generated and saved to {dose_moving_path}")


    def make_data_dict(self):
        raw_data_dir = self.raw_data_dir

        # Initialize data_dict as an empty dictionary
        self.data_dict = {}

        pCTp0_path = os.path.join(raw_data_dir, self.patient_id, 'pCTp0')
        if os.path.exists(pCTp0_path):
            pCT_struct_dir = glob.glob(os.path.join(pCTp0_path, "RS*.dcm"))
            if pCT_struct_dir:
                self.data_dict['pCTp0'] = {
                    'ct_dir': pCTp0_path,
                    'struct_dir': pCT_struct_dir[0]
                }
            else:
                print(f"No structure files found.")
        else:
            print(f"{pCTp0_path} does not exist!!!!")

        # Construct the path for the rCT
        rct_path = os.path.join(raw_data_dir, self.patient_id, self.rct_part)
        if os.path.exists(rct_path):
            struct_dir = glob.glob(os.path.join(rct_path, "RS*.dcm"))
            if struct_dir:
                dose_dir = glob.glob(os.path.join(raw_data_dir, self.patient_id, 'A1PHH', "RD*.dcm"))
                plan_dir = glob.glob(os.path.join(raw_data_dir, self.patient_id, 'A1PHH', "RP*.dcm"))
                self.data_dict[self.rct_part] = {
                    'ct_dir': rct_path,
                    'struct_dir': struct_dir[0],
                    'dose_dir': dose_dir[0] if dose_dir else None,
                    'plan_dir': plan_dir[0] if plan_dir else None
                }
            else:
                print(f"No structure files found in {rct_path}")
        else:
            print(f"{rct_path} does not exist!!!!")

        self.add_reg_dir()  # Add registration information to the dictionary
        print("UPDATING DONE !!!!")
        return self.data_dict  # Return the updated data_dict



    def add_reg_dir(self):
        reg_dir_list = glob.glob(os.path.join(self.reg_dir, "*.json"))

        for reg_file in reg_dir_list:
            string_list = reg_file.split('/')
            pat_id = string_list[-1][:9]  # Assumes the patient ID can be extracted this way
            
            if pat_id == self.patient_id:
                self.row['reg_dir'] = reg_file
                break



    def new_transfer_param(self, rig_matrix):

          
        pred_matrix = rig_matrix.copy() 
        true_matrix = rig_matrix.copy()
        
        pred_trans_values = {'z': float(self.row['pred_0']), 'y': float(self.row['pred_1']), 'x': float(self.row['pred_2'])}
        true_trans_values = {'z': float(self.row['true_0']), 'y': float(self.row['true_1']), 'x': float(self.row['true_2'])}

        # Update the translation components at the corresponding indices in the 1D array
        for idx, key in zip([3, 7, 11], ['x', 'y', 'z']):
            # Modify the translation component (at indices 3, 7, 11)
            pred_matrix[idx] = (pred_matrix[idx] * 10) + true_trans_values[key] - pred_trans_values[key]
            true_matrix[idx] = (true_matrix[idx] * 10)  
               
        print("####################")
        print(rig_matrix)
        print(true_trans_values)
        print(true_matrix)
        print("###############")
        return pred_matrix.reshape(4, 4), true_matrix.reshape(4, 4),



    def register_ct_struct(self, ct_moving_path, coord_type):
        inf = self.data_dict[self.rct_part]
        # print(self.data_dict)
        # print(inf)
        # Load reference dose image
        ref_sitk = sitk.ReadImage(inf['dose_dir'])

        # Print dimensions before registration
        print(f"Before Registration: Reference image (dose) dimensions: {ref_sitk.GetSize()}")
        for roi_name, mask in self.CTobject.masks_structures.items():
            print(f"Before Registration: Mask '{roi_name}' dimensions: {mask.shape}")
        
     
        # Check if registration data is available
        if self.row['reg_dir']:
            with open(self.row['reg_dir']) as f:
                reg_file = json.load(f)


            # Extract registration matrix and frame of reference UID
            registration_matrix = np.array(reg_file['examinations'][self.rct_part]['registration_to_planning_examinations']['A1PHH']['rigid_transformation_matrix'])


            frame_of_uid = reg_file['examinations'][self.rct_part]['equipment_info']['frame_of_reference']

            # Apply transformation and update the final dictionary
            pred_matrix, true_matrix = self.new_transfer_param(registration_matrix)
            
            if coord_type == 'pred':
                self.CTobject.transform_and_resample(transformation_matrix=pred_matrix, reference_sitk=ref_sitk, new_FoR_UID=frame_of_uid)

            else:
                # Apply transformation and resample the CT object
                self.CTobject.transform_and_resample(transformation_matrix=true_matrix, reference_sitk=ref_sitk, new_FoR_UID=frame_of_uid)

            # Print dimensions after transformation and resampling
            for roi_name, mask in self.CTobject.masks_structures.items():
                print(f"After Resampling: Mask '{roi_name}' dimensions: {mask.shape}")
        else:
            print(f"No registration file found for {self.rct_part}, skipping registration.")
            self.CTobject.resample(reference_sitk=ref_sitk, square_slices=False)

            # Print dimensions after resampling (without transformation)
            for roi_name, mask in self.CTobject.masks_structures.items():
                print(f"After Resampling: Mask '{roi_name}' dimensions: {mask.shape}")

        # Apply overrides
        self.CTobject.override(self.externalROI_moving, self.overrideROIs_moving)

        # Print final CT dimensions after override
        print(f"After Override: CT image dimensions: {sitk.GetArrayFromImage(self.CTobject.image).shape}")

        # Save the updated CT object
        self.CTobject.save(ct_moving_path, save_struct_file=True)

        shutil.copy2(inf['plan_dir'], ct_moving_path)



    def make_nrrd(self, ct_moving_path, dose_moving_path):
        dicom_file_path = os.path.join(dose_moving_path, 'RD-moqui.dcm')
        nrrd_file_path = os.path.splitext(dicom_file_path)[0] + '.nrrd'
        dicom_image = sitk.ReadImage(dicom_file_path)
        sitk.WriteImage(dicom_image, nrrd_file_path)

        self.row['pred_dose_moving'] = nrrd_file_path
        print(f"File converted and saved as: {nrrd_file_path}")


        sorted_ct_file_names = self.sort_dicom_by_instance_number(ct_moving_path)


        if not sorted_ct_file_names:
            print("No CT DICOM files found.")
            return
        

       # Use SimpleITK to read the sorted series of DICOM slices and create a 3D volume
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_ct_file_names)
        # Ensure DICOM orientation is respected
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        ct_volume = reader.Execute()

        # # Save the CT volume as NRRD
        # ct_nrrd_file_path = os.path.join(ct_moving_path, 'CT_volume.nrrd')  # Ensure the correct directory
        # sitk.WriteImage(ct_volume, ct_nrrd_file_path)


        # Flip the CT volume along the Z-axis to correct orientation
        ct_volume_flipped = sitk.Flip(ct_volume, [False, False, True])  # Flip the Z-axis (third axis)

        # Save the flipped CT volume as NRRD
        ct_nrrd_file_path = os.path.join(ct_moving_path, 'CT_volume.nrrd')  # Ensure the correct directory
        sitk.WriteImage(ct_volume_flipped, ct_nrrd_file_path)

        
        self.row['pred_ct_moving'] = ct_nrrd_file_path
        print(f"CT series converted and saved as: {ct_nrrd_file_path}")


    def construct_ct_obj(self, ct_struct_dirs):

        # Get structures with override information
        externalROI, overrideROIs = get_structures_with_override(self.moqui_path,  ct_struct_dirs['struct_dir'])
        roi_names = [externalROI['ROIObservationLabel']] + [struct['ROIObservationLabel'] for struct in overrideROIs] + ["CTV_7000", "CTV_5425"]
        roi_names = list(set(roi_names))
        
        if 'Vullingen-implan' in roi_names:
            roi_names.remove('Vullingen-implan')

        print(ct_struct_dirs['ct_dir'], ct_struct_dirs['struct_dir'], roi_names)
        # Construct the CT object only if valid DICOM files are present
        CTobject = construct_CT_object('CT', ct_struct_dirs['ct_dir'], ct_struct_dirs['struct_dir'], roi_names = roi_names)
        return CTobject, externalROI, overrideROIs
    


    # def process(self):
    #     try:
    #         self.start_time = time.time()

    #         # Generate dose paths
    #         ct_moving_path_pred, dose_moving_path_pred, ct_moving_path_true, dose_moving_path_true = self.generate_dose_paths()

            
    #         # Define dose and CT file paths
    #         dose_file_path = os.path.join(dose_moving_path_pred, 'RD-moqui.nrrd')
    #         ct_file_path = os.path.join(ct_moving_path_pred, 'CT_volume.nrrd')

    #         dose_exists = os.path.exists(dose_file_path)
    #         ct_exists = os.path.exists(ct_file_path)


    #         if dose_exists and ct_exists:
                
    #             # Update self.row with existing file paths
    #             self.row['dose_moving_pred'] = dose_file_path
    #             self.row['ct_moving_pred'] = ct_file_path

    #             # Return the updated row without running dose generation
    #             return self.row

    #         else:
    #             print("No existing dose or CT found. Proceeding with dose and CT generation.")
                
    #             # Generate the data dictionary
    #             self.data_dict = self.make_data_dict()
    #             print("Data dictionary generated.")

    #             # Generate CT objects
    #             self.generate_ct_objects()
    #             print("CT objects generated.")

    #             # Register and adjust CT object
    #             self.register_ct_struct(ct_moving_path_pred)
    #             print("CT registration finished.")

    #             # Run dose calculation
    #             self.run_dose_calculation(ct_moving_path_pred, dose_moving_path_pred)

    #             # Generate NRRD files (CT and dose)
    #             self.make_nrrd(ct_moving_path_pred, dose_moving_path_pred)
    #             print("Dose calculation and NRRD generation finished.")

    #             # Update self.row with new file paths
    #             self.row['pred_dose_moving'] = dose_file_path
    #             self.row['pred_ct_moving'] = ct_file_path

    #     except Exception as e:
    #         print(f"Error in processing dose generation: {e}")

    #     return self.row

    def process(self):
        try:
            self.start_time = time.time()

            # Generate dose paths for both predicted and true
            ct_moving_path_pred, dose_moving_path_pred, ct_moving_path_true, dose_moving_path_true = self.generate_dose_paths()

            # Define dose and CT file paths for both true and pred
            paths = {
                'pred': (dose_moving_path_pred, ct_moving_path_pred),
                'true': (dose_moving_path_true, ct_moving_path_true)
            }

            for coord_type in ['pred', 'true']:
                dose_file_path, ct_file_path = paths[coord_type]

                dose_exists = os.path.exists(os.path.join(dose_file_path, 'RD-moqui.nrrd'))
                ct_exists = os.path.exists(os.path.join(ct_file_path, 'CT_volume.nrrd'))

                if dose_exists and ct_exists:
                    print(f"{coord_type.capitalize()} Dose and CT already exist. Reusing existing files.")
                    
                    # Update self.row with existing file paths
                    self.row[f'dose_moving_{coord_type}'] = os.path.join(dose_file_path, 'RD-moqui.nrrd')
                    self.row[f'ct_moving_{coord_type}'] = os.path.join(ct_file_path, 'CT_volume.nrrd')

                else:
                    print(f"Generating {coord_type.capitalize()} Dose and CT.")

                    # Generate the data dictionary
                    self.data_dict = self.make_data_dict()
                    print("Data dictionary generated.")

                    # Generate CT objects
                    self.generate_ct_objects()
                    print("CT objects generated.")

                    # Register and adjust CT object
                    self.register_ct_struct(ct_file_path, coord_type)
                    print("CT registration finished.")

                    # Run dose calculation
                    self.run_dose_calculation(ct_file_path, dose_file_path)

                    # Generate NRRD files (CT and dose)
                    self.make_nrrd(ct_file_path, dose_file_path)
                    print(f"{coord_type.capitalize()} Dose and CT generation finished.")

                    # Update self.row with new file paths
                    self.row[f'dose_moving_{coord_type}'] = os.path.join(dose_file_path, 'RD-moqui.nrrd')
                    self.row[f'ct_moving_{coord_type}'] = os.path.join(ct_file_path, 'CT_volume.nrrd')

        except Exception as e:
            print(f"Error in processing dose generation: {e}")

        return self.row


class OutlierDetector:
    def __init__(self, df, json_df):
        """
        Initialize the OutlierDetector with the main dataframe and json dataframe.
        PathName class is used to handle path processing and appending differences to PatientID.
        """
        # Create an instance of PathName to process paths
        self.path_finder = PathName(df, json_df)
        self.df = self.path_finder.process()  # Process paths using the PathName class
        self.outliers_df = None  # Will hold the concatenated outlier DataFrame

    def compute_outliers(self):
        """
        Compute the outliers for each axis (0_dis, 1_dis, 2_dis, Euc_dis, L1_dis, L2_dis).
        """
        outliers_0_dis = self.find_outliers('0_dis')
        outliers_1_dis = self.find_outliers('1_dis')
        outliers_2_dis = self.find_outliers('2_dis')
        outliers_euclidean_dist = self.find_outliers('Euc_dis')
        outliers_L1_dis = self.find_outliers('L1_dis')
        outliers_L2_dis = self.find_outliers('L2_dis')

        # Set the 'type' column for each outlier DataFrame
        outliers_0_dis['type'] = '0_dis'
        outliers_1_dis['type'] = '1_dis'
        outliers_2_dis['type'] = '2_dis'
        outliers_euclidean_dist['type'] = 'Euc_dis'
        outliers_L1_dis['type'] = 'L1_dis'
        outliers_L2_dis['type'] = 'L2_dis'

        # Concatenate all outlier DataFrames into a single DataFrame
        self.outliers_df = pd.concat([outliers_0_dis, outliers_1_dis, outliers_2_dis, outliers_euclidean_dist, outliers_L1_dis, outliers_L2_dis], ignore_index=True)

    def find_outliers(self, column):
        """
        Helper function to find outliers for a specific column using the IQR method.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]

    def append_differences_to_patient_id(self, selected_mode_key):
        """
        Append the exact difference for the selected axis (or metric) to PatientID_with_diffs.
        """
        def append_selected_diff(row):
            # Calculate the difference based on the selected mode key
            diff_value = None

            if selected_mode_key == '0_dis':
                diff_value = row['pred_0'] - row['true_0']
            elif selected_mode_key == '1_dis':
                diff_value = row['pred_1'] - row['true_1']
            elif selected_mode_key == '2_dis':
                diff_value = row['pred_2'] - row['true_2']
            elif selected_mode_key == 'Euc_dis':
                diff_value = row['Euc_dis']
            elif selected_mode_key == 'L1_dis':
                diff_value = row['L1_dis']
            elif selected_mode_key == 'L2_dis':
                diff_value = row['L2_dis']

            # Round the difference value to 2 decimal places
            diff_value = round(diff_value, 2)

            # Create a mapping to determine which axis is being used
            mode_mapping = {
                '0_dis': 'X',
                '1_dis': 'Y',
                '2_dis': 'Z',
                'Euc_dis': 'R',
                'L1_dis': 'L1',
                'L2_dis': 'L2'
            }

            # Append the difference and axis to the PatientID_with_diffs
            return f"{row['PatientID_with_diffs']}_{mode_mapping[selected_mode_key]}({diff_value})"

        # Apply the function to each row in the outliers DataFrame
        self.outliers_df['PatientID_with_diffs'] = self.outliers_df.apply(append_selected_diff, axis=1)

        return self.outliers_df

    def filter_outliers_by_mode(self, selected_mode_key):
        """
        Filter the outliers based on the selected mode key (e.g., '0_dis', '1_dis', etc.).
        """
        if self.outliers_df is None:
            raise ValueError("Outliers have not been computed. Call `compute_outliers` first.")
        # Filter outliers based on the selected mode key
        filtered_outliers_df = self.outliers_df[self.outliers_df['type'] == selected_mode_key].copy()
        
        # Initialize new columns with empty strings
        filtered_outliers_df.loc[:, 'dose_fixed'] = ""
        filtered_outliers_df.loc[:, 'dose_moving'] = ""


        return filtered_outliers_df


    def get_outlier_options(self, filtered_outliers_df):
        """
        Get the unique PatientID options from the filtered outliers.
        """
        return filtered_outliers_df['PatientID_with_diffs'].unique()

    def process(self, selected_mode_key):
        """
        Compute outliers and append the exact axis outlier to PatientID based on the selected mode.
        """
        self.compute_outliers()  # Compute outliers for all axes/metrics
        self.append_differences_to_patient_id(selected_mode_key)  # Append the selected axis difference
        return self.df







class PathName:
    def __init__(self, df, json_df):
        """
        Initialize the PathFinder with the main dataframe and json dataframe.
        """
        self.df = df.copy()  # Work on a copy of the df to avoid modifying the original
        self.json_df = json_df

    def extract_name(self, moving_path):
        """
        Extract relevant details (Iter and rCTp) from the moving path using regex.
        """
        if moving_path and isinstance(moving_path, str):
            # First attempt to match both Iter and rCTp in the path
            match = re.search(r'Iter(\d+).*?_rCTp(\d+)', moving_path)
            if match:
                return f"Iter{match.group(1)}_rCTp{match.group(2)}"
            # else:
            #     # Try to match only Iter or rCTp if both aren't available together
            #     match_iter = re.search(r'Iter(\d+)', moving_path)
            #     match_rCTp = re.search(r'rCTp(\d+)', moving_path)
            #     if match_iter and match_rCTp:
            #         return f"Iter{match_iter.group(1)}_rCTp{match_rCTp.group(1)}"
            #     elif match_iter:
            #         return f"Iter{match_iter.group(1)}"
            #     elif match_rCTp:
            #         return f"rCTp{match_rCTp.group(1)}"
        return ""  # If no match, return empty string

    def find_paths(self):
        """
        Find the matching paths in the json_df and update the fixed and moving paths in the dataframe.
        """
        self.df['fixed'] = None
        self.df['moving'] = None

        # Iterate through each row in the main dataframe
        for index, outlier in self.df.iterrows():
            patient_id = outlier['PatientID']
            patient_id_0 = patient_id.split('_')[0]

            true_0, true_1, true_2 = outlier['true_0'], outlier['true_1'], outlier['true_2']

            # Find matching entries in json_df
            matching_entries = self.json_df[
                (self.json_df['PatientID'] == patient_id_0) &
                (np.isclose(self.json_df['true_0'], true_0, atol=1e-3)) &
                (np.isclose(self.json_df['true_1'], true_1, atol=1e-3)) &
                (np.isclose(self.json_df['true_2'], true_2, atol=1e-3))
            ]

            if matching_entries.empty:
                print(f"No match found for Patient ID: {patient_id} with coordinates {true_0}, {true_1}, {true_2}")
                pass
            elif len(matching_entries) > 1:
                print(f"Multiple matches found for Patient ID: {patient_id} with coordinates {true_0}, {true_1}, {true_2}")
            else:
                for _, entry in matching_entries.iterrows():
                    self.df.at[index, 'fixed'] = entry['fixed']
                    self.df.at[index, 'moving'] = entry['moving']
                    break
        # print(self.df['fixed'])
        return self.df

    def append_moving_extraction(self):
        """
        Append the extracted name from the moving column to the PatientID.
        """
        # Extract the name from 'moving' and create a new PatientID with appended information
        self.df['extracted_name'] = self.df.apply(lambda row: self.extract_name(row['moving']), axis=1)

        # Use the extracted name to create the final PatientID and PatientID_with_diffs
        self.df['PatientID_with_diffs'] = self.df.apply(lambda row: f"{row['PatientID']}_{row['extracted_name']}", axis=1)
        # self.df['PatientID'] = self.df['PatientID_with_diffs']

        # Drop the temporary column after use
        self.df.drop(columns=['extracted_name'], inplace=True)

        # print(self.df['PatientID'])
        return self.df



    def process(self):
        """
        Run both the find_paths and append_moving_extraction functions.
        """
        self.find_paths()
        self.append_moving_extraction()
        return self.df

import json
# Function to extract both normal and average losses from folder names
def extract_losses(folder_name):
    # Regex for standard train/val/test values
    match = re.search(r'tr_(\d+\.\d+)_val_(\d+\.\d+)_test_(\d+\.\d+)', folder_name)
    # Regex for ensemble with avg values (optional)
    avg_match = re.search(r'avg_tr_(\d+\.\d+)_val_(\d+\.\d+)_test_(\d+\.\d+)', folder_name)

    # Extract regular train/val/test
    if match:
        train_loss = float(match.group(1))
        val_loss = float(match.group(2))
        test_loss = float(match.group(3))
    else:
        train_loss, val_loss, test_loss = None, None, None

    # Extract average train/val/test (if present)
    if avg_match:
        avg_train_loss = float(avg_match.group(1)) if avg_match.group(1) else None
        avg_val_loss = float(avg_match.group(2)) if avg_match.group(2) else None
        avg_test_loss = float(avg_match.group(3)) if avg_match.group(3) else None
    else:
        avg_train_loss, avg_val_loss, avg_test_loss = None, None, None

    return train_loss, val_loss, test_loss, avg_train_loss, avg_val_loss, avg_test_loss

# Function to extract the 'tr' value from folder names (e.g., '_LReLu_1_')
def extract_tr_value(folder_name):
    tr_match = re.search(r'_LReLu_(\d+)_', folder_name)
    if tr_match:
        return int(tr_match.group(1))  # Return the numeric value as an integer
    return None

def extract_loss_function_from_folder(folder_path):
    # Look for any file starting with "suggested_params_trial"
    for file_name in os.listdir(folder_path):

        
        if 'params_trial' in file_name and file_name.endswith('.json'):
            json_file_path = os.path.join(folder_path, file_name)

            try:
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                    return data.get('loss_function_name', None)  # Return the loss function name
            except (FileNotFoundError, json.JSONDecodeError):
                return None  # Return None if there's an issue with the file
    return None  # Return None if no matching JSON file is found


import os 

def keep_lowest_single_losses(df):
    # Drop rows with any NaN in train, val, or test losses to ensure we're comparing valid rows
    cols_to_drop = ['avg_val_loss', 'avg_test_loss', 'avg_train_loss']

    # Filter only the columns that exist in the DataFrame
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    # Drop those columns if they exist
    df_losses = df.drop(columns=cols_to_drop)
    
    df_losses['tag'] = 'Other'

    # Tagging the row with the lowest train loss
    lowest_train_idx = df_losses['train_loss'].idxmin()
    formatted_losses_train = format_losses(
        df_losses.loc[lowest_train_idx, 'train_loss'], 
        df_losses.loc[lowest_train_idx, 'val_loss'], 
        df_losses.loc[lowest_train_idx, 'test_loss']
    )
    df_losses.loc[lowest_train_idx, 'tag'] = f"Best_Train_single: [{formatted_losses_train}]"

    # Tagging the row with the lowest validation loss
    lowest_val_idx = df_losses['val_loss'].idxmin()
    formatted_losses_val = format_losses(
        df_losses.loc[lowest_val_idx, 'train_loss'], 
        df_losses.loc[lowest_val_idx, 'val_loss'], 
        df_losses.loc[lowest_val_idx, 'test_loss']
    )
    df_losses.loc[lowest_val_idx, 'tag'] = f"Best_Val_single: [{formatted_losses_val}]"

    # Tagging the row with the lowest test loss
    lowest_test_idx = df_losses['test_loss'].idxmin()
    formatted_losses_test = format_losses(
        df_losses.loc[lowest_test_idx, 'train_loss'], 
        df_losses.loc[lowest_test_idx, 'val_loss'], 
        df_losses.loc[lowest_test_idx, 'test_loss']
    )
    df_losses.loc[lowest_test_idx, 'tag'] = f"Best_Test_single: [{formatted_losses_test}]"

    # Select the rows with the lowest train, validation, and test losses
    lowest_losses_df = df_losses.loc[[lowest_train_idx, lowest_val_idx, lowest_test_idx]]

    # Drop duplicates if the same row has the lowest loss for multiple metrics
    lowest_losses_df = lowest_losses_df.drop_duplicates()

    return lowest_losses_df


def keep_lowest_ensemble_losses(df_losses):
    # Add 'tag' column for marking rows
    df_losses['tag'] = 'Other'  # Default tag

    # Ensure we drop rows where avg losses are NaN for valid comparisons
    df_losses_clean = df_losses.dropna(subset=['avg_train_loss', 'avg_val_loss', 'avg_test_loss'], how='all')

    # # Tagging the row with the lowest avg train loss
    # if not df_losses_clean['avg_train_loss'].isnull().all():
    #     lowest_avg_train_idx = df_losses_clean['avg_train_loss'].idxmin()
    #     df_losses.loc[lowest_avg_train_idx, 'tag'] = (
    #         f"Esb_Train: [{df_losses.loc[lowest_avg_train_idx, 'avg_train_loss']}, "
    #         f"{df_losses.loc[lowest_avg_train_idx, 'avg_val_loss']}, "
    #         f"{df_losses.loc[lowest_avg_train_idx, 'avg_test_loss']}]"
    #     )

    # # Tagging the row with the lowest avg validation loss
    # if not df_losses_clean['avg_val_loss'].isnull().all():
    #     lowest_avg_val_idx = df_losses_clean['avg_val_loss'].idxmin()
    #     df_losses.loc[lowest_avg_val_idx, 'tag'] = (
    #         f"Esb_Val: [{df_losses.loc[lowest_avg_val_idx, 'avg_train_loss']}, "
    #         f"{df_losses.loc[lowest_avg_val_idx, 'avg_val_loss']}, "
    #         f"{df_losses.loc[lowest_avg_val_idx, 'avg_test_loss']}]"
    #     )

    # Tagging the row with the lowest avg test loss
    
    lowest_avg_test_idx = df_losses_clean['avg_test_loss'].idxmin()
    df_losses.loc[lowest_avg_test_idx, 'tag'] = (
            f"Esb_Test: [{df_losses.loc[lowest_avg_test_idx, 'avg_train_loss']}, "
            f"{df_losses.loc[lowest_avg_test_idx, 'avg_val_loss']}, "
            f"{df_losses.loc[lowest_avg_test_idx, 'avg_test_loss']}]"
        )

    # Now we return only the rows that were tagged with Esb_Train, Esb_Val, or Esb_Test
    tagged_df = df_losses[df_losses['tag'] != 'Other'].drop_duplicates()

    return tagged_df  # Return DataFrame with only the lowest losses




def format_losses(train_loss, val_loss, test_loss):
    # Round the values to 4 decimal places and format them with a space after commas
    return f"{round(train_loss, 2)}, {round(val_loss, 2)}, {round(test_loss, 2)}"

# Function to read CSV data and attach it to the DataFrame
def attach_csv_outputs(_path, row):
    try:

        _path = os.path.join(_path, row['Folder'])

        train_output, val_output, test_output = None, None, None

        # Paths to the expected CSV files
        test_file = os.path.join(_path, 'Dual_DCNN_LReLu_test_outputs.csv')
        val_file = os.path.join(_path, 'Dual_DCNN_LReLu_val_outputs.csv')
        train_file = os.path.join(_path, 'Dual_DCNN_LReLu_train_outputs.csv')

        # Read test CSV file if it exists
        if os.path.exists(test_file):
            test_output = pd.read_csv(test_file)
            row['test_output'] = test_output  # Attach the test CSV data (you can customize how to attach the content)

        # Read val CSV file if it exists
        if os.path.exists(val_file):
            val_output = pd.read_csv(val_file)
            row['val_output'] = val_output  # Attach the validation CSV data

        # Read train CSV file if it exists
        if os.path.exists(train_file):
            train_output = pd.read_csv(train_file)
            row['train_output'] = train_output  # Attach the train CSV data


    except:
        
            
        _path 
        train_output, val_output, test_output = None, None, None

        # Paths to the expected CSV files
        test_file = os.path.join(_path, 'Dual_DCNN_LReLu_test_outputs.csv')
        val_file = os.path.join(_path, 'Dual_DCNN_LReLu_val_outputs.csv')
        train_file = os.path.join(_path, 'Dual_DCNN_LReLu_train_outputs.csv')

        # Read test CSV file if it exists
        if os.path.exists(test_file):
            test_output = pd.read_csv(test_file)
            row['test_output'] = test_output  # Attach the test CSV data (you can customize how to attach the content)

        # Read val CSV file if it exists
        if os.path.exists(val_file):
            val_output = pd.read_csv(val_file)
            row['val_output'] = val_output  # Attach the validation CSV data

        # Read train CSV file if it exists
        if os.path.exists(train_file):
            train_output = pd.read_csv(train_file)
            row['train_output'] = train_output  # Attach the train CSV data

    return row


def attach_data_dict_paths(_path, row):
    """
    Attach paths for train_dict, val_dict, and test_dict based on whether _path is a root path or a folder path.
    """

    # Initialize empty values for train, val, and test dictionaries
    train_dict, val_dict, test_dict = None, None, None

    # Determine if _path is a root directory or a specific folder path by checking for numbers in the path.
    if re.search(r'\d{8}_\d{6}', _path):  # Look for pattern with numbers like "20240829_225243"
        path = _path  # _path is already the folder path
    else:
        # _path is a root directory, so join it with the folder from the row
        path = os.path.join(_path, row['Folder'])
    
    # Extract the tr value from the row and build the full path to the JSON file
    tr_value = int(row['tr'])
    json_file_path = os.path.join(path, f'Data_dict_{tr_value}.json')

    # Check if the JSON file exists and read it
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

            # Extract the paths from train_dict, val_dict, and test_dict
            train_dict = data.get('train_dict', [])
            val_dict = data.get('val_dict', [])
            test_dict = data.get('test_dict', [])

            # Attach the data to the row
            row['train_dict'] = train_dict
            row['val_dict'] = val_dict
            row['test_dict'] = test_dict
    else:
        print(f"JSON file not found: {json_file_path}")
    
    return row


def folder_approach(input_path, output_path):
    # print(input_path)
    all_results = []
    folder_parts = input_path.split('_')
    folder = '_'.join(folder_parts[5:])

    # print(folder)
    if len(folder_parts) < 6:
        ensemble_id = None  # Assign None if we can't find the ensemble_id
    else:
        ensemble_id = folder_parts[6]
    train_loss, val_loss, test_loss, avg_train_loss, avg_val_loss, avg_test_loss = extract_losses(input_path)
    tr_value = extract_tr_value(input_path)
    all_results.append({
        'esm': ensemble_id,
        'tr': tr_value,
        'Folder': folder,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
        'avg_train_loss': avg_train_loss,
        'avg_val_loss': avg_val_loss,
        'avg_test_loss': avg_test_loss
    })

    # Create DataFrames from the results
    df_all_models = pd.DataFrame(all_results)

    # Drop rows where the ensemble_id is missing (optional)
    df_cleaned = df_all_models.dropna(subset=['esm'], how='all')
    df_cleaned.loc[:, 'loss_function_name'] = df_cleaned['Folder'].apply(lambda folder: extract_loss_function_from_folder(input_path))
    df_cleaned = keep_lowest_single_losses(df_cleaned)
    df_cleaned = df_cleaned.apply(lambda row: attach_csv_outputs(input_path, row), axis=1)
    df_cleaned = df_cleaned.apply(lambda row: attach_data_dict_paths(input_path, row), axis=1)
    df_cleaned.to_pickle(output_path)

