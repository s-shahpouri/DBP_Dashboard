

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import re
import pandas as pd


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
                'Euc_dis': 'Euc',
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
        
        # Filter the outliers DataFrame based on the selected mode key
        return self.outliers_df[self.outliers_df['type'] == selected_mode_key]

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
            else:
                # Try to match only Iter or rCTp if both aren't available together
                match_iter = re.search(r'Iter(\d+)', moving_path)
                match_rCTp = re.search(r'rCTp(\d+)', moving_path)
                if match_iter and match_rCTp:
                    return f"Iter{match_iter.group(1)}_rCTp{match_rCTp.group(1)}"
                elif match_iter:
                    return f"Iter{match_iter.group(1)}"
                elif match_rCTp:
                    return f"rCTp{match_rCTp.group(1)}"
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
            true_0, true_1, true_2 = outlier['true_0'], outlier['true_1'], outlier['true_2']

            # Find matching entries in json_df
            matching_entries = self.json_df[
                (self.json_df['PatientID'] == patient_id) &
                (np.isclose(self.json_df['true_0'], true_0, atol=1e-3)) &
                (np.isclose(self.json_df['true_1'], true_1, atol=1e-3)) &
                (np.isclose(self.json_df['true_2'], true_2, atol=1e-3))
            ]

            if matching_entries.empty:
                # print(f"No match found for Patient ID: {patient_id} with coordinates {true_0}, {true_1}, {true_2}")
                pass
            elif len(matching_entries) > 1:
                print(f"Multiple matches found for Patient ID: {patient_id} with coordinates {true_0}, {true_1}, {true_2}")
            else:
                for _, entry in matching_entries.iterrows():
                    self.df.at[index, 'fixed'] = entry['fixed']
                    self.df.at[index, 'moving'] = entry['moving']
                    break

        return self.df

    def append_moving_extraction(self):
        """
        Append the extracted name from the moving column to the PatientID.
        """
        self.df['PatientID_with_diffs'] = self.df.apply(lambda row: f"{row['PatientID']}{self.extract_name(row['moving'])}", axis=1)
        self.df['PatientID'] = self.df.apply(lambda row: f"{row['PatientID']}_{self.extract_name(row['moving'])}", axis=1)
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
    # Drop rows with any NaN in avg_train_loss, avg_val_loss, or avg_test_loss to ensure we're comparing valid rows

    # Add 'tag' column for marking rows
    df_losses['tag'] = 'Other'

    # # Tagging the row with the lowest avg train loss
    # lowest_avg_train_idx = df_losses['avg_train_loss'].idxmin()
    # df_losses.loc[lowest_avg_train_idx, 'tag'] = (
    #     f"Esb_Train: [{df_losses.loc[lowest_avg_train_idx, 'avg_train_loss']}, "
    #     f"{df_losses.loc[lowest_avg_train_idx, 'avg_val_loss']}, "
    #     f"{df_losses.loc[lowest_avg_train_idx, 'avg_test_loss']}]"
    # )

    # # Tagging the row with the lowest avg validation loss
    # lowest_avg_val_idx = df_losses['avg_val_loss'].idxmin()
    # df_losses.loc[lowest_avg_val_idx, 'tag'] = (
    #     f"Esb_Val: [{df_losses.loc[lowest_avg_val_idx, 'avg_train_loss']}, "
    #     f"{df_losses.loc[lowest_avg_val_idx, 'avg_val_loss']}, "
    #     f"{df_losses.loc[lowest_avg_val_idx, 'avg_test_loss']}]"
    # )

    # Tagging the row with the lowest avg test loss
    lowest_avg_test_idx = df_losses['avg_test_loss'].idxmin()
    df_losses.loc[lowest_avg_test_idx, 'tag'] = (
        f"Esb_Test: [{df_losses.loc[lowest_avg_test_idx, 'avg_train_loss']}, "
        f"{df_losses.loc[lowest_avg_test_idx, 'avg_val_loss']}, "
        f"{df_losses.loc[lowest_avg_test_idx, 'avg_test_loss']}]"
    )

    # Find the index for the row with the lowest train loss
    lowest_train_idx = df_losses['train_loss'].idxmin()

    # Find the index for the row with the lowest validation loss
    lowest_val_idx = df_losses['val_loss'].idxmin()

    # Find the index for the row with the lowest test loss
    lowest_test_idx = df_losses['test_loss'].idxmin()

    # Extract the rows with the lowest losses for train, val, and test
    lowest_losses_df = df_losses.loc[[lowest_train_idx, lowest_val_idx, lowest_test_idx]]

    # Drop duplicates if the same row has the lowest loss for multiple metrics
    lowest_losses_df = lowest_losses_df.drop_duplicates()

    return lowest_losses_df



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

    print(f"Using path: {path}")
    
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


def folder_approach(path):
    print(path)
    all_results = []
    folder_parts = path.split('_')
    folder = '_'.join(folder_parts[5:])

    print(folder)
    if len(folder_parts) < 6:
        ensemble_id = None  # Assign None if we can't find the ensemble_id
    else:
        ensemble_id = folder_parts[6]
    train_loss, val_loss, test_loss, avg_train_loss, avg_val_loss, avg_test_loss = extract_losses(path)
    tr_value = extract_tr_value(path)
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
    df_cleaned.loc[:, 'loss_function_name'] = df_cleaned['Folder'].apply(lambda folder: extract_loss_function_from_folder(path))
    df_cleaned = keep_lowest_single_losses(df_cleaned)
    df_cleaned = df_cleaned.apply(lambda row: attach_csv_outputs(path, row), axis=1)
    df_cleaned = df_cleaned.apply(lambda row: attach_data_dict_paths(path, row), axis=1)
    pickle_file_path = "df_user.pkl"
    df_cleaned.to_pickle(pickle_file_path)

    return pickle_file_path
