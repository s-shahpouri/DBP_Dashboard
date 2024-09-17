import os
import pandas as pd
from assess import (
    extract_losses, extract_tr_value, extract_loss_function_from_folder,
    keep_lowest_ensemble_losses, keep_lowest_single_losses,
    attach_csv_outputs, attach_data_dict_paths, format_losses
)


class BackDash:
    def __init__(self, main_root):
        """Initialize the BackDash class with a main root directory."""
        self.main_root = main_root  # Set the root directory
        self.folders = self.get_folders()  # Get the folders from the root directory
        self.all_results = []  # To store all extracted results
        self.df_cleaned = None  # Placeholder for the cleaned DataFrame


    def get_folders(self):
        """Get all directories in the main root directory."""
        return [f for f in os.listdir(self.main_root) if os.path.isdir(os.path.join(self.main_root, f))]

    def extract_folder_info(self):
        """Iterate over the folders and extract losses and other information."""
        for folder in self.folders:
            folder_parts = folder.split('_')
            # Ensure the folder has at least 3 parts for ensemble_id
            ensemble_id = folder_parts[2] if len(folder_parts) >= 3 else None

            # Extract losses from folder name
            train_loss, val_loss, test_loss, avg_train_loss, avg_val_loss, avg_test_loss = extract_losses(folder)

            # Extract 'tr' value from folder name
            tr_value = extract_tr_value(folder)

            # Append data to the all_results list
            self.all_results.append({
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

    def create_dataframe(self):
        """Create a DataFrame from the extracted folder information."""
        # Create DataFrame from the results
        df_all_models = pd.DataFrame(self.all_results)

        # Drop rows where the ensemble_id ('esm') is missing (optional)
        self.df_cleaned = df_all_models.dropna(subset=['esm'], how='all')

        # Extract and add the loss function name for each folder
        self.df_cleaned.loc[:, 'loss_function_name'] = self.df_cleaned['Folder'].apply(
            lambda folder: extract_loss_function_from_folder(os.path.join(self.main_root, folder))
        )
        
        return self.df_cleaned

    def process(self):
        """Main method to extract folder info and generate a cleaned DataFrame."""
        self.extract_folder_info()  # Extract losses and info from folders
        return self.create_dataframe()  # Create and return the cleaned DataFrame


    def process_ensemble(self, df_cleaned):
        """Process the ensemble approach and generate df_best_esms."""

        print("0000000000")
        print(df_cleaned.head())
        # Create df_ensemble
        df_ensemble = keep_lowest_ensemble_losses(df_cleaned)
        print("11111111111111")
        print(df_ensemble)
        # Step 1: Identify rows with 'Best_Esmb_X' in the tag
        best_esmb_rows = df_ensemble[df_ensemble['tag'].str.startswith('Esb_')]

        # Step 2: Get the unique 'esm' values from the 'esm' column of these rows
        best_esm_values = best_esmb_rows['esm'].unique()

        # Step 3: Filter the DataFrame to keep all rows that share the same 'esm' as the best rows
        df_best_esms = df_cleaned[df_cleaned['esm'].isin(best_esm_values)]
        print("2222222222222222")
        print(df_best_esms.head())
        # Attach CSV outputs
        df_best_esms = df_best_esms.apply(lambda row: attach_csv_outputs(self.main_root, row), axis=1)
        df_best_esms = df_best_esms.reset_index(drop=True)
        print("3333333333333333")
        print(df_best_esms.head())
        df_best_esms = df_best_esms.apply(lambda row: attach_data_dict_paths(self.main_root, row), axis=1)
        print("4444444444444")
        print(df_best_esms)
        # Update or assign the 'tag' column
        df_best_esms['tag'] = df_best_esms.apply(self.update_tag, axis=1)
        df_best_esms = df_best_esms.reset_index(drop=True)

        return df_best_esms

    def process_single(self, df_cleaned):
        """Process the single approach and generate df_single."""

        # Create df_single
        df_single = keep_lowest_single_losses(df_cleaned)

        # Attach CSV outputs
        df_single = df_single.apply(lambda row: attach_csv_outputs(self.main_root, row), axis=1)
        df_single = df_single.reset_index(drop=True)

        # Attach data dict paths
        df_single = df_single.apply(lambda row: attach_data_dict_paths(self.main_root, row), axis=1)

        return df_single

    @staticmethod
    def update_tag(row):
        """Update or assign the 'tag' column for all rows."""
        formatted_losses = format_losses(row['train_loss'], row['val_loss'], row['test_loss'])
        if pd.isna(row['tag']):
            return f"Esb_Test_AVG_[{formatted_losses}]"
        elif pd.notna(row['tr']):
            return f"Esb_Test_{int(row['tr'])}_[{formatted_losses}]"
        return row['tag']

    def average_outputs(self, df_best_esms, columns):
        """Average the specified output columns and update the last row."""
        for col in columns:
            split_dfs = []

            for i in range(len(df_best_esms) - 1):
                data = df_best_esms.iloc[i][col]
                df_split = data['PatientID;pred_0;pred_1;pred_2;true_0;true_1;true_2;Mode'].str.split(';', expand=True)
                df_split.columns = ['PatientID', 'pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2', 'Mode']
                split_dfs.append(df_split)

            for df in split_dfs:
                df[['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']] = df[
                    ['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']].apply(pd.to_numeric, errors='coerce')

            df_avg = split_dfs[0].copy()
            for df in split_dfs[1:]:
                df_avg[['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']] += df[
                    ['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']]

            df_avg[['pred_0', 'pred_1', 'pred_2', 'true_0', 'true_1', 'true_2']] /= len(split_dfs)
            df_avg['PatientID'] = split_dfs[0]['PatientID']
            df_avg['Mode'] = split_dfs[0]['Mode']

            df_avg_str = df_avg.to_csv(index=False, header=True, sep=';').strip()
            df_best_esms.at[df_best_esms.index[-1], col] = df_avg_str

        return df_best_esms


    def save_ensemble_pickle(self, df, pickle_file_path):

        df.to_pickle(pickle_file_path)


