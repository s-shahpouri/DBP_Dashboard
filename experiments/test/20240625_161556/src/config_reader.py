"""
Explanation: This module contains a class that reads the config_main file

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
# General libraries
import os
import json
import torch

# Costum libraries
import config_main
from assistant_classes import PathProcessor, Writer

class ConfigReader():
    def __init__(self):
        self.pp = PathProcessor()
        self.config_file = self.read_json('config_main.json') # NOTE: do NOT change this name.
        self.directories_dict = self.read_directories_config()
        self.device_dict = self.read_device_config()
        self.data_dict = self.read_data_config()
        self.data_preprocessing_dict = self.read_data_preprocessing_config()
        self.optimization_dict = self.read_optimization_config()
        self.transfer_learning_dict = self.read_transfer_learning_config()
        self.optuna_dict = self.read_optuna_config()
        
        
    def read_json(self, json_file_name):
        try:
            with open (json_file_name, 'r') as config_name:
                config_file = json.load(config_name)
                return config_file

        except FileNotFoundError:
            print(f'Warning: the JSON file is not available')
            config_file = config_main.main_dict
            config_main.make_json_file(config_file)
            print(f'The JSON file (config_main.json) has beeen made.')
            return config_file
        
        except Exception as e:
            raise f'Error: The config file cannot be red.'

        
    def read_directories_config(self):
        
        # Seperate the main directory dict and also folder names
        dir_dict =  self.config_file['directories']
        root_dir = dir_dict['root_dir']
        folder_names = dir_dict['folder_name']
        experiment_names = dir_dict['experiments']
        
        # Create the paths to the basic directions
        folder_paths_dict = self.pp.make_folder_dirs(root_dir, folder_names)
        exp_paths_dict = self.pp.make_experiment_dirs(experiment_names)
        
        # Make the path dict
        paths_dict =  folder_paths_dict.copy()
        paths_dict.update(exp_paths_dict)
        
        # Create all the unavailable folders
        self.pp.create_unavailable_dir(paths_dict)

        # Add the directory dict to the main directory dict
        dir_dict['directories'] = paths_dict
        
        return dir_dict


    def read_device_config(self):
        device_dict = self.config_file['device']
        
        # Add some torch parameters to the device
        device_dict['torch_version'] = torch.__version__
        device_dict['gpu_condition'] = torch.cuda.is_available()
        device_dict['pin_memory'] = True if device_dict['data_loader']['num_workers'] > 0 else False
        device_dict['device'] = torch.device('cuda') if device_dict['gpu_condition'] else torch.device('cpu')

        return device_dict


    def read_optuna_config(self):
        return self.config_file['optuna']
        
    def read_data_config(self):
        return self.config_file['data']

    def read_data_preprocessing_config(self):
        # print(self.config_file['data_preprocessing'])
        return self.config_file['data_preprocessing']
        
    def read_optimization_config(self):
        return self.config_file['optimization']
    
    def read_model_config(self):
        return self.config_file['model'] 

    def read_transfer_learning_config(self):
        return self.config_file['transfer_learning'] 

    def read_training_config(self):
        return self.config_file['training'] 

    def read_plotting_config(self):
        return self.config_file['plotting'] 


if __name__ == '__main__':
    config_reader_obj = ConfigReader()
    writer_obj = Writer()
    writer_obj.make_json_file(config_reader_obj.optuna_dict, 'optuna_dict.json')

    