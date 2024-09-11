"""
Explanation: This modules is used for initializing Optuna package. 
It will be served as the first point in one of the arms of the factory design.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
# General libraries
import os
import json
import optuna
import joblib
import wandb
from datetime import datetime
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Costume Libraries
from config_reader import ConfigReader
from assistant_classes import PathProcessor, Writer
from optuna_optimizer import OptunaOptimizer
from main import Main

# global ijk

class OptunaInitializer():

    def __init__(self):
        
        self.pp = PathProcessor()
        self.main_obj = Main()
        self.writer_obj = Writer()
        self.config_obj = ConfigReader()
        self.optuna_optimizer_obj = OptunaOptimizer()
        
        self.directories_dict = self.config_obj.directories_dict
        self.optuna_dict = self.config_obj.optuna_dict
        self.device_dict = self.config_obj.device_dict

        self.exp_root_dir = self.directories_dict['experiments']['root_dir']
        self.optuna_path_pickles = self.directories_dict['experiments']['optuna_path_pickles']
        self.optuna_path_figures = self.directories_dict['directories']['optuna_figures_dir']
        self.optuna_study_name = self.directories_dict['file_names']['optuna_study_name']
        self.optuna_sampler_name = self.directories_dict['file_names']['optuna_sampler_name']
        self.optuna_n_trials = self.optuna_dict['optuna_n_trials'] 

    def extract_datetime(self, folder_name):
        date_str = folder_name.split('_')[0]
        time_str = folder_name.split('_')[1]
        return datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

    def find_latest_exp_path(self):
        exp_dir_list = os.listdir(self.exp_root_dir)

        # Extract datetimes and sort folders
        folders_with_datetimes = [(folder, self.extract_datetime(folder)) for folder in exp_dir_list]
        folders_with_datetimes.sort(key=lambda x: x[1], reverse=True)

        # Get the latest folder
        latest_file = folders_with_datetimes[1][0]
        
        return os.path.join(self.exp_root_dir, latest_file, self.optuna_path_pickles)

        
    def check_iteration_num(self):
        
        # Find the latest experiment
        optuna_path_pickles = self.find_latest_exp_path()
        optuna_file_study_list = [x for x in os.listdir(optuna_path_pickles) if self.optuna_study_name in x]
        optuna_file_sampler_list = [x for x in os.listdir(optuna_path_pickles) if self.optuna_sampler_name in x]
        
        print('HIIIIIIIIIIIIIIII',optuna_path_pickles,  optuna_file_study_list)
                
        if len(optuna_file_study_list) > 0 and len(optuna_file_sampler_list) > 0:
            
            # Find latest study, and add 1 for the next study run
            optuna_study_run_nr = max([int(x.split('_')[0]) for x in optuna_file_study_list]) + 1
            optuna_sampler_run_nr = max([int(x.split('_')[0]) for x in optuna_file_sampler_list]) + 1
            assert optuna_study_run_nr == optuna_sampler_run_nr

            optuna_in_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr - 1) 
                                                + self.optuna_study_name)
            optuna_in_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr - 1) 
                                                + self.optuna_sampler_name)

            print('Resuming previous study: {}'.format(optuna_in_file_study))
        
            optuna_sampler = joblib.load(optuna_in_file_sampler)
            optuna_study = joblib.load(optuna_in_file_study)
        else:
            optuna_study_run_nr = 0
            optuna_sampler_run_nr = 0
            
            optuna_sampler = optuna.samplers.TPESampler(seed=123)
            optuna_study = optuna.create_study(sampler=optuna_sampler, direction='minimize') # Think about this parameters
        
        return optuna_study, optuna_study_run_nr, optuna_sampler_run_nr, optuna_path_pickles


    def initialize_param_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.initialize_param_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


    def initializer(self):
        # globals()['ijk'] = 0

        # check whether the 
        # for p in [self.exp_root_dir, self.optuna_path_pickles, self.optuna_path_figures]:
        #     self.pp.make_folder(p)

        optuna_study, optuna_study_run_nr, optuna_sampler_run_nr, optuna_path_pickles = self.check_iteration_num()

        # Create study
        optuna_out_file_study = os.path.join(optuna_path_pickles, '{}_'.format(optuna_study_run_nr) + self.optuna_study_name)
        optuna_out_file_sampler = os.path.join(optuna_path_pickles, '{}_'.format(optuna_sampler_run_nr) + self.optuna_sampler_name)
        joblib.dump(optuna_study, optuna_out_file_study)
        joblib.dump(optuna_study.sampler, optuna_out_file_sampler)

        # Create main_parameters dictionary
        main_parameters_dict = self.initialize_param_dict(self.config_obj.config_file)
        self.writer_obj.make_json_file(main_parameters_dict, 'main_parameters.json')
        
        # Run hyperparameter tuning
        # optuna_study.optimize(self.optuna_optimizer_obj.main, n_trials=self.optuna_n_trials, param_dict=main_parameters_dict)
        optuna_study.optimize(
                              lambda trial: self.main_obj.main(trial, param_dict=main_parameters_dict,
                                                               optuna_study=optuna_study,
                                                               optuna_out_file_study=optuna_out_file_study, 
                                                               optuna_out_file_sampler=optuna_out_file_sampler
                                                               ), 
                              n_trials=self.optuna_n_trials
                             )
        
        joblib.dump(optuna_study, optuna_out_file_study)
        joblib.dump(optuna_study.sampler, optuna_out_file_sampler)


if __name__ == '__main__':
    initializer_obj = OptunaInitializer()
    initializer_obj.initializer()
