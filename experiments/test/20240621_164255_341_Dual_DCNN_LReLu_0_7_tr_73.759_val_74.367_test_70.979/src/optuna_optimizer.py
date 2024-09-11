"""
Explanation: This module contains Optuna optimizer and will produce the values and return them?

Problems:
1. NOTE: change the following method 'suggest_linear_units_list' since it is not working poperly (read its note.)
2. NOTE: DESIGN OF THE LINEAR LAYERS SHOULD CHANGE, IT IS QUITE BAD, AND IT IS SUSEPTIBLE TO ERRORS.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
# General Libraries
import os
import torch
import random
import numpy as np
from monai.metrics import MSEMetric, ROCAUCMetric
from monai.transforms import (
                              Activations,
                              AsDiscrete,
                             )
# Costume Libraries
from config_reader import ConfigReader
from assistant_classes import PathProcessor,Writer

class OptunaOptimizer():
    
    def __init__(self):
        # self.trial = trial
        self.pp = PathProcessor()
        self.config_obj = ConfigReader()
        self.writer_obj = Writer()
        self.optuna_dict = self.config_obj.optuna_dict


    def suggest_batch_size_list(self, name, parameters, trial): 
        parameters['batch_size_idx'] = trial.suggest_int('batch_size_idx', 0, len(self.optuna_dict[name]) - 1, 1) 
        parameters['batch_size'] = self.optuna_dict[name][parameters['batch_size_idx']]
        return parameters


    def suggest_linear_units_size(self, name, parameters, trial):
        parameters['linear_units_size_idx'] = trial.suggest_int('linear_units_size_idx', 0, len(self.optuna_dict[name]) -1, 1) 
        parameters['linear_units_size'] = self.optuna_dict[name][parameters['linear_units_size_idx']]
        
        return parameters


    def suggest_linear_units_list(self, name, parameters, trial):
        # NOTE: CHANGE THIS. I DO NOT LIKE IT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # THE LAYERS HAVE THE THE SAME NUMBER OF NODES.
        linear_units = list()
        for _ in range(parameters['linear_units_size']): 
            linear_units_idx = trial.suggest_int('linear_units_idx', 0, len(self.optuna_dict[name]) -1, 1) #############
            linear_units_j = self.optuna_dict[name][linear_units_idx]
            linear_units.append(linear_units_j)
        
        parameters['linear_units'] = linear_units
        return parameters


    def suggest_dropout_p_j(self, name, parameters, trial):
        dropout_p = list()

        for _ in range(parameters['linear_units_size']): 
            dropout_p_j = trial.suggest_float('dropout_p_j', self.optuna_dict[name][0], self.optuna_dict[name][1])
            dropout_p.append(dropout_p_j)
        
        parameters['dropout_p_j'] = dropout_p
        return parameters


    def suggest_clinical_variables_linear_units_size(self, name, parameters, trial):
        
        parameters['clinical_variables_linear_units_size_idx'] = \
            trial.suggest_int('clinical_variables_linear_units_size_idx', 0, len(self.optuna_dict[name]) -1, 1) 
    
        parameters['clinical_variables_linear_units_size'] = \
            self.optuna_dict[name][parameters['clinical_variables_linear_units_size_idx']]
        
        return parameters        


    def suggest_clinical_variables_linear_units_list(self, name, parameters, trial):
        # NOTE: CHANGE THIS. I DO NOT LIKE IT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
        # THE LAYERS HAVE THE THE SAME NUMBER OF NODES.
        clinical_variables_linear_units = list()
        for _ in range(parameters['clinical_variables_linear_units_size']): 
            clinical_variables_linear_units_idx = trial.suggest_int('clinical_variables_linear_units_idx', 0, len(self.optuna_dict[name]) -1, 1) #############
            clinical_variables_linear_units_j = self.optuna_dict[name][clinical_variables_linear_units_idx]
            clinical_variables_linear_units.append(clinical_variables_linear_units_j)
        
        parameters['clinical_variables_linear_units'] = clinical_variables_linear_units
        return parameters


    def suggest_clinical_variables_dropout_p_j(self, name, parameters, trial):
        clinical_variables_dropout_p = list()

        for _ in range(parameters['clinical_variables_linear_units_size']): 
            dropout_p_j = trial.suggest_float('clinical_variables_dropout_p_j', self.optuna_dict[name][0], self.optuna_dict[name][1])
            clinical_variables_dropout_p.append(dropout_p_j)
        
        parameters['clinical_variables_dropout_p_j'] = clinical_variables_dropout_p
        return parameters
            

    def suggest_filters_list(self, name, parameters, trial):
        parameters['filter_size_idx'] = trial.suggest_int('filter_size_idx', 0, len(self.optuna_dict[name]) - 1, 1)###############
        parameters['filter_size'] = self.optuna_dict[name][parameters['filter_size_idx']]
        return parameters


    def suggest_features_dl_list(self, name, parameters, trial):
        parameters['features_dl_idx'] = trial.suggest_int('features_dl_idx', 0, len(self.optuna_dict[name]) - 1, 1) #############################
        parameters['features_dl'] = self.optuna_dict[name][parameters['features_dl_idx']]
        return parameters         

    def suggest_optuna_step_size_up(self, name, parameters, trial):
        parameters['optuna_step_size_up'] = [16*parameters['batch_size'], 32*parameters['batch_size'], 64*parameters['batch_size']]
        parameters['step_size_up_idx'] = trial.suggest_int('step_size_up_idx', 0, len(parameters['optuna_step_size_up']) - 1, 1)
        parameters['step_size_up'] = parameters['optuna_step_size_up'][parameters['step_size_up_idx']]

        return parameters


    def suggest_float_num(self, name, values, parameters, trial):    
        parameters[name] = trial.suggest_float(name, values[0], values[1])
        return parameters


    def suggest_int_num(self, name, values, parameters, trial):
        parameters[name] = trial.suggest_int(name, values[0], values[1])
        return parameters


    def suggest_category(self, name, values, parameters, trial):
        parameters[name] = trial.suggest_categorical(name, values)
        return parameters

    
    def suggest_boolian(self, name, values, parameters, trial):  
        parameters[name] = trial.suggest_categorical(name, values)
        return parameters    


    def suggest_ranges(self, name, parameters, trial):

        ranges = self.optuna_dict[name]
        
        for key, values in ranges.items():
            if isinstance(values[0], float):
                parameters = self.suggest_float_num(key, values, parameters, trial)
                
            elif isinstance(values[0], bool):
                parameters = self.suggest_boolian(key, values, parameters, trial)   

            elif isinstance(values[0], int):
                parameters = self.suggest_int_num(key, values, parameters, trial)
            
            elif isinstance(values[0], str):
                parameters = self.suggest_category(key, values, parameters, trial)
            
        return parameters


    def suggest_kernel_stride(self, name, parameters, trial):
        
        kernel_sizes_i_tmp = [self.optuna_dict[name][1]]

        kernel_sizes = list()
        strides = list()

        for i in range(parameters['n_layers']):  
            # Note: `min(kernel_sizes_i_tmp)` is used to make sure that kernel_size does not become larger for later layers
            # min(..., 128 / (2**(i+1)) is used to make sure that kernel_size is not larger than feature_map size
            if round(128 / (2**(i+1))) >= self.optuna_dict[name][0]:
                parameters[f'kernel_sizes_{i}'] = trial.suggest_int(f'kernel_sizes_{i}', 
                                                    self.optuna_dict[name][0], 
                                                    min(kernel_sizes_i_tmp))
            # If the feature map size is smaller than `optuna_min_kernel_size` (e.g.,: 2x2 vs. 3), then use 
            # kernel_size smaller than `optuna_min_kernel_size`.
            # elif: round(128 / (2**(i+1))) < optuna_min_kernel_size:
            else: 
                parameters[f'kernel_sizes_{i}'] = trial.suggest_int('kernel_sizes_{}'.format(i), 
                                                    1, min(min(kernel_sizes_i_tmp), 128 / (2**(i+1)))) 

            kernel_sizes_i_tmp.append(parameters[f'kernel_sizes_{i}'] )
            kernel_sizes.append([1, parameters[f'kernel_sizes_{i}'] , parameters[f'kernel_sizes_{i}'] ])

            if i == parameters['n_layers'] - 1:
                # NOTE: We can make this part also costumize, but for preveting errors I will fix this part
                parameters['stride_value_last_layer'] = trial.suggest_int('stride_value_last_layer', 1, 2)  
                strides.append([parameters['stride_value_last_layer']] * 3)
            else:
                strides.append([2] * 3)
                
        parameters['kernel_sizes'] = kernel_sizes
        parameters['strides'] = strides

        return parameters


    def suggest_loss_weights(self, name, parameters, trial):
        keys_with_loss = [key for key in parameters.keys() if 'loss' in key]
        parameters[name] = [parameters[key] for key in keys_with_loss]

        return parameters

    
    def correct_parameters(self, parameters, trial):
        
        if 'filter_size' in parameters.keys() and 'n_layers' in parameters.keys():
            parameters['filters'] = parameters['filter_size'][:parameters['n_layers']]
        
        if 'kernel_size_range' in self.optuna_dict.keys():
            name = 'kernel_size_range'
            parameters = self.suggest_kernel_stride(name, parameters, trial)
        
        if 'use_momentum' not in parameters.keys() or  parameters['use_momentum'] == False:
            parameters['momentum'] = 0
        
        if any('loss' in key for key in parameters.keys()):
            name = 'loss_weights'
            parameters = self.suggest_loss_weights(name, parameters, trial)

        return parameters

    def add_classes(self,trial_parameters, trial):
        trial_parameters['sigmoid_act'] = torch.nn.Sigmoid() # Is the activation function here
        trial_parameters['softmax_act'] = Activations(softmax=True)
        trial_parameters['to_onehot'] = AsDiscrete(to_onehot=trial_parameters['num_ohe_classes']) # Determine the descrete classes
        trial_parameters['mse_metric'] = MSEMetric() # mean squared error
        trial_parameters['auc_metric'] = ROCAUCMetric() # AUC-ROC
        trial_parameters['excl'] = ''

        return trial_parameters

    def suggest_hyperparameters(self, trial, param_dict):
        trial_parameters = param_dict.copy() # This should go to the initializer

        for key in self.optuna_dict.keys():
            
            func_name = f'suggest_{key}'

            try:
                func = getattr(self, func_name)
                trial_parameters = func(key, trial_parameters, trial)

            except AttributeError:
                print(f'Method {func_name} not found in OptunaOptimizer.')

            except Exception as e:
                print(f'An error occurred while calling {func_name}: {e}')            
        
        trial_parameters = self.correct_parameters(trial_parameters, trial)
        trial_parameters = self.add_classes(trial_parameters, trial)
            
        print(trial_parameters)  

        # Save trial_parameters in the experiment folder
        self.writer_obj.make_json_file(trial_parameters, os.path.join(trial_parameters['exp_dir'], f'trial_params_trial{trial.number}.json'))

        return trial_parameters

    




if __name__ == '__main__':
    optimizer_obj = OptunaOptimizer()
    trial_parameters = optimizer_obj.suggest_hyperparameters()