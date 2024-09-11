"""
Foreword: The purpose of this pipeline is to make an infrastructure for some of the deep learning projects of the group. 
          I will complete it step by step and mainly based on different projects. 
Explanation: .....
last revision: ...  

NOTE: CHANGE THE STRUCTURE OF THE DBP PIPELINE TO THIS STRUCTURE TO BE COMPATIBLE WITH THIS.
NOTE: Add a multi-model that also reads a group of the features.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""

# General libraries
import os
import time
import torch
import optuna
import plotly
import joblib
import shutil
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from monai.utils import set_determinism
from monai.metrics import MSEMetric, ROCAUCMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
)

# Costume Libraries
from general_initializer import *
from div.aug_mix import AugMix
from div.train_val import TrainValTest
from config_reader import ConfigReader
from div.data_loader import DataLoaderMaker
from optuna_optimizer import OptunaOptimizer
from assistant_classes import GeneralFunctions, Writer, Plotter


# global ijk

class Main():

    def __init__(self):

        self.am_obj = AugMix()
        self.writer_obj = Writer()
        self.plotter_obj = Plotter()
        self.config = ConfigReader()
        self.tvt_obj = TrainValTest()
        self.oo_obj = OptunaOptimizer()
        self.dl_obj = DataLoaderMaker()
        self.df_obj = GeneralFunctions()
        self.mi_obj = ModelInitializer()
        self.wi_obj = WeightInitializer()
        self.si_obj = SchedularInitializer()
        self.init_obj = GeneralInitializer()
        self.lfi_obj = LossFuncInitializer()
        self.oi_obj = OptimizerInitializer()
        self.lri_obj = LearningRateInitializer()

        

        self.cv_dict = self.config.data_dict['cross_validation']

    def main(self, trial, param_dict):
        self.df_obj.set_seed(param_dict)
        trial_parameters = self.oo_obj.suggest_hyperparameters(trial, param_dict)
        logger = self.init_obj.initialize_logger(trial_parameters)
        train_dict, val_dict, test_dict = self.dl_obj.get_files(logger, trial_parameters)
        
        # Initialize result lists
        train_loss_list = list()
        val_loss_list = list()
        test_loss_list = list()
        
        # Start recording time
        start = time.time()

        # Implement Cross-Validation
        for fold in range(self.cv_dict['cv_folds']):
            # globals()['ijk'] += 1

            logger.my_print('Fold: {}'.format(fold))
            trial_parameters = self.init_obj.initialize_main_stream(fold, trial_parameters)
            
            if trial_parameters['cv_folds'] > 1:
                
                # If fold is equal to zero find the index of the data for each fold
                if fold == 0:
                    cv_idx_list = self.dl_obj.get_cv_index_list(logger, trial_parameters, train_dict, val_dict) # NOTE:This part should change!!!

                # Extract the data for each fold (train and validation)
                train_dict, val_dict = self.dl_obj.load_fold_data(fold, cv_idx_list, train_dict, val_dict)
            
            # Make the DataLoaders for this fold
            dataloader_dict = self.dl_obj.make_dataloaders(logger, train_dict, val_dict, test_dict, trial_parameters, fold)
            
            # Initialize the model
            model = self.mi_obj.get_model(trial_parameters)
            logger.my_print(model)
            model.to(device=trial_parameters['device'])
            
            # Initialize the weights
            model = self.wi_obj.get_initializer(model, trial_parameters, logger)
            
            # Get model summary
            self.mi_obj.save_model_summary(model, trial_parameters)
            
            # Initialize loss function
            loss_function = self.lfi_obj.get_loss_function(trial_parameters, logger)
            
            # Initialize optimizer
            optimizer = self.oi_obj.get_optimizer(model, trial_parameters, logger)
            
            # Initialize learning rate
            trial_parameters, optimizer = self.lri_obj.get_learning_rate(model, dataloader_dict, optimizer, loss_function, 
                                                                               trial_parameters, self.am_obj, logger)
            
            # Initialize scheduler
            scheduler = self.si_obj.get_scheduler(optimizer, trial_parameters, logger)
            
            # Train
            trial_parameters = self.tvt_obj.train(dataloader_dict, model, loss_function, optimizer, scheduler, trial_parameters, self.am_obj, 
                                            self.lri_obj, self.oi_obj, self.plotter_obj, self.writer_obj,logger)
            
            # Test (BEST epoch model)
            train_loss_list, val_loss_list, test_loss_list = self.tvt_obj.test(dataloader_dict, model, loss_function, trial_parameters, 
                                                                            train_loss_list, val_loss_list, test_loss_list, self.writer_obj,logger)
            
            # Save the results
            self.writer_obj.save_epoch_results(trial_parameters, train_loss_list, val_loss_list, test_loss_list, fold,logger)

        end = time.time()
        logger.my_print(F"Elapsed time: {round(end - start, trial_parameters['nr_of_decimals'])} seconds.")
        logger.my_print('DONE!')
        logger.close()

        # Optuna objectives to minimize (Loss value)
        return np.mean(val_loss_list)             


if __name__ == '__main__':
    main_obj = Main()
    main_obj.main()
