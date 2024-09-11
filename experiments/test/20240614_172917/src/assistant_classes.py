"""
Explanation: 
This module contains all the classes used in the whole pipeline.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""

# General libraries
import os
import json
import torch
import shutil
import random
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Any
from datetime import datetime
import matplotlib.pyplot as plt
from collections.abc import Callable, Iterable, Sequence


_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark


class Reader():
    pass

class Writer():

    def save_epoch_results(self, tp, train_list, val_list, test_list, fold,logger):
        """
        Explanation: This method is used to save and transfer the result of each fold and experiment.
        """

        src_folder_dir = tp['exp_dir']

        # Generate a three-digit random number
        rn = random.randint(100, 999)

        # Rename folder
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_value = round(train_list[fold], tp['nr_of_decimals'])
        val_value = round(val_list[fold], tp['nr_of_decimals'])
        test_value = round(test_list[fold], tp['nr_of_decimals'])

        dst_folder_name = f"{exp_name}_{rn}_{tp['model_name']}_{fold}_{tp['best_epoch']}_tr_{train_value}_val_{val_value}_test_{test_value}"
        dst_folder_dir = os.path.join(tp['exp_root_dir'], dst_folder_name)

        # If this is the last fold, add the mean values to the folder name
        if fold+1 == tp['cv_folds']:
            train_mean = round(np.mean(train_list), tp['nr_of_decimals'])
            val_mean = round(np.mean(val_list), tp['nr_of_decimals'])
            test_mean = round(np.mean(test_list), tp['nr_of_decimals'])

            # Add the mean values to the name
            dst_folder_dir = f"{dst_folder_dir}_avg_tr_{train_mean}_val_{val_mean}_test_{test_mean}"
            dst_folder_name = f"{dst_folder_name}_avg_tr_{train_mean}_val_{val_mean}_test_{test_mean}"

        if not os.path.exists(dst_folder_dir):
            os.makedirs(dst_folder_dir)

        # Copy all the information to the new folder.
        shutil.copytree(src_folder_dir, dst_folder_dir, dirs_exist_ok=True)
        # shutil.copy2(src_folder_dir, dst_folder_dir)
        logger.my_print(f"Experiment has been saved in {dst_folder_name} folder.")


    def save_predictions(self, tp, y_pred, y, dl_mode, inf_dict, mode, logger):
        """
        Save prediction and corresponding true labels to csv.
        """

        # Make the lists
        y_list = [element for element in y]    
        y_pred_list = [element for element in y_pred]    
        mode_list = [mode] * len(y_pred)
        patient_ids = [element['pat_id'] for element in inf_dict]

        # Print outputs
        logger.my_print(f"Model_name: {tp['model_name']}.")
        logger.my_print(f'patient_ids: {patient_ids}.')
        logger.my_print(f'y_pred_list: {y_pred_list}.')
        logger.my_print(f'y_true_list: {y_list}.')

        # Convert to CPU
        y_pred = [x.cpu().numpy() for x in y_pred_list]
        y_true = [x.cpu().numpy() for x in y_list]

        # Save to DataFrame
        num_cols = y_pred[0].shape[0]
        df_patient_ids = pd.DataFrame(patient_ids, columns=['PatientID'])
        df_y_pred = pd.DataFrame(y_pred, columns=['pred_{}'.format(c) for c in range(num_cols)])
        df_y_true = pd.DataFrame(y_true, columns=['true_{}'.format(c) for c in range(num_cols)])
        df_mode = pd.DataFrame(mode_list, columns=['Mode'])
        df_y = pd.concat([df_patient_ids, df_y_pred, df_y_true, df_mode], axis=1)

        # Save to file
        df_y.to_csv(os.path.join(tp['exp_dir'], tp['i_outputs_csv'].format(i=f"{tp['model_name']}_{dl_mode}")), sep=';', index=False)



    def make_json_file(self, main_dict, name):
        # Specify the file path
        file_path = name # NOTE: Do NOT change this name

        # Save the dictionary as a JSON file
        with open(file_path, 'w') as json_file:
            json.dump(main_dict, json_file, indent=4, default=self.convert_obj)


    def convert_obj(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        # raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")  
         
        if isinstance(obj, dict):
            return {k: self.convert_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_obj(v) for v in obj]

        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj) 
        
    


class Logger:
    def __init__(self, output_filename=None):
        logging.basicConfig(filename=output_filename, format='%(asctime)s - %(message)s', level=logging.INFO,
                            filemode='w')

    def my_print(self, message, level='info'):
        """
        Manual print operation.

        Args:
            message: input string.
            level: level of logging.

        Returns:

        """
        if level == 'info':
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        elif level == 'exception':
            print_message = 'EXCEPTION: {}'.format(message)
            logging.exception(print_message)
        elif level == 'warning':
            print_message = 'WARNING: {}'.format(message)
            logging.warning(print_message)
        else:
            print_message = 'INFO: {}'.format(message)
            logging.info(print_message)
        print(print_message)

    def close(self):
        logging.shutdown()


class PathProcessor():

    def prepare_data_nrrd_for_CT(self, data_dir, patient_ids):
        pct_paths = []
        rct_paths = []
        reg_pos = []
        pat_id_list = []

        # Load the JSON data
        with open(os.path.join(data_dir, 'file_info.json'), 'r') as json_file:  
            nrrd_info = json.load(json_file)
        
        for patient in nrrd_info:
            if patient['id'] in patient_ids:
                for examination_detail in patient['examination_details']:
                    # Paths for planning CT and repeated CT
                    planning_ct_path = os.path.join(data_dir, patient['id'], examination_detail['planningCT_filename'])

                    planning_ct_path = planning_ct_path.replace('//zkh/AppData/data/shahpouriz/Processed_CT/nrrd/proton' ,
                                                  '/data/bahrdoh/CT_Model/Test/proton')
                    repeated_ct_path = os.path.join(data_dir, patient['id'], examination_detail['repeatedCT_filename'])                

                    repeated_ct_path = repeated_ct_path.replace('//zkh/AppData/data/shahpouriz/Processed_CT/nrrd/proton' ,
                                                  '/data/bahrdoh/CT_Model/Test/proton')          
                    
                    # Append paths if they exist
                    if os.path.exists(planning_ct_path) and os.path.exists(repeated_ct_path):
                        pct_paths.append(planning_ct_path)
                        rct_paths.append(repeated_ct_path)
                        
                        # Append registration position
                        reg_pos.append([
                            examination_detail['final_translation_coordinate']['z'],
                            examination_detail['final_translation_coordinate']['y'],
                            examination_detail['final_translation_coordinate']['x']
                        ])
                        pat_id_list.append(patient['id'])
        
        reg_pos_array = np.array(reg_pos, dtype=np.float32)

        return pct_paths, rct_paths, reg_pos_array, pat_id_list

    def extract_patient_ids(self, data_path):
        """
        List all directories in the base_directory.
        Each directory represents a patient.
        """
        try:
            patient_folders = [name for name in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path, name))]
            return patient_folders
        except FileNotFoundError:
            print(f"Directory {data_path} was not found.")
            return []


    def make_folder_dirs(self, root_dir, folder_names):
        
        # Add a directory dictionary to the dir_dict
        folder_paths_dict = {}
        
        # Check wherher the main root is available
        self.make_folder(root_dir)
        
        # Add base directories to the directory dict
        folder_paths_dict['root_dir'] = root_dir
        folder_paths_dict['models_dir'] = os.path.join(root_dir, folder_names['models_folder'])
        folder_paths_dict['optimizers_dir'] = os.path.join(root_dir, folder_names['optimizers_folder'])
        folder_paths_dict['data_preproc_dir'] = os.path.join(root_dir, folder_names['data_preproc_folder'])
        folder_paths_dict['save_root_dir'] = os.path.join(root_dir, folder_names['save_root_folder'])
        # folder_paths_dict['data_dir'] = os.path.join(root_dir, folder_names['data_folder'])
        
        return folder_paths_dict


    def make_experiment_dirs(self, experiments_dict):

        exp_paths_dict = {}
        
        # Create paths to the experiment directions
        self.make_folder(experiments_dict['root_dir'])

        # Make the experiment name
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exp_paths_dict['exp_root_dir'] = experiments_dict['root_dir']
        exp_paths_dict['exp_dir'] = os.path.join(experiments_dict['root_dir'], exp_name)
        exp_paths_dict['exp_src_dir'] = os.path.join(exp_paths_dict['exp_dir'], experiments_dict['src_folder'])
        exp_paths_dict['exp_models_dir'] = os.path.join(exp_paths_dict['exp_src_dir'], experiments_dict['models_folder'])
        exp_paths_dict['exp_optimizers_dir'] = os.path.join(exp_paths_dict['exp_src_dir'], experiments_dict['optimizers_folder'])
        exp_paths_dict['exp_data_preproc_dir'] = os.path.join(exp_paths_dict['exp_src_dir'], experiments_dict['data_preproc_folder'])
        exp_paths_dict['exp_figures_dir'] = os.path.join(exp_paths_dict['exp_dir'], experiments_dict['figures_folder'])
        exp_paths_dict['optuna_figures_dir'] = os.path.join(exp_paths_dict['exp_dir'], experiments_dict['figures_folder'])
        exp_paths_dict['optuna_pickles_dir'] = os.path.join(exp_paths_dict['exp_dir'], experiments_dict['optuna_path_pickles'])

        return exp_paths_dict


    def create_unavailable_dir(self, paths_dict):
        for key, value in paths_dict.items():
            self.make_folder(value)


    def make_folder(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)


    def copy_folder(self, src, dst):

        try:
            shutil.copytree(src, dst)

        except FileExistsError:
            # Handle the situation when the destination directory already exists
            print(f"Destination directory '{dst}' already exists. Skipping copy.")
        
        except NotADirectoryError:
            # Handle the situation when the destination directory already exists
            print(f"The directory name is invalid {src}")        

    def copy_file(self, src, dst):
        shutil.copy(src, dst)


class GeneralFunctions():
    def __init__(self):
        pass

    def issequenceiterable(self, obj: Any) -> bool:
        """
        Determine if the object is an iterable sequence and is not a string.
        """
        try:
            if hasattr(obj, "ndim") and obj.ndim == 0:
                return False  # a 0-d tensor is not iterable
        except Exception:
            return False
        return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))

    def ensure_tuple(self, vals: Any, wrap_array: bool = False) -> tuple:
        """
        Returns a tuple of `vals`.

        Args:
            vals: input data to convert to a tuple.
            wrap_array: if `True`, treat the input numerical array (ndarray/tensor) as one item of the tuple.
                if `False`, try to convert the array with `tuple(vals)`, default to `False`.

        """
        if wrap_array and isinstance(vals, (np.ndarray, torch.Tensor)):
            return (vals,)
        return tuple(vals) if self.issequenceiterable(vals) else (vals,)

    def set_determinism(self,
        seed: int | None = np.iinfo(np.uint32).max,
        use_deterministic_algorithms: bool | None = None,
        additional_settings: Sequence[Callable[[int], Any]] | Callable[[int], Any] | None = None,
    ) -> None:
        """
        Set random seed for modules to enable or disable deterministic training.

        Args:
            seed: the random seed to use, default is np.iinfo(np.int32).max.
                It is recommended to set a large seed, i.e. a number that has a good balance
                of 0 and 1 bits. Avoid having many 0 bits in the seed.
                if set to None, will disable deterministic training.
            use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
            additional_settings: additional settings that need to set random seed.

        Note:

            This function will not affect the randomizable objects in :py:class:`monai.transforms.Randomizable`, which
            have independent random states. For those objects, the ``set_random_state()`` method should be used to
            ensure the deterministic behavior (alternatively, :py:class:`monai.data.DataLoader` by default sets the seeds
            according to the global random state, please see also: :py:class:`monai.data.utils.worker_init_fn` and
            :py:class:`monai.data.utils.set_rnd`).
        """
        if seed is None:
            # cast to 32 bit seed for CUDA
            seed_ = torch.default_generator.seed() % (np.iinfo(np.uint32).max +1)
            torch.manual_seed(seed_)
        else:
            seed = int(seed) % (np.iinfo(np.uint32).max +1)
            torch.manual_seed(seed)

        global _seed
        _seed = seed
        random.seed(seed)
        np.random.seed(seed)

        if additional_settings is not None:
            additional_settings = self.ensure_tuple(additional_settings)
            for func in additional_settings:
                func(seed)

        if torch.backends.flags_frozen():
            warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
            torch.backends.__allow_nonbracketed_mutation_flag = True

        if seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # restore the original flags
            torch.backends.cudnn.deterministic = _flag_deterministic
            torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
        if use_deterministic_algorithms is not None:
            if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
                torch.use_deterministic_algorithms(use_deterministic_algorithms)
            elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
                torch.set_deterministic(use_deterministic_algorithms)
            else:
                warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")


    def set_seed(self, param_dict):
        # Set seed for reproducibility
        torch.manual_seed(seed=param_dict['seed'])
        self.set_determinism(seed=param_dict['seed'])
        random.seed(a=param_dict['seed'])
        np.random.seed(seed=param_dict['seed'])
        torch.backends.cudnn.benchmark = param_dict['cudnn_benchmark'] # For now, it is equal to False, but it may change to True in the future.  


class Plotter():
    """
    Explanation: This class is used to plot important variables.
    """

    def plot_values(self, result_dict, best_epoch, tp):
        """
        Create and save line plot of a list of loss values (per epoch).

        """
        y_label_list = list(result_dict.keys())
        legend_list = [['Training', 'Internal validation'] for _ in range(len(y_label_list) - 2)] + [None, None]

        fig, ax = plt.subplots(nrows=len(y_label_list), ncols=1, figsize=tuple(tp['figsize']))
        ax = ax.flatten()

        for i, (y_label, y_values) in enumerate(result_dict.items()):
            for y_i in y_values:
                epochs = [e + 1 for e in range(len(y_i))]
                ax[i].plot(epochs, y_i)
            ax[i].set_ylim(bottom=0)
            ax[i].set_xlabel('Epoch')
            ax[i].axvline(x=best_epoch, color='red', linestyle='--')
            if y_label is not None:
                ax[i].set_title(y_label)
            if legend_list[i] is not None:
                ax[i].legend(legend_list[i], bbox_to_anchor=(1, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(tp['exp_dir'], tp['results_png']))
        plt.close(fig)