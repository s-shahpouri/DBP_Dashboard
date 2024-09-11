"""
Explanation: This module contains all the classes and methods used to load and make dataLoaders

NOTE:
1. Add exclusion section to 'get_files' function. For now I do NOT need this part since we do NOT have 
any exclusion criteria, but it should be added when I wanted to try Multi-input models.
2. TODO: Make a class for transformers.
3. TODO: To add the multi-input model to this pipeline, the transformer should change!! Be careful about it.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""

# General Libraries
import random
import torch
import numpy as np
from sklearn.model_selection import KFold

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    Spacingd, SpatialPadd, CenterSpatialCropd, RandWeightedCropd,
    EnsureTyped, MapTransform, ToDeviced, Rand3DElasticd)

from monai.data import (
                        Dataset, CacheDataset, PersistentDataset,
                        DataLoader, ThreadDataLoader, ITKReader
                        )
# Costumize Libraries
from config_reader import ConfigReader
from assistant_classes import PathProcessor, Writer
from div.transformer import TransformGenerator

 
class DataLoaderMaker():

    def __init__(self):
        # self.tp = trial_parameters

        self.pp_obj = PathProcessor()        
        self.writer = Writer()


    def get_sampler(self, train_dict, logger, trial_parameters):
        # Shuffle is not necessary for val_dl and test_dl, but shuffle can be useful for plotting random patients in main.py
        # Weighted random sampler
        if trial_parameters['use_sampler']:
            # Source: https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader
            logger.my_print('Using WeightedRandomSampler.')
            shuffle = False

            label_raw_train = np.array([x['label'] for x in train_dict])
            # len(weights) = num_classes
            weights = 1 / np.array([np.count_nonzero(1 - label_raw_train), np.count_nonzero(label_raw_train)])
            samples_weight = np.array([weights[t] for t in label_raw_train])
            samples_weight = torch.from_numpy(samples_weight)
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            shuffle = True
            sampler = None
        
        return shuffle, sampler
        
    def get_dataset_class(self, trial_parameters):
        
        update_dict = None
        # Define Dataset class
        if trial_parameters['dataset_type'] in ['standard', None]:
            ds_class = Dataset
        elif trial_parameters['dataset_type'] == 'cache':
            ds_class = CacheDataset
            update_dict = {'cache_rate': trial_parameters['cache_rate'], 'num_workers': trial_parameters['num_workers']}
        elif trial_parameters['dataset_type'] == 'persistent':
            ds_class = PersistentDataset
            update_dict = {'cache_dir': trial_parameters['cache_dir']}
            self.pp_obj.make_folder(trial_parameters['cache_dir'])
        else:
            raise ValueError(f"Invalid dataset_type: {trial_parameters['dataset_type']}.")   
        
        return ds_class, update_dict     


    def get_dataloader_class(self, trial_parameters):
        # Define DataLoader class
        if trial_parameters['dataloader_type'] in ['standard', None]:
            dl_class = DataLoader
        elif trial_parameters['dataloader_type'] == 'thread':
            dl_class = ThreadDataLoader
        else:
            raise ValueError(f"Invalid dataloader_type: {trial_parameters['dataloader_type']}.")
        
        return dl_class
    
    
    def get_dataloaders(self, train_dict, val_dict, test_dict, train_transforms, val_test_transforms, trial_parameters, logger):
        """
        Construct PyTorch Dataset object, and then DataLoader.

        CacheDataset: caches data in RAM storage. By caching the results of deterministic / non-random preprocessing
            transforms, it accelerates the training data pipeline.
        PersistentDataset: caches data in disk storage instead of RAM storage.
        Source: https://docs.monai.io/en/stable/data.html
        
        NOTE: Batch size can be different and equal to one for validation and test sets.
        """
        train_size = len(train_dict)
        val_size = len(val_dict)
        test_size = len(test_dict)
        
        update_dict = None

        logger.my_print(f"Dataset type: {trial_parameters['dataset_type']}.")
        logger.my_print(f"Dataloader type: {trial_parameters['dataloader_type']}.")

        # Define Dataset class
        ds_class, update_dict = self.get_dataset_class(trial_parameters)

        # Define Dataset function arguments
        train_ds_args_dict = {'data': train_dict, 'transform': train_transforms}
        val_ds_args_dict = {'data': val_dict, 'transform': val_test_transforms}
        test_ds_args_dict = {'data': test_dict, 'transform': val_test_transforms}

        # Update Dataset function arguments based on type of Dataset class
        if update_dict is not None:
            train_ds_args_dict.update(update_dict)
            val_ds_args_dict.update(update_dict)
            test_ds_args_dict.update(update_dict)
        # print('hiiiiiiiiiii', len(train_ds_args_dict['data']))
        # Initialize Dataset
        train_ds = ds_class(**train_ds_args_dict)
        # print('hiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
        val_ds = ds_class(**val_ds_args_dict)
        test_ds = ds_class(**test_ds_args_dict)

        # Get the DataLoader class
        dl_class = self.get_dataloader_class(trial_parameters)

        # Define Dataloader function arguments
        shuffle, sampler = self.get_sampler(train_dict, logger, trial_parameters)

        logger.my_print('Train dataloader arguments.')
        logger.my_print('\tBatch_size: {}.'.format(trial_parameters['batch_size']))
        logger.my_print('\tShuffle: {}.'.format(shuffle))
        logger.my_print('\tSampler: {}.'.format(sampler))
        logger.my_print('\tNum_workers: {}.'.format(trial_parameters['num_workers']))
        logger.my_print('\tDrop_last: {}.'.format(trial_parameters['dataloader_drop_last']))
        
        # Make the dictionaty of the aurguments for the DataLoaders
        train_dl_args_dict = {'dataset': train_ds, 'batch_size': trial_parameters['batch_size'], 'shuffle': shuffle, 'sampler': sampler,
                            'num_workers': trial_parameters['num_workers'], 'drop_last': trial_parameters['dataloader_drop_last']}
        

        val_dl_args_dict = {'dataset': val_ds, 'batch_size': trial_parameters['batch_size'], 'shuffle': False, 'num_workers': int(trial_parameters['num_workers']),
                            'drop_last': False}
        
        test_dl_args_dict = {'dataset': test_ds, 'batch_size': trial_parameters['batch_size'], 'shuffle': False, 'num_workers': int(trial_parameters['num_workers']),
                            'drop_last': False}

        # Initialize DataLoader
        train_dl = dl_class(**train_dl_args_dict) if train_size > 0 else None
        val_dl = dl_class(**val_dl_args_dict) if val_size > 0 else None
        test_dl = dl_class(**test_dl_args_dict) if test_size > 0 else None

        return train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict
        
        
        
    def get_mean_and_std(self, dataloader, trial_parameters):
        """
        Calculate mean and std of the two (CT and RTDOSE) channels in each batch and average them at the end.

        Source: https://xydida.com/2022/9/11/ComputerVision/Normalize-images-with-transform-in-pytorch-dataloader/
        Source: https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
        
        NOTE: Add another approach to calculate mean and STD.
        NOTE: ADJUST THIS FOR GENERAL CASES.
        """
        mean = torch.zeros(2)
        std = torch.zeros(2)

        for batch_data in dataloader:

            plan_input = batch_data[trial_parameters['image_keys'][0]]
            repeat_input = batch_data[trial_parameters['image_keys'][1]]

            mean[0] += plan_input.mean()
            mean[1] += repeat_input.mean()
            
            std[0] += plan_input.std()
            std[1] += repeat_input.std()
                        

        mean.div_(len(dataloader))
        std.div_(len(dataloader))

        return list(mean.numpy()), list(std.numpy())
            
            
    def get_normalization_dataloader(self, trial_parameters, train_dict, val_transforms):
        """
        Construct PyTorch Dataset for computing normalization parameters (mean, std), using training data with
        val_transforms.
        """

        train_size = len(train_dict)
        ds_class, update_dict = self.get_dataset_class(trial_parameters)
        
        # Define Dataset function arguments
        ds_args_dict = {'data': train_dict, 'transform': val_transforms}

        # Update Dataset function arguments based on type of Dataset class
        if update_dict is not None:
            ds_args_dict.update(update_dict)

        # Initialize Dataset
        ds = ds_class(**ds_args_dict)

        # Get the DataLoader class
        dl_class = self.get_dataloader_class(trial_parameters)
        
        # Make a dict from its parameters
        dl_args_dict = {'dataset': ds, 'batch_size': trial_parameters['batch_size'], 'shuffle': False, 'num_workers': int(trial_parameters['num_workers']),
                        'drop_last': False}

        # Initialize DataLoader
        dl = dl_class(**dl_args_dict) if train_size > 0 else None

        return dl

    def get_transforms(self, logger, trial_parameters):
        """
        NOTE: Check the idea behind adding "weight_map".
        Author: Sama Shapouri
        """
    
        logger.my_print(f"To_device: {trial_parameters['to_device']}.")

        generic_transforms = Compose([
            LoadImaged(keys=trial_parameters['image_keys'][:2], reader=ITKReader()),
            EnsureChannelFirstd(keys=trial_parameters['image_keys'][:2]),
            NormalizeIntensityd(keys=trial_parameters['image_keys'][:2]),
            Spacingd(keys=trial_parameters['image_keys'][:2], pixdim=trial_parameters['spacing'], mode=trial_parameters['ct_interpol_mode_3d']),
            SpatialPadd(keys=trial_parameters['image_keys'][:2], spatial_size=trial_parameters['output_size'], mode='constant'),
            CenterSpatialCropd(keys=trial_parameters['image_keys'][:2], roi_size=trial_parameters['output_size']),
        ])

        train_transforms = generic_transforms
        val_transforms = generic_transforms

        # if trial_parameters['perform_data_aug']:
        #     weighted_crop_transform = Compose([
        #         CreateCustomWeightMapd(keys=trial_parameters['image_keys'][:2], w_key='weight_map'),#trial_parameters['image_keys'][-1]),  # Custom weight map
        #         EnsureTyped(keys=trial_parameters['image_keys']),
        #         RandWeightedCropd(
        #             keys=trial_parameters['image_keys'][:2],
        #             w_key='weight_map', #trial_parameters['image_keys'][-1],
        #             spatial_size=trial_parameters['cropping_dim'],
        #             num_samples=trial_parameters['num_sample']),
        #         SpatialPadd(keys=trial_parameters['image_keys'][:2], spatial_size=trial_parameters['output_size'], mode='constant'),
        #     ])

        #     train_transforms = Compose([
        #         train_transforms,
        #         ConditionalTransform(keys=trial_parameters['image_keys'][:2], transform=weighted_crop_transform, prob=trial_parameters['data_aug_p']),
        #         Rand3DElasticd(
        #             keys=trial_parameters['image_keys'][:2],
        #             sigma_range=trial_parameters['sigma_elastic'],
        #             magnitude_range=trial_parameters['magnitude_elastic'],
        #             prob=trial_parameters['data_aug_p'], shear_range=None,
        #             mode='nearest', padding_mode='zeros'),
        #     ])

        if trial_parameters['to_device']:
            
            train_transforms = Compose([
                                        train_transforms,
                                        ToDeviced(keys=[trial_parameters['concat_key']], 
                                                  device=trial_parameters['device']),
                                        ])
            
            val_transforms = Compose([
                                        val_transforms,
                                        ToDeviced(keys=[trial_parameters['concat_key']], 
                                                  device=trial_parameters['device']),
                                    ])

        train_transforms = train_transforms.flatten()
        val_transforms = val_transforms.flatten()

        logger.my_print(f'Transformers have been made successfully.')
        
        return train_transforms, val_transforms


    def get_files(self, logger, trial_parameters):

        # Find the patient IDs available in the data folder
        patient_ids_list = self.pp_obj.extract_patient_ids(trial_parameters['data_dir'])

        logger.my_print(f'{len(patient_ids_list)} patients have been found in the data directory.')


        total_patients = len(patient_ids_list)
        train_val_size = int(total_patients * trial_parameters['train_frac'])
        
        # Divide the dataset into train-val-test sets
        train_val_ids = patient_ids_list[:train_val_size]
        test_ids = patient_ids_list[train_val_size:]
        
        # Devide train-val into train val pat IDs
        val_size = int(train_val_size * trial_parameters['val_frac'])
        val_ids = train_val_ids[:val_size]
        train_ids = train_val_ids[val_size:]
        
        # Extract the directories to the images
        train_pct, train_rct, train_pos, train_ids_list = self.pp_obj.prepare_data_nrrd_for_CT(trial_parameters['data_dir'], train_ids)
        val_pct, val_rct, val_pos, val_ids_list = self.pp_obj.prepare_data_nrrd_for_CT(trial_parameters['data_dir'], val_ids)
        test_pct, test_rct, test_pos, test_ids_list = self.pp_obj.prepare_data_nrrd_for_CT(trial_parameters['data_dir'], test_ids)

        # Create dictionaries for each dataset with pos values as NumPy arrays divided by 3 
        train_dict = [{"fixed": img, "moving": tar, "pos": pos, "pat_id":pat_id} for img, tar, pos, pat_id in zip(train_pct, train_rct, train_pos, train_ids_list)]
        val_dict = [{"fixed": img, "moving": tar, "pos": pos, "pat_id":pat_id} for img, tar, pos, pat_id in zip(val_pct, val_rct, val_pos, val_ids_list)]
        test_dict = [{"fixed": img, "moving": tar, "pos": pos, "pat_id":pat_id} for img, tar, pos, pat_id in zip(test_pct, test_rct, test_pos, test_ids_list)]

        # train_dict = [{"fixed": img, "moving": tar, "pos": pos} for img, tar, pos in zip(train_pct, train_rct, train_pos)]
        # val_dict = [{"fixed": img, "moving": tar, "pos": pos} for img, tar, pos in zip(val_pct, val_rct, val_pos)]
        # test_dict = [{"fixed": img, "moving": tar, "pos": pos} for img, tar, pos in zip(test_pct, test_rct, test_pos)]
        
        
        # Add some inf into logger
        logger.my_print(f'Train set contains {len(train_ids)} patients.')
        logger.my_print(f'Val set contains {len(val_ids)} patients.')
        logger.my_print(f'Test set contains {len(test_ids)} patients.')
        
        # Check whether all the patients are in the data sets
        assert total_patients == (len(train_ids) + len(val_ids) + len(test_ids))

        return train_dict, val_dict, test_dict


    def load_fold_data(self, fold, cv_idx_list, train_dict, val_dict):
        
        # For this fold: select training and internal validation indices
        train_idx, valid_idx = cv_idx_list[fold]

        train_val_dict = train_dict + val_dict
        
        # Fetch training set using cv indices in 'cv_idx_list'
        train_dict = list()
        for i in train_idx:
            train_dict.append(train_val_dict[i])

        # Fetch internal validation set using cv indices in 'cv_idx_list'
        val_dict = list()
        for i in valid_idx:
            val_dict.append(train_val_dict[i])     
        
        return train_dict, val_dict
    
    
    def get_cv_index_list(self, logger, trial_parameters, train_dict, val_dict):
        """
        NOTE: This code has a great deal of bias now, change it in the future!!!!!
        """
        logger.my_print(f"Performing {trial_parameters['cv_folds']}-fold Cross Validation.")
        cv_idx_list = list()
        train_val_dict = train_dict + val_dict
        cv_object = KFold(n_splits=trial_parameters['cv_folds'], shuffle=True, random_state=trial_parameters['seed'])

        for idx in cv_object.split(X=train_val_dict):
            cv_idx_list.append(idx) 

        return cv_idx_list


    def make_dataloaders(self, logger, train_dict, val_dict, test_dict, trial_parameters, fold):
        
        total_data_num = len(train_dict) + len(val_dict) + len(test_dict)

        # Make the main dict
        main_dict = {'train_dict':train_dict, 'val_dict': val_dict, 'test_dict': test_dict}

        # Save some inf about the data dictionaries in the logger
        logger.my_print(f'Train Set: {len(train_dict)}/{total_data_num}, percent: {round(len(train_dict)/total_data_num, 3)}')
        logger.my_print(f'Val Set: {len(val_dict)}/{total_data_num}, percent: {round(len(val_dict)/total_data_num, 3)}')
        logger.my_print(f'Test Set: {len(test_dict)}/{total_data_num}, percent: {round(len(test_dict)/total_data_num, 3)}')

        # Save the data dictionaries
        self.writer.make_json_file(main_dict, f"{trial_parameters['exp_dir']}/Data_dict_{fold}.json")

        # Get the transformers for the train and validation sets
        transform_generator = TransformGenerator()

        # Generate both training and validation transformations
        train_transforms, val_transforms = transform_generator.generate_transforms(trial_parameters, logger)

        # # Get the normalization DataLoader NOTE: it is too heavy for the program
        # train_dl_normalization =self.get_normalization_dataloader(trial_parameters, train_dict, val_test_transforms)
        # norm_mean, norm_std = self.get_mean_and_std(train_dl_normalization, trial_parameters) # Get mean and std

        # # Add the mean and STD to the logger
        # logger.my_print(f'Normalization mean: {norm_mean}')
        # logger.my_print(f'Normalization std: {norm_std}')
        # del train_dl_normalization # Delet the normalization DataLoader to freee some space

        # Get training, internal validation and test dataloaders
        train_dl, val_dl, test_dl, train_ds, dl_class, train_dl_args_dict = self.get_dataloaders(
            train_dict, val_dict, test_dict, train_transforms, val_test_transforms, trial_parameters, logger
            )

        print('HIIIIIIIIIIIIIIIIIIIIIIIIII', train_dl.dataset, train_dl.batch_size)
        # Pack variables in one dictionary
        dataloader_dict = {'train_dl': train_dl, 'val_dl': val_dl, 'test_dl': test_dl, 'train_ds': train_ds,
                           'dl_class': dl_class, 'train_dl_args_dict': train_dl_args_dict, 'train_dict': train_dict,
                           'val_dict': val_dict, 'test_dict': test_dict}#, 'norm_mean': norm_mean, 'norm_std': norm_std
                            # }
        print('HIIIIIIIIIIIIIIIIIIIIIIIIII', train_dl.dataset, train_dl.batch_size)
        
        # for num, x in enumerate(train_dl):
        #     print(num)        
        return dataloader_dict        
        
        