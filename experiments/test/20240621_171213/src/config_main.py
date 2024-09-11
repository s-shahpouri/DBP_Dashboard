"""
Explanation: 
This file contains all the dictionaries that should be adjust ro initialize the model.

VERY IMPORTANT NOTE:
DO NOT CHANGE THE KEYS.

NOTE:
1. TODO: Add features to the models, maybe to the last layers of the model.
2. TODO: Add clinical variables to the linear layers of the model. IF IT IS APPLICABLE HERE.
3. TODO: Add plotting to this

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
import json
# Config file for constants that will be red in the class Config(), 
# and it will be the interphase between the users and the pipeline.


main_dict = { 'directories': {'root_dir': '/data/bahrdoh/Deep_Learning_Pipeline', #'//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Hooman Bahrdo/Models/Deep_Learning/DL_NTCP_Xerostomia',
                              'data_dir': '/data/bahrdoh/CT_Model/Test/proton', #//zkh/AppData/RT-Meetdata/GPTC/13 Research & Testing/DoseBasedPositioning/Project_data/Processed_CT_data/Test1',#Project_data/Extra_pat2/nrrd/proton',
                              
                              'folder_name':{'models_folder': 'models',
                                             'optimizers_folder': 'optimizers',
                                             'data_preproc_folder': 'data_preproc',
                                             'save_root_folder': 'datasets',
                                            #  'data_folder': 'dataset_old_v2',
                                             'cache_folder':'persistent_cache' # (only if 'dataset_type': 'persistent')
                                            },
                              'file_names':{ 'stratified_sampling_test_csv': 'stratified_sampling_test_manual_94.csv',  # input file 
                                              'stratified_sampling_full_csv': 'stratified_sampling_full_manual_94.csv',  # output file
                                              'train_val_test_patient_ids_json': 'train_val_test_patient_ids.json',
                                              'results_csv': 'results.csv',
                                              'results_overview_csv': 'results_overview.csv',
                                              'i_outputs_csv':'{i}_outputs.csv',
                                              'lr_finder_png': 'lr_finder.png',
                                              'model_txt': 'model.txt',
                                              'best_model_pth': 'best_model.pth',
                                              'results_png': 'results.png',
                                              'optuna_study_name': 'optuna_study.pkl',
                                              'optuna_sampler_name': 'optuna_sampler.pkl',
                                              'filename_best_model_pth': 'best_model.pth',
                                              'exclude_patients_csv': 'exclude_patients.csv',
                                              'filename_lr_finder_png': 'lr_finder.png',
                                              'exp_result_txt': 'epoch_results.txt'
                                            },
                              'experiments':{ 'root_dir': '/data/bahrdoh/Deep_Learning_Pipeline/experiments/test',
                                              'src_folder': 'src',
                                              'models_folder': 'models',
                                              'optimizers_folder': 'optimizers',
                                              'data_preproc_folder': 'data_preproc',
                                              'figures_folder': 'figures',
                                              'optuna_path_pickles': 'pickles'
                                            }
                              },

              'device': { 'wandb_mode': 'online', # 'online' | 'offline' | 'disabled' # Is a nice platform that helps visualize the model performance
                          'seed': 4, # This seed is just for reproducibility.
                          'nr_of_decimals': 3,
                          'cudnn_benchmark': True,  # True if gpu_condition else False  # `True` will be faster, but potentially at cost of reproducibility
                                                    # This flag allows us to enable the inbuilt CuDNN auto-tuner to find the best algorithm to use for our hardware. 
                                                    # Only enable if input sizes of our network do not vary.

                          'data_loader':{'dataset_type': 'cache',  # 'standard' | 'cache' | 'persistent'. If None, then 'standard'.
                                         'cache_rate': 1.0,  # (dataset_type='cache') Cache: caches data in RAM storage. Persistent: caches data in disk storage instead 
                                                             # of RAM storage.
                                         'num_workers': 4,   # `4 * num_GPUs` (dataset_type='cache') # 8 or ten will be faster
                                         'to_device': False, # if num_workers > 0 else True  # Whether or not to apply `ToDeviced()` in Dataset' transforms.
                                                             # See load_data.py. Suggestion: set True on HPC, set False on local computer.
                                         'dataloader_type': 'standard',  # 'standard' | 'thread'. If None, then 'standard'. Thread: leverages the separate thread
                                         # to execute preprocessing to avoid unnecessary IPC between multiple workers of DataLoader.
                                         'use_sampler': False,  # Whether to use WeightedRandomSampler or not.
                                         'dataloader_drop_last': False  # (standard, thread)
                                        }
                        },

              'data': { 'n_input_channels': 1,
                        'image_keys': ["fixed", "moving", 'pos'],  # Do not change (subtractionct)
                        'concat_key': 'fixed_moving',#'plan_repeat', 
                        'train_frac': 0.9,  # training-internal_validation-test split. The same test set will be used for Cross-Validation.
                        'val_frac': 0.15,  # training-internal_validation-test split. The same test set will be used for Cross-Validation.
                        'sampling_type': 'stratified',  # ['random', 'stratified']. Method for dataset splitting.
                        'perform_stratified_sampling_full': True,  # (Stratified Sampling). Whether or not to recreate stratified_sampling_full.csv.
                        # Note: if stratified_sampling_full.csv does not exist, then we will perform stratified sampling to create the file.
                        'strata_groups': ['pat_id'], # (Stratified Sampling). Note: order does not matter.
                        'split_col': 'Split',  # (Stratified Sampling). Column of the stratified sampling outcome ('train', 'val', 'test').
                        'features_dl': ['pat_id'],
                        'cross_validation':{ 'cv_strata_groups': ['pat_id'],  # (TODO: implement) Stratified Cross-Validation groups
                                             'cv_folds': 2, # (Cross-Validation) If cv_folds=1, then perform train-val-test-split. (For testing, I put it equal to 1)
                                             'cv_type': 'stratified'  # (Stratified CV, only if cv_folds > 1) None | 'stratified'. Stratification is performed on endpoint value.
                                            },
                        'output_size': (128, 128, 128), 
                        'spacing': (4.0, 4.0, 4.0)                  
                      },

              'data_preprocessing':{'augmentation':{'perform_data_aug': True,
                                                    'data_aug_p': 0.5,  # Probability of a data augmentation transform: data_aug_p=0 is no data aug.
                                                    'data_aug_strength': 1,  # Strength of data augmentation: strength=0 is no data aug (except flipping).
                                                    },
                                    'clipping':{'ct':{'ct_clip': False,
                                                      'ct_a_min': -200,
                                                      'ct_a_max': 400 ,
                                                      'ct_b_min': 0.0,
                                                      'ct_b_max': 1.0
                                                     },
                                                'rtdose':{'rtdose_clip': False,
                                                          'rtdose_a_min': 0,
                                                          'rtdose_a_max': 8000,
                                                          'rtdose_b_min': 0.0,
                                                          'rtdose_b_max': 1.0
                                                         },
                                                'segmentation_map':{'seg_map_clip': False 
                                                                   }
                                                },
                                    'cropping': {'perform_cropping': True,
                                                 'cropping_dim': (110,110,110),
                                                 'num_sample': 4
                                                },
                                    'augmix': {'perform_augmix': False,
                                               'mixture_width': 3,  # 3 (default)
                                               'mixture_depth': [1, 3],  # [1, 3] (default)
                                               'augmix_strength': 3
                                              },
                                    'interpolation': {
                                                      'ct_interpol_mode_3d': 'trilinear',  # OLD 'bilinear'
                                                      'rtdose_interpol_mode_3d': 'trilinear',  # OLD 'bilinear'
                                                      'segmentation_interpol_mode_3d': 'nearest',  # OLD 'bilinear'
                                                      'ct_dose_seg_interpol_mode_3d': 'trilinear',
                                                      'ct_interpol_mode_2d': 'bilinear',
                                                      'rtdose_interpol_mode_2d': 'bilinear',
                                                      'segmentation_interpol_mode_2d': 'nearest',  # OLD 'bilinear'
                                                      'ct_dose_seg_interpol_mode_2d': 'bilinear'
                                                     },
                                    'elastic_deformation':{'sigma_elastic': (8, 8),
                                                           'magnitude_elastic': (100, 100)
                                                          }
                                    },

              'model':{'model_name': 'Dual_DCNN_LReLu',
                       'n_layers':5,
                       'filters': [8, 8, 16, 16, 32],
                       'kernel_sizes': [7, 5, 4, 3, 3],
                       'strides': [[2]*3, [2]*3, [2]*3, [2]*3, [2]*3],
                       'pad_value': 0, # (Padding) Value used for padding.
                       'lrelu_alpha': 0.1,  # (LeakyReLU) Negative slope.
                       'pooling_conv_filters': None,  # Either int or None (i.e. no pooling conv before flattening). 
                                                      # NOTE: This one a final Conv layer. Be sure the model that you use it has this option.
                       'perform_pooling': False,  # Whether to perform (Avg)Pooling or not. If pooling_conv_filters is not None, then
                       'linear_units': [16], # (Avg)Pooling will not be applied.
                       'dropout_p_j': [0],  # Should have the same length as `linear_units`  
                       'use_bias': True,
                       'num_classes': 2,  # Model outputs size. IMPORTANT: define `label_weights` such that len(label_weights) == num_classes.##############################
                       'num_ohe_classes': 2 , # Size of One-Hot Encoding output.              #######################################
                       'max_pool_dict':{'kernel_size_pool': 3, # This can be a list of three integers, or a solitary int, 
                                                           # NOTE: is you set perform max pooling True, you MUST have these three items
                                        'stride_pool': 1, # This can be a list of three integers, or a solitary int
                                        'padding_pool': 0 # This can be a list of three integers, or a solitary int
                                        },
                       'perform_max_pool': [True, True],
                       'perform_activation': [True, True] # This list determines the activation of two conv layer of each block
                      },

              'transfer_learning':{'pretrained_path': None  # None, path_to_and_including_pth (e.g. './Hooman Bahrdo/Final_results/DCNN/xer_12/dcnn_regular/best_model.pth)
                                  },

              'optimization': { 'initializer':{'weight_init_name': 'kaiming_uniform',  # [None, 'kaiming_uniform', 'uniform', 'xavier_uniform', 'kaiming_normal',
                                                                                      # 'normal', 'xavier_normal', 'orthogonal']. If None, then PyTorch's default 
                                                                                      # (i.e. 'kaiming_uniform', but with
                                                                                      # a = math.sqrt(5)). Kaiming works well if the network has (Leaky) P(ReLU) activations.
                                              'kaiming_a': 2.2360679775, # [math.sqrt(5) (default), lrelu_alpha]. Only used when kaiming_nonlinearity = 'leaky_relu'.
                                              'kaiming_mode': 'fan_in',  # ['fan_in' (default), 'fan_out'].
                                              'kaiming_nonlinearity': 'leaky_relu',  # ['leaky_relu' (default), 'relu', 'selu', 'linear']. When using 
                                                                                     # weight_init_name = kaiming_normal for initialisation with SELU activations, then 
                                                                                     # nonlinearity='linear' should be used 
                                                                                     # instead of nonlinearity='selu' in order to get Self-Normalizing Neural Networks.
                                              'gain': 0  # [0 (default), torch.nn.init.calculate_gain('leaky_relu', lrelu_alpha)].
                                             },
                                'optimizer':{'optimizer_name': 'ada_bound',  # ['acc_sgd', 'ada_belief', 'ada_bound', 'ada_bound_w', 'ada_hessian', 'ada_mod', 'adam',
                                                                            # 'adam_w', 'apollo', 'diff_grad', 'madgrad', 'novo_grad', 'qh_adam', 'qhm', 'r_adam', 
                                                                            # 'ranger_21', 'ranger_qh', 'rmsprop', 'pid', 'sgd', 'sgd_w', 'swats', 'yogi']
                                            'optimizer_name_next': [],  # [] | ['sgd']. Next optimizers after nr_epochs_not_improved_opt >= patience_lr.
                                            'momentum': 0,  # 0 (default), 0.85, 0.9 (common), 0.95. For optimizer_name in ['rmsprop', 'sgd', 'sgd_w'].
                                            'weight_decay': 0.05,  # 0 (default), 0.01 (for optimizer_name in ['adam_w', 'sgd_w']). L2 regularization penalty.
                                            'hessian_power': 1.0, # (AdaHessian)
                                            'use_lookahead': False,  # (Lookahead)
                                            'lookahead_k': 5,  # (Lookahead) 5 (default), 10.
                                            'lookahead_alpha': 0.5,  # (Lookahead) 0.5 (default), 0.8.
                                            'patience_opt': 9  # Use next optimizer in 'optimizer_name_next' after nr_epochs_not_improved_opt >= patience_opt.
                                            },
                                'loss_function':{'loss_function_name': 'l1_loss', # [None, 'bce' (num_classes = 1), 'cross_entropy' (num_classes = 2), 'cross_entropy', 
                                                                                        # 'dice', 'f1', 'ranking', 'soft_auc', 'custom']. Note: if 'bce', then also change 
                                                                                        # label_weights to list of 1 element. # use costum again
                                                                                        # Note: model output should be logits, i.e. NO sigmoid() (BCE) nor softmax() (CE) applied.
                                                'loss_weights': [1, 0, 1, 1, 0, 0], # [1/6, 1/6, 1/6, 1/6, 1/6, 1/6].
                                                                                    # (loss_function_name='custom') list of weight for [ce, dice, f1, l1, ranking, soft_auc].
                                                'label_weights': [1, 1.5],  # [1, 1] (ce) | [1] (bce) | w_jj = (1 - /beta) / (1 - /beta^{n_samples_j}) (\beta = 0.9, 0.99) |
                                                                            # wj=n_samples / (n_classes * n_samplesj). Rescaling (relative) weight given to each class, has 
                                                                            # to be list of size C.
                                                'loss_reduction': 'mean'},
                                'scheduler':{ 'scheduler_name':'cosine', # [None, 'cosine', 'cyclic', 'exponential', 'step_lr', 'reduce_lr', 'lr_finder', 'reduce_lr', 'batch_size',
                                              'step_size_up': 79 * 15,  # (CyclicLR) Number of training iterations in the increasing half of a cycle.
                                              'gamma': 0.95,  # (ExponentialLR, StepLR) Multiplicative factor of learning rate decay every epoch (ExponentialLR) or
                                                              # step_size (StepLR).                                                                       # 'manual_lr']. If None, then no scheduler will be applied.}
                                              'grad_max_norm': None,  # (GradientClipping) Maximum norm of gradients. If None, then no grad clipping will be applied.
                                              'step_size': 15,  # (StepLR) Decays the learning rate by gamma every step_size epochs.
                                              'schedular_mode': 'min',
                                              'schedular_factor': 0.1,
                                              'verbose': True,
                                              'patience': 10  # (EarlyStopping): stop training after this number of consecutive epochs without
                                            },
                                'learning_rate':{'lr': 1e-4,  # 0.0004750810162102798  # Redundant if perform_lr_finder=True.
                                                 'lr_finder_lower_lr': 1e-6,  # (LearningRateFinder) Maximum learning rate value to try.
                                                 'lr_finder_upper_lr': 1e-3, # (LearningRateFinder) Minimum learning rate value to try.
                                                 'lr_finder_num_iter': 0,  # (LearningRateFinder) Number of learning rates to try within [lower_lr, upper_lr].
                                                 'warmup_batches': 0,  # (LearningRateWarmUP) Number of warmup batches. If warmup = 0, then no warmup. Does not work with
                                                                       # manual schedulers (e.g. 'manual_lr', see misc.get_scheduler()).
                                                 'T0': 8,  # (CosineAnnealingWarmRestarts) Number of epochs until a restart.
                                                 'T_mult': 1,  # (CosineAnnealingWarmRestarts) A factor increases after a restart. Default: 1.
                                                 'eta_min': 1e-8,  # (CosineAnnealingWarmRestarts) Minimum learning rate. Default: 0.
                                                 'base_lr': 1e-7,  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Minimum and starting learning rate.
                                                 'max_lr': 1e-4,  # (CylicLR, only if perform_lr_finder = False, see get_scheduler()) Maximum learning rate.


                                                 'factor_lr': 0.5,  # (Reduce_LR) Factor by which the learning rate will be updated: new_lr = lr * factor.
                                                 'min_lr': 1e-8,  # (Reduce_LR) Minimum learning rate allowed.
                                                 'patience_lr': 7,  # (LR_Finder, Reduce_LR) Perform LR Finder after this number of consecutive epochs without
                                                 'manual_lr': [1e-3, 1e-5, 1e-6]  # (Manual_LR) LR per epoch, if epoch > len(manual_lr), then use lr = manual_lr[-1].
                                                                                  # Note: if perform_lr_finder = True, then the first value of manual_lr will not be used!
                                                }
                                },

              'training':{ # Use training keys if you are NOT using OPTUNA
                          'nr_runs': 1,
                          'max_epochs': 200,
                          'batch_size': 8,
                          'max_batch_size': 16,
                          'eval_interval': 1,
                         },

              'plotting':{'plot_interval': 10, # It will be added to the program in the next steps
                          'max_nr_images_per_interval': 1,
                          'figsize': (12, 12)},

              'optuna':{'optuna_n_trials': 100,
                        'batch_size_list': [8, 16, 32, 64],
                        'filters_list': [[16, 16, 32, 32, 64], [8, 8, 16, 32, 64], [16, 16, 32, 64, 128], 
                                         [16, 16, 16, 32, 32], [8, 8, 16, 16, 32], [8, 8, 8, 16, 16]],
                        'kernel_size_range': [3, 7],

                        'linear_units_size': [1, 2, 3],
                        'linear_units_list': [8, 16, 32, 64],
                        'dropout_p_j': [0.0, 0.5],     
                
                        'clinical_variables_linear_units_size': [1, 2, 3],
                        'clinical_variables_linear_units_list': [8, 16, 32, 64],
                        'clinical_variables_dropout_p_j': [0.0, 0.5],                        
                      
                        'features_dl_list': [[]], # Options: [['xer_wk1_not_at_all', 'xer_wk1_little', 'xer_wk1_moderate_to_severe', 'sex', 'age']] 
                                                        #[[] , ['xer_wk1_not_at_all', 'xer_wk1_little', 'xer_wk1_moderate_to_severe', 'sex', 'age']], and so on
                        'optuna_step_size_up':[], # NOTE: DO NOT FILL THIS
                        'ranges':{'data_aug_strength': [0.0, 3.0],
                                    'data_aug_p': [0.9, 1.0],
                                    'augmix_strength': [0.0, 3.0],
                                    'loss_function_weights_ce': [0.0, 1.0],
                                    'loss_function_weights_f1': [0.0, 1.0],
                                    'loss_function_weights_dice': [0.0, 1.0],
                                    'loss_function_weights_l1': [0.0, 1.0],
                                    'loss_function_weights_ranking': [0.0, 1.0],
                                    'loss_function_weights_softauc': [0.0, 1.0],
                                    'loss_function_name':['mse_loss', 'l1_loss'], # 'sep_channel_mae_loss', 'sep_channel_chamfer_loss' , 'mse_loss', 'l1_loss',
                                    'optimizer_name': ['adam','rmsprop', 'sgd_w', 'sgd', 'acc_sgd'],
                                    'scheduler_name': ['cyclic', 'reduce_lr', 'step_lr', 'cosine'], # 'cyclic', 'reduce_lr', 'step_lr', 'cosine'
                                    'n_layers': [4, 5],
                                    'stride_value_last_layer': [1, 2],
                                    'lr': [5e-6, 1e-3], # 1e-2]
                                    'T0_cosine': [8, 64],
                                    'gamma': [0.9, 1.0],
                                    'use_momentum': [True, False],
                                    'momentum': [0.0, 1.0],
                                    'weight_decay': [0.0, 0.1],
                                    'label_smoothing': [0.0, 0.1],
                                    'T0': [8, 64]
                                  }
                        },

                'segmentation_structures': [], # Fill this if you have segmentation maps as your input data or you are cropping based on the OARs.

}


def make_json_file(main_dict):
  # Specify the file path
  file_path = 'config_main.json' # NOTE: Do NOT change this name

  # Save the dictionary as a JSON file
  with open(file_path, 'w') as json_file:
      json.dump(main_dict, json_file, indent=4)


if __name__ == '__main__':
  make_json_file(main_dict)