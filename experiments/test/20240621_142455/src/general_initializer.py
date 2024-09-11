"""
Foreword: The purpose of this pipeline is to make an infrastructure for some of the deep learning projects of the group. 
          I will complete it step by step and mainly based on different projects. 

Explanation:
TODO: Change the learning rate initializer. it is not efficient.

Author: Hooman Bahrdo
Last Revision: 06/14/2024 
"""

# General libraries
import os
import math
import json
import torch
import optuna
import joblib
import wandb
from torchinfo import summary
import matplotlib.pyplot as plt

# Costume Libraries
import optimizers
from config_reader import ConfigReader
from div.lr_finder import LearningRateFinder, LearningRateWarmUp
from assistant_classes import PathProcessor, Writer, Logger
from models.dual_dcnn_lrelu import Dual_DCNN_LReLU
from div.loss_functions import *

class GeneralInitializer():

    def __init__(self):
        self.pp_obj = PathProcessor()
        self.writer_obj = Writer()

    def initialize_wandb(self):
        if self.device_dict['wandb_mode'] != 'disabled':
            return wandb.init(project='DBP_project', reinit=True, mode=self.device_dict['wandb_mode'])
        
        else:
            return None
    
    def initialize_logger(self, trial_parameters):
   
        # Logger output filename
        output_filename = os.path.join(trial_parameters['exp_dir'], f'log.txt')

        # Initialize logger
        logger = Logger(output_filename=output_filename)

        # Print variables
        logger.my_print('Device: {}.'.format(trial_parameters['device']))
        logger.my_print('Torch version: {}.'.format(trial_parameters['torch_version']))
        logger.my_print('Torch.backends.cudnn.benchmark: {}.'.format(torch.backends.cudnn.benchmark))
        logger.my_print('Model name: {}'.format(trial_parameters['model_name']))
        logger.my_print('Seed: {}'.format(trial_parameters['seed']))
        return logger


    def initialize_main_stream(self,fold, trial_parameters):
        """
        Set up experiment and save the experiment path to be able to save results.
        """

        # Fetch folders and files that should be copied to exp folder
        # TODO: Add the exclusion part, it will be used when multi-input model is used (e.g. RT Dose and CT)
        src_files = [x for x in os.listdir(trial_parameters['root_dir']) if x.endswith('.py')] 

        # Copy src files to exp folder
        for f in src_files:
            self.pp_obj.copy_file(src=os.path.join(trial_parameters['root_dir'], f), 
                                  dst=os.path.join(trial_parameters['exp_src_dir'], f))

        # Copy folders to exp folder
        self.pp_obj.copy_folder(src=trial_parameters['models_dir'], dst=trial_parameters['exp_models_dir'])
        self.pp_obj.copy_folder(src=trial_parameters['optimizers_dir'], dst=trial_parameters['exp_optimizers_dir'])
        self.pp_obj.copy_folder(src=trial_parameters['data_preproc_dir'], dst=trial_parameters['exp_data_preproc_dir'])
        
        # It means that if you have a pretrained model. NOTE: This section is used for transfer learning.
        if trial_parameters['pretrained_path'] is not None: 
            trial_parameters['pretrained_path_i'] = os.path.join(trial_parameters['pretrained_path'], str(fold), trial_parameters['filename_best_model_pth'])
        else:
           trial_parameters['pretrained_path_i'] = None
        
        return trial_parameters

class ModelInitializer():
    """
    Explanation: This class is used to initialize the model. I will right some criteria for anybody who 
                 wants to add a new model to this pipeline. 
    """
    
    def call_Dual_DCNN_LReLu(self, trial_parameters):
        return Dual_DCNN_LReLU(trial_parameters)


    def get_model(self, trial_parameters):
        """
        Initialize model.
        """
        
        func_name = f"call_{trial_parameters['model_name']}"

        try:
            func = getattr(self, func_name)
            model = func(trial_parameters)

        except AttributeError:
            print(f'Method {func_name} not found in OptunaOptimizer.')

        except Exception as e:
            print(f'An error occurred while calling {func_name}: {e}')
        
        return model
    
    def save_model_summary(self, model, tp):
        """
        This method saves a c=summary of the model in a text file, it is using torchinfo
        NOTE: This should be corrected when I want to add multi-model to this architecture.
        NOTE: Very important not, it is adjusted for two input arms, if I want to adjust the
              model to get features as the input, I MUST add another dimension properly to the model.
        """
        # Make the input size 
        input_size = [(tp['batch_size'], tp['n_input_channels'], tp['output_size'][0], tp['output_size'][1], tp['output_size'][2]), 
                      (tp['batch_size'], tp['n_input_channels'], tp['output_size'][0], tp['output_size'][1], tp['output_size'][2])]

        # Get and save summary
        txt = str(summary(model=model, input_size=input_size, device=tp['device']))
        file = open(os.path.join(tp['exp_dir'], tp['model_txt']), 'a+', encoding='utf-8')
        file.write(txt)
        file.close()



class WeightInitializer():
    
    def initialize_selu_parameters(self, trial_parameters):
        # Source: https://github.com/bioinf-jku/SNNs/tree/master/Pytorch
        trial_parameters['weight_init_name'] = 'kaiming_normal'
        trial_parameters['kaiming_a'] = None
        trial_parameters['kaiming_mode'] = 'fan_in'
        trial_parameters['kaiming_nonlinearity'] = 'linear'
        
        return trial_parameters

    def call_xavier_uniform(self, m, trial_parameters):
        torch.nn.init.xavier_uniform_(m.weight, gain=trial_parameters['gain'])
        # return torch.nn.init.xavier_uniform_(m.weight, gain=trial_parameters['gain'])
    
    def call_uniform(self, m, trial_parameters):
        return torch.nn.init.uniform_(m.weight)    

    def call_kaiming_uniform(self, m, trial_parameters):
        return torch.nn.init.kaiming_uniform_(m.weight, a=trial_parameters['kaiming_a'], 
                                              mode=trial_parameters['kaiming_mode'], 
                                              nonlinearity=trial_parameters['kaiming_nonlinearity'])  

    def call_kaiming_normal(self, m, trial_parameters):
        return torch.nn.init.kaiming_normal_(m.weight, a=trial_parameters['kaiming_a'], 
                                             mode=trial_parameters['kaiming_mode'], 
                                             nonlinearity=trial_parameters['kaiming_nonlinearity'])   

    def call_normal(self, m, trial_parameters):
        return torch.nn.init.normal_(m.weight)   

    def call_xavier_normal(self, m, trial_parameters):
        return torch.nn.init.xavier_normal_(m.weight, gain=trial_parameters['gain'])   
    
    def call_orthogonal(self, m, trial_parameters):
        return torch.nn.init.orthogonal_(m.weight, gain=trial_parameters['gain'])       

    def check_bias(self, model):
        if model.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.weight)
            bound = 1 / math.sqrt(fan_in)  # if fan_in > 0 else 0
            torch.nn.init.uniform_(model.bias, -bound, bound)

                 
    def initialize_weights(self, m, trial_parameters):
        # Initialize variables
        classname = m.__class__.__name__
 
        if 'output' in classname.lower():
            m.apply(lambda model: self.call_xavier_uniform(model, trial_parameters))
            m.apply(lambda model: self.check_bias(model))
        
        elif any(sub in classname for sub in ('Linear', 'Conv')):
            func_name = f"call_{trial_parameters['weight_init_name']}"
            
            try:
                func = getattr(self, func_name)
                m.apply(lambda model: func(model, trial_parameters))
                m.apply(lambda model: self.check_bias(model))
                
            except AttributeError:
                print(f'Method {func_name} not found in WeightInitializer.')

            except Exception as e:
                print(f'An error occurred while calling {func_name}: {e}')
       
        
    def get_initializer(self, model, trial_parameters, logger):
        """
        Explanation: In this class the requested weight initializer will be used or based on the 
                    activation function or the pretrained model, it will be chosen.
        """
        
        if 'selu' in trial_parameters['model_name']:
            trial_parameters = self.initialize_selu_parameters(trial_parameters)
            logger.my_print(f"Weight init name: {trial_parameters['weight_init_name']}.")
            
       # Initialize the weights
       # Initialize the weights of the privious model if using transfer learning
        if trial_parameters['pretrained_path_i'] is not None:
            logger.my_print(f"Using pretrained weights: {trial_parameters['pretrained_path_i']}.")

        # Initialize weights based on the name of the initializer
        elif (trial_parameters['pretrained_path_i'] is None and trial_parameters['weight_init_name'] 
              and all(sub not in trial_parameters['model_name'] for sub in ('efficientnet', 'convnext', 'resnext'))):
            model.apply(lambda m: self.initialize_weights(m, trial_parameters)) # Source: https://pytorch.org/docs/stable/generated/torch.nn.Module.html         
            logger.my_print(f"Weight init name: {trial_parameters['weight_init_name']}.")
        
        else:
            logger.my_print(f"Using default PyTorch weights init scheme for {trial_parameters['model_name']}.")
        
        return model


class LossFuncInitializer():
    """
    Explanation: This class initialize the loss function.
    """
    def call_l1_loss(self, trial_parameters):
        pytorch_loss_obj = PyTorchLossFunctions()
        return pytorch_loss_obj.l1_loss()

    def call_mse_loss(self, trial_parameters):
        pytorch_loss_obj = PyTorchLossFunctions()
        return pytorch_loss_obj.mse_loss()
    
    def call_sep_channel_mae_loss(self, trial_parameters):
        return SeparatedChannelMAELoss()
        
    def call_sep_channel_mse_loss(self, trial_parameters):
        return SeparatedChannelMSELoss()

    def call_sep_channel_chamfer_loss(self, trial_parameters):
        return SeparatedChannelChamferLoss()        

    def call_sep_channel_weighted_mse_loss(self, trial_parameters):
        return SeparatedChannelWeightedMSELoss()
    
    def call_sep_channel_weighted_mae_loss(self, trial_parameters):
        return SeparatedChannelWeightedMAELoss()

    def get_loss_function(self, trial_parameters, logger):
        """
        CrossEntropyLoss: softmax is computed as part of the loss. In other words, the model outputs should be logits,
            i.e. the model output should NOT be softmax'd.
        BCEWithLogitsLoss: sigmoid is computed as part of the loss. In other words, the model outputs should be logits,
            i.e. the model output should NOT be sigmoid'd.

        Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
        Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

        """
        func_name = f"call_{trial_parameters['loss_function_name']}"

        try:
            func = getattr(self, func_name)
            loss_func = func(trial_parameters)

        except AttributeError:
            logger.my_print(f'Method {func_name} not found in OptunaOptimizer.')
            raise f'Method {func_name} not found in OptunaOptimizer.'

        except Exception as e:
            logger.my_print(f'An error occurred while calling {func_name}: {e}')
            raise f'Method {func_name} not found in OptunaOptimizer.'
        
        return loss_func        


 
class OptimizerInitializer():
    """
    Explanation: This class is responsible for initializing the optimizer
    """  
    def get_optimizer(self, model, trial_parameters, logger):
        """
        CrossEntropyLoss: softmax is computed as part of the loss. In other words, the model outputs should be logits,
            i.e. the model output should NOT be softmax'd.
        BCEWithLogitsLoss: sigmoid is computed as part of the loss. In other words, the model outputs should be logits,
            i.e. the model output should NOT be sigmoid'd.

        Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CrossEntropyLoss
        Source: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

        """
        func_name = f"call_{trial_parameters['optimizer_name']}"

        try:
            func = getattr(self, func_name)
            optimizer = func(model, trial_parameters)
            
            if trial_parameters['use_lookahead']:
                optimizer = optimizers.Lookahead(optimizer, k=trial_parameters['lookahead_k'], 
                                                 alpha=trial_parameters['lookahead_alpha'])

        except AttributeError:
            logger.my_print(f'Method {func_name} not found in OptunaOptimizer.')
            raise f'Method {func_name} not found in OptunaOptimizer.'

        except Exception as e:
            logger.my_print(f'An error occurred while calling {func_name}: {e}')
            raise f'Method {func_name} not found in OptunaOptimizer.'
        
        return optimizer
    
    def call_adam(self, model, tp):
        return torch.optim.Adam(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])
    
    def call_adam_w(self, model, tp):
        return torch.optim.AdamW(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])
    
    def call_rmsprop(self, model, tp):
        return torch.optim.RMSprop(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])   
    
    def call_sgd(self, model, tp):
        return torch.optim.SGD(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])    

    def call_sgd_w(self, model, tp):
        return optimizers.SGDW(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])  
    
    def call_ada_hessian(self, model, tp):
        return optimizers.Adahessian(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], hessian_power=tp['hessian_power'])          

    def call_acc_sgd(self, model, tp):
        return optimizers.AccSGD(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])
    
    def call_ada_belief(self, model, tp):
        return optimizers.AdaBelief(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])    

    def call_ada_bound(self, model, tp):
        return optimizers.AdaBound(model.parameters(), lr=tp['lr'], final_lr=tp['lr'] * 100, weight_decay=tp['weight_decay'])   
        
    def call_ada_bound_w(self, model, tp):
        return optimizers.AdaBoundW(model.parameters(), lr=tp['lr'], final_lr=tp['lr'] * 100, weight_decay=tp['weight_decay'])   

    def call_ada_mod(self, model, tp):
        return optimizers.AdaMod(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])  
    
    def call_apollo(self, model, tp):
        return optimizers.Apollo(model.parameters(), lr=tp['lr'], init_lr=tp['lr'] / 100, warmup=500, weight_decay=tp['weight_decay'])           

    def call_diff_grad(self, model, tp):
        return optimizers.DiffGrad(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])  

    def call_madgrad(self, model, tp):
        return optimizers.MADGRAD(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])  

    def call_novo_grad(self, model, tp):
        return optimizers.NovoGrad(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])          

    def call_pid(self, model, tp):
        return optimizers.PID(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])  
        
    def call_qh_adam(self, model, tp):
        return optimizers.QHAdam(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], nus=(0.7, 1.0), betas=(0.995, 0.999))          

    def call_qhm(self, model, tp):
        return optimizers.QHM(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'])          
        
    def call_r_adam(self, model, tp):
        return optimizers.RAdam(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])   

    def call_ranger_qh(self, model, tp):
        return optimizers.RangerQH(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])           
        
    def call_ranger_21(self, model, tp):
        return optimizers.Ranger21(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'], momentum=tp['momentum'],
                                   num_epochs=150, num_batches_per_epoch=tp['batch_size']) 
        
    def call_swats(self, model, tp):
        return optimizers.SWATS(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])   
        
    def call_yogi(self, model, tp):
        return optimizers.Yogi(model.parameters(), lr=tp['lr'], weight_decay=tp['weight_decay'])   


class SchedularInitializer():

    def call_cosine(self, optimizer, tp):
        # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=tp['T0'],
                                                                        T_mult=tp['T_mult'],
                                                                        eta_min=tp['eta_min'])     

    def call_cyclic(self, optimizer, tp):
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR
            if tp['lr_finder_num_iter'] > 0:
                base_lr = 0.5 * tp['lr']  # 0.8 * optimal_lr
                max_lr = 2 * tp['lr']  # 1.2 * optimal_lr
            else:
                base_lr = tp['base_lr']
                max_lr = tp['max_lr']

            return torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=base_lr, max_lr=max_lr,
                                                        step_size_up=tp['step_size_up'], cycle_momentum=False)

    def call_exponential(self, optimizer, tp):
            # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
            return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=tp['gamma'])        

    def call_step_lr(self, optimizer, tp):
            # Source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=tp['step_size'], gamma=tp['gamma']) 
    
    def call_reduce_lr(self, optimizer, tp): # Main schedular that Sama used
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=tp['schedular_mode'], 
                                                          factor=tp['schedular_factor'], patience=tp['patience'], verbose=tp['verbose'])
                        
    def get_scheduler(self, optimizer, trial_parameters, logger):
        """
        Explanation: This program returns the schedular.
        """

        scheduler = None
        func_name = f"call_{trial_parameters['scheduler_name']}"

        try:
            func = getattr(self, func_name)
            scheduler = func(optimizer, trial_parameters)
            
            # Wrap LearningRateWarmUp
            # Source: https://github.com/developer0hye/Learning-Rate-WarmUp
            # Source: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
            if (trial_parameters['warmup_batches'] > 0) and (trial_parameters['lr'] is not None):
                scheduler = LearningRateWarmUp(optimizer=optimizer, tp=trial_parameters, after_scheduler=scheduler)
            
        except AttributeError:
            logger.my_print(f'Method {func_name} not found in OptunaOptimizer.')
            raise f'Method {func_name} not found in OptunaOptimizer.'

        except Exception as e:
            logger.my_print(f'An error occurred while calling {func_name}: {e}')
            raise f'Method {func_name} not found in OptunaOptimizer.'
        
        return scheduler


class LearningRateInitializer():
    
    def check_schedular(self, tp):
        if tp['scheduler_name'] == 'manual_lr':
            tp['lr'] = tp['manual_lr'][0]
            tp['manual_lr'].pop(0)

        else:
            tp['lr'] = tp['base_lr']
        
        return tp


    def find_optimal_lr(self, model, dataloader_dict, optimizer, loss_function, tp, am_obj, logger):
        """
        Find optimal learning rate.
        Note: At the end of the whole test: the model and optimizer are restored to their initial states.
        """
        # Store initial warmup value and temporarily set off for LR_Finder procedure
        if tp['optimizer_name'] in ['apollo']:
            for param_group in optimizer.param_groups:
                init_warmup = param_group['warmup']
                param_group['warmup'] = 0

        elif tp['optimizer_name'] in ['ranger_21']:
            init_use_warmup = optimizer.use_warmup
            optimizer.use_warmup = False

        # Apply lower learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = tp['lr_finder_lower_lr']

        # Initiate learning rate finder
        lr_finder = LearningRateFinder(model=model, optimizer=optimizer, tp=tp, criterion=loss_function, logger=logger)

        # Run test
        lr_finder.range_test(train_loader=dataloader_dict['train_dl'], val_loader=dataloader_dict['val_dl'], 
                             end_lr=tp['lr_finder_upper_lr'], num_iter=tp['lr_finder_num_iter'], am_obj=am_obj)

        optimal_lr, _ = lr_finder.get_steepest_gradient()
        logger.my_print('Optimal learning rate: {}.'.format(optimal_lr))

        # Apply optimal learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = optimal_lr

        # Apply initial warmup
        if tp['optimizer_name'] in ['apollo']:
            for param_group in optimizer.param_groups:
                param_group['warmup'] = init_warmup

        elif tp['optimizer_name'] in ['ranger_21']:
            optimizer.user_warmup = init_use_warmup

        # Plot optimal learning rate
        if tp['filename_lr_finder_png'] is not None:
            ax = plt.subplots(1, 1, figsize=tp['figsize'], facecolor='white')[1]
            lr_finder.plot(ax=ax)
            plt.savefig(os.path.join(tp['exp_dir'], tp['filename_lr_finder_png']))

        return optimal_lr, optimizer        

          
    def get_learning_rate(self, model, dataloader_dict, optimizer, loss_function, trial_parameters, am_obj, logger):
        
        tp_dict = self.check_schedular(trial_parameters)
        
        if tp_dict['lr_finder_num_iter'] > 0 and tp_dict['max_epochs'] > 0:
            tp_dict['lr'], optimizer = self.find_optimal_lr(model, dataloader_dict, optimizer, loss_function, tp_dict, am_obj, logger)
            tp_dict['new_lr'] = tp_dict['lr']

        tp_dict['starting_lr'] = tp_dict['lr']
        
        return tp_dict, optimizer
        
        
        

