"""
Explanation: This program contains a class that performs train and validation processes
NOTE: some schedulers generally update the LR every epoch instead of every batch, but epoch-wise update
      will mess up LearningRateWarmUp. Batch-wise LR update is always valid and works with LearningRateWarmUp.
NOTE: ADD PATIENT IDS FOR SAVING (Done) 

NOTE: Add plotting part to the train later

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
# General Libraries
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from monai.metrics import MSEMetric, MAEMetric



class TrainValTest():
    
    def find_best_model(self, model, tp, epoch, train_loss_value, best_loss_val, val_loss_value, nr_epochs_not_improved, best_epoch, logger):

        # Select the BEST model
        with open(tp['exp_result_txt'], 'a') as f: #a-append
            f.write(f"Epoch: {epoch+1}/{tp['max_epochs']}, Loss_train: {train_loss_value}, Loss_val: {val_loss_value}\n")
            logger.my_print(f"Epoch: {epoch+1}/{tp['max_epochs']}, Loss_train: {train_loss_value}, Loss_val: {val_loss_value}")
    
            if val_loss_value <= best_loss_val and epoch > 0:
                best_loss_val = val_loss_value
                best_epoch = epoch + 1
                nr_epochs_not_improved = {key:0 for key in nr_epochs_not_improved.keys()}
                torch.save(model.state_dict(), os.path.join(tp['exp_dir'], tp['filename_best_model_pth']))
                logger.my_print(f"Saved new best metric model for epoch {epoch+1}.")

            else:
                nr_epochs_not_improved = {key:val + 1 for key, val in nr_epochs_not_improved.items()}
            
            return best_loss_val, best_epoch, nr_epochs_not_improved


        
    def compute_mse(self, tp, train_y_pred, train_y):
        mse_metric = MSEMetric()
        sigmoid_act = torch.nn.Sigmoid() 
        
        # Check if the loss function is BCE
        if tp['loss_function_name'] in ['bce']:
            train_y_pred_list = sigmoid_act(train_y_pred)
        
        # Conver the torch tensors into lists
        train_y_list = [y for y in train_y]    
        train_y_pred_list = [y_pred for y_pred in train_y_pred]    
        
        # Calculate MSE metric
        mse_value = mse_metric(train_y_pred_list, train_y_list).mean().item()
        mse_value = round(mse_value, tp['nr_of_decimals'])
        
        return mse_value
        
    def compute_mae(self, tp, train_y_pred, train_y):
        mse_metric = MAEMetric()
        sigmoid_act = torch.nn.Sigmoid() 
        
        # Check if the loss function is BCE
        if tp['loss_function_name'] in ['bce']:
            train_y_pred_list = sigmoid_act(train_y_pred)
        
        # Conver the torch tensors into lists
        train_y_list = [y for y in train_y]    
        train_y_pred_list = [y_pred for y_pred in train_y_pred]    
        
        # Calculate MSE metric
        mse_value = mse_metric(train_y_pred_list, train_y_list).mean().item()
        mse_value = round(mse_value, tp['nr_of_decimals'])
        
        return mse_value            

    def adjust_lr(self, optimizer, dataloader_dict, tp, model, loss_function, nr_epochs_not_improved, cur_batch_size, lri_obj, am_obj, logger):

        # Scheduler
        if nr_epochs_not_improved['lr'] >= tp['patience_lr']:
            # LR_Finder scheduler
            if tp['scheduler_name'] == 'lr_finder':
                for param_group in optimizer.param_groups:
                    last_lr = param_group['lr']
                    # Initialize learning rate
                    tp, optimizer = lri_obj.get_learning_rate(model, dataloader_dict, optimizer, loss_function, 
                                                                                    tp, am_obj, logger)
                if tp['new_lr'] is None:
                    # Use adjusted version of old but valid lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = max(last_lr * tp['factor_lr'], tp['min_lr'])

            # ReduceLROnPlateau scheduler
            elif tp['scheduler_name'] == 'reduce_lr':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * tp['factor_lr'], tp['min_lr'])

            # Batch_size scheduler
            elif tp['scheduler_name'] == 'batch_size':
                new_batch_size = min(cur_batch_size * 2, tp['max_batch_size'])
                dataloader_dict['train_dl_args_dict']['batch_size'] = new_batch_size
                dataloader_dict['train_dl'] = dataloader_dict['dl_class'](**dataloader_dict['train_dl_args_dict'])

            nr_epochs_not_improved['lr'] = 0

        # Manual_LR scheduler: LR per epoch
        if (tp['scheduler_name'] == 'manual_lr') and (len(tp['manual_lr']) > 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = tp['manual_lr'][0]
            tp['manual_lr'].pop(0)

        return tp, optimizer, nr_epochs_not_improved, dataloader_dict

    def adjust_optimizer(self, tp, model, scheduler, optimizer, oi_obj,nr_epochs_not_improved, logger):
        # Next optimizer
        if nr_epochs_not_improved['opt'] >= tp['patience_opt']:
            # Consider new optimizer
            if len(tp['optimizer_name_next']) >= 1:
                optimizer_name_i = tp['optimizer_name_next'][0]
                tp['optimizer_name_next'].pop(0)

                optimizer = oi_obj.get_optimizer(model, tp, logger)
                scheduler.optimizer = optimizer
                logger.my_print('Next optimizer: {}.'.format(optimizer_name_i))

            nr_epochs_not_improved['opt'] = 0

        return scheduler, optimizer, nr_epochs_not_improved

    def validate(self, model, dataloader_dict, dl_mode, mode, loss_function, tp, writer_obj, logger, save_outputs=True):
        """
        Explanation: This method performs validation on all the relevant dataloaders.
        """
        # Initialize variable
        model.eval()
        loss_list = list()
        dataloader = dataloader_dict[f'{dl_mode}_dl']

        with torch.no_grad():
            y_pred = torch.as_tensor([], dtype=torch.float32, device=tp['device'])
            y = torch.as_tensor([], dtype=torch.int8, device=tp['device'])
            for data in dataloader:
                # Load data
                val_fixed, val_moving, val_positions = (
                                                        data['fixed'].to(tp['device']),
                                                        data['moving'].to(tp['device']),
                                                        data['pos'].clone().to(tp['device'])
                                                        )

                # Predict by using the model
                outputs = model(x_fixed=val_fixed, x_moving=val_moving)

                # Calculate the loss
                try:
                    loss = loss_function(outputs, val_positions)

                except:
                    loss = loss_function(outputs, torch.reshape(val_positions, outputs.shape).to(outputs.dtype))

                # Evaluate model (internal validation set)
                loss_list.append(loss.item())
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_positions], dim=0)
            
            # Calculate Loss, MSE and MAE values
            loss_value = np.mean(loss_list)
            mse_value = self.compute_mse(tp, y_pred, y)
            mae_value = self.compute_mae(tp, y_pred, y)

            # If the validation is done on the test set, save the result.
            if 'test' in mode.lower():
                # Save outputs to csv
                if save_outputs:                    
                    writer_obj.save_predictions(tp, y_pred, y, dl_mode, dataloader_dict[f'{dl_mode}_dict'], mode, logger) #patient_ids=patient_ids, Add patient ID to this data
                                        
                return loss_value, mse_value, mae_value # patient_ids,

            # Prints validation dataset parameters 
            logger.my_print(f'{mode} final results:')
            logger.my_print(f'{mode} loss: {loss_value:.3f}.')
            logger.my_print(f'{mode} MAE: {mae_value:.3f}.')
            logger.my_print(f'{mode} MSE: {mse_value:.3f}.')

        return loss_value, mse_value, mae_value


    def train(self, dataloader_dict, model, loss_function, optimizer, scheduler, tp, am_obj, lri_obj, oi_obj, plotter_obj, writer_obj, logger):
        """
        Explanation: This method performs the training process.
        """
        
        # Initialize some important params
        best_epoch = 0
        lr_list = list()
        best_loss_val = np.inf
        batch_size_list = list()
        train_mse_values_list = list() 
        train_mae_values_list = list()  
        train_loss_values_list = list()
        val_mse_values_list = list() 
        val_mae_values_list = list()  
        val_loss_values_list = list()
        train_num_iterations = len(dataloader_dict['train_dl'])    
        nr_epochs_not_improved = {'general': 0, 'lr':0, 'opt':0}

        logger.my_print('Number of training iterations per epoch: {}.'.format(train_num_iterations))
        # logger.my_print('Number of training iterations per epoch: {}.'.format(tp['batch_size']))
        
        # Begin the process of training for the epochs
        for epoch in range(tp['max_epochs']):
            logger.my_print(f"Epoch {epoch + 1}/{tp['max_epochs']}...")
            
            # Add the learning rates of the optimizer to the lr_list for all the epochs
            for param_group in optimizer.param_groups:
                logger.my_print(f"Learning rate: {param_group['lr']}.")
                lr_list.append(param_group['lr'])
            
            # Find the current batch size
            cur_batch_size = dataloader_dict['train_dl'].batch_size
            logger.my_print('Batch size: {}.'.format(cur_batch_size))
            batch_size_list.append(cur_batch_size)
            
            # Initiate training
            model.train()
            # train_loss_value = 0
            train_y_pred = torch.as_tensor([], dtype=torch.float32, device=tp['device'])
            train_y = torch.as_tensor([], dtype=torch.int8, device=tp['device'])
            
            loss_list = []
            
            logger.my_print("Dataset:", dataloader_dict['train_dl'].dataset)
            logger.my_print("Batch size:", dataloader_dict['train_dl'].batch_size)
            logger.my_print("Number of workers:", dataloader_dict['train_dl'].num_workers)

            # Iterate through batches
            for i, batch_data in tqdm(enumerate(dataloader_dict['train_dl'])):
                # Debug: Print batch data structure and types
                print(f"Batch {i}:")
                # Load data
                train_fixed, train_moving, train_positions = (
                    batch_data['fixed'].to(tp['device']),
                    batch_data['moving'].to(tp['device']),
                    batch_data['pos'].clone().to(tp['device'])
                )

                # Perform AugMix NOTE: THIS MAY HAVE SOME ERRORS, FIX IT BEFORE USE IT. 
                if tp['perform_augmix']:
                    for b in range(len(train_fixed)):
                        # Generate a random 32-bit integer
                        seed_b = random.getrandbits(32)
                        train_fixed[b] = am_obj.aug_mix(arr=train_fixed[b],
                                                        mixture_width=tp['mixture_width'],
                                                        mixture_depth=tp['mixture_depth'],
                                                        augmix_strength=tp['augmix_strength'],
                                                        seed=seed_b, tp=tp)

                        train_moving[b] = am_obj.aug_mix(arr=train_moving[b],
                                                         mixture_width=tp['mixture_width'],
                                                         mixture_depth=tp['mixture_depth'],
                                                         augmix_strength=tp['augmix_strength'],
                                                         seed=seed_b, tp=tp)
                        
                # NOTE: I can add some preprocessing steps here.
                # Zero the parameter gradients and make predictions
                optimizer.zero_grad(set_to_none=True)

                # Train the model
                train_outputs = model(x_fixed=train_fixed, x_moving=train_moving)

                # Calculate loss
                try:
                    # Cross-Entropy, Ranking, Custom
                    train_loss = loss_function(train_outputs, train_positions)
                except:
                    # BCE
                    train_loss = loss_function(train_outputs,
                                            torch.reshape(train_positions, train_outputs.shape).to(train_outputs.dtype))
                
                # Calculate the gradients in backpropagation act
                if tp['optimizer_name'] in ['ada_hessian']:
                    train_loss.backward(create_graph=True)

                else:
                    train_loss.backward()

                # Perform gradient clipping
                # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
                if tp['grad_max_norm'] is not None:
                    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=tp['grad_max_norm'])

                # Print gradient values
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         logger.my_print(f'HIIIIIIIIIIIIIIIIIIIIIIIIIII Gradient for {name}: {param.grad}')
                #     else:
                #         logger.my_print(f'HIIIIIIIIIIIIIIIIIIIIIIIIIII No gradient for {name}')
                # Print the learning rate
                # for param_group in optimizer.param_groups:
                #     logger.my_print(f'HIIIIIIIIIIIIIIIIIIIIIIIIIII Learning rate: {param_group["lr"]}')



                # Update model weights
                optimizer.step()
                
                # Reset the .grad fields of our parameters to None after use to break the cycle and avoid the memory leak
                if tp['optimizer_name'] in ['ada_hessian']:
                    for p in model.parameters():
                        p.grad = None

                # # Scheduler: step() called after every batch update
                # if tp['scheduler_name'] in ['cosine', 'exponential', 'step']:
                #     scheduler.step(epoch + (i + 1) / train_num_iterations)
                # elif tp['scheduler_name'] in ['cyclic']:
                #     scheduler.step()

                # Add the predictions to the following torch tensors
                train_y_pred = torch.cat([train_y_pred, train_outputs], dim=0)
                train_y = torch.cat([train_y, train_positions], dim=0)

                # Append the train loss values to the list
                loss_list.append(train_loss.item())
                logger.my_print(f"Epoch: {epoch+1}/{tp['max_epochs']}, Batch: {i+1}/{train_num_iterations}, Batch_Loss_Train: {train_loss.item():.3f}")

            # Calculate the Loss value of the training set
            train_loss_value = np.mean(loss_list)

            # Calculate MSE and MAE values
            train_mse_value = self.compute_mse(tp, train_y_pred, train_y)
            train_mae_value = self.compute_mae(tp, train_y_pred, train_y)

            # Prints
            logger.my_print(f"{epoch+1}/{tp['max_epochs']} final results:")
            logger.my_print(f'Training loss: {train_loss_value:.3f}.')
            logger.my_print(f'Training MAE: {train_mae_value:.3f}.')
            logger.my_print(f'Training MSE: {train_mse_value:.3f}.')

            train_loss_values_list.append(train_loss_value)
            train_mse_values_list.append(train_mse_value)
            train_mae_values_list.append(train_mae_value)
            
            # Initialize the validation process
            if (epoch + 1) % tp['eval_interval'] == 0:
                # Perform internal validation  
                val_loss_value, val_mse_value, val_mae_value = self.validate(model=model, dataloader_dict=dataloader_dict,
                                                                             dl_mode='val', mode='test', 
                                                                             loss_function=loss_function, tp=tp,
                                                                             writer_obj=writer_obj, logger=logger, 
                                                                             save_outputs=True)
                val_loss_values_list.append(val_loss_value)
                val_mse_values_list.append(val_mse_value)
                val_mae_values_list.append(val_mae_value)


                # Scheduler: step() called after every batch update
                if tp['scheduler_name'] in ['cosine', 'exponential', 'step']:
                    scheduler.step(epoch + (i + 1) / train_num_iterations)
                else:
                    scheduler.step(val_loss_value)
                    
                    

                # Find the BEST mode
                best_loss_val, best_epoch, nr_epochs_not_improved = self.find_best_model(model, tp, epoch, train_loss_value, best_loss_val,
                                                                                         val_loss_value, nr_epochs_not_improved, best_epoch, logger)
                logger.my_print(f'Best internal validation val_loss: {best_loss_val:.3f} at epoch: {best_epoch}.')
                tp['best_epoch'] = best_epoch
            # EarlyStopping
            if nr_epochs_not_improved['general'] >= tp['patience']:
                logger.my_print(f'Warning: No internal validation improvement during the last {nr_epochs_not_improved} consecutive epochs. (STOP TRAINING)')   
                return tp
            
            # Adjust Learning rate based on the lr patience
            tp, optimizer, nr_epochs_not_improved, dataloader_dict = self.adjust_lr(optimizer, dataloader_dict, tp, model, loss_function, nr_epochs_not_improved, 
                                                                                    cur_batch_size, lri_obj, am_obj, logger)

            # Adjust the optimizer based on the optimizer patience
            scheduler, optimizer, nr_epochs_not_improved = self.adjust_optimizer(tp, model, scheduler, optimizer, oi_obj,nr_epochs_not_improved, logger)
        
        # Make a dictionary of the results
        result_dict = { 'Loss': [train_loss_values_list, val_loss_values_list],
                        'MAE': [train_mae_values_list, val_mae_values_list],
                        'MSE': [train_mse_values_list, val_mse_values_list],
                        'LR':[lr_list],
                        'Batch_size':[batch_size_list]
                        }

        # Plot training and internal validation losses
        plotter_obj.plot_values(result_dict=result_dict, best_epoch=best_epoch, tp=tp)

        return tp


    def test(self, dataloader_dict, model, loss_function, tp, train_loss_list, val_loss_list, test_loss_list, writer_obj, logger):
        """
        Explanation: This method is used for testing the best model (based on epochs)
        """
        # Load best model
        model.load_state_dict(torch.load(os.path.join(tp['exp_dir'], tp['filename_best_model_pth'])))

        # Test the TRAINING set (NOTE: train_dl still performs data augmentation!)
        train_loss_value, train_mse_value, train_mae_value = self.validate( model=model, dataloader_dict=dataloader_dict,
                                                                            dl_mode='train', mode='test', 
                                                                            loss_function=loss_function, tp=tp,
                                                                            writer_obj=writer_obj, logger=logger, 
                                                                            save_outputs=True)        

        # Test the VALIDATION set 
        val_loss_value, val_mse_value, val_mae_value = self.validate( model=model, dataloader_dict=dataloader_dict,
                                                                            dl_mode='val', mode='test', 
                                                                            loss_function=loss_function, tp=tp,
                                                                            writer_obj=writer_obj, logger=logger, 
                                                                            save_outputs=True)
        
        # Test the TEST set
        test_loss_value, test_mse_value, test_mae_value = self.validate(model=model, dataloader_dict=dataloader_dict,
                                                                            dl_mode='test', mode='test', 
                                                                            loss_function=loss_function, tp=tp,
                                                                            writer_obj=writer_obj, logger=logger, 
                                                                            save_outputs=True)

        train_loss_list.append(train_loss_value)
        val_loss_list.append(val_loss_value)
        test_loss_list.append(test_loss_value)

        return train_loss_list, val_loss_list, test_loss_list


        

            
            

                
                
                