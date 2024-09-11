# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# NOTE: FOR MULTI-MODEL, I NEED TO ADD ALL THE PARAMETERES RELATED TO THE FEATURES TO THIS


import random
import pickle
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import DataLoader

from monai.networks.utils import eval_mode
from monai.optimizers.lr_scheduler import ExponentialLR, LinearLR
from monai.utils import StateCacher, copy_to_device, optional_import


if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    has_matplotlib = True
    import tqdm

    has_tqdm = True
else:
    plt, has_matplotlib = optional_import('matplotlib.pyplot')
    tqdm, has_tqdm = optional_import('tqdm')

__all__ = ['LearningRateFinder']


class DataLoaderIter:
    def __init__(self, data_loader: DataLoader,
                 # image_extractor: Callable,
                 # ct_extractor: Callable, rtdose_extractor: Callable, segmentation_map_extractor: Callable,
                 ct_dose_seg_extractor: Callable,
                #  features_extractor: Callable,
                 label_extractor: Callable) -> None:
        if not isinstance(data_loader, DataLoader):
            raise ValueError(
                f'Loader has unsupported type: {type(data_loader)}. Expected type was `torch.utils.data.DataLoader`.'
            )
        self.data_loader = data_loader
        self._iterator = iter(data_loader)
        # self.image_extractor = image_extractor
        # self.ct_extractor = ct_extractor
        # self.rtdose_extractor = rtdose_extractor
        # self.segmentation_map_extractor = segmentation_map_extractor
        self.ct_dose_seg_extractor = ct_dose_seg_extractor
        # self.features_extractor = features_extractor
        self.label_extractor = label_extractor

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        # inputs = self.image_extractor(batch_data)
        # ct = self.ct_extractor(batch_data)
        # rtdose = self.rtdose_extractor(batch_data)
        # segmentation_map = self.segmentation_map_extractor(batch_data)
        ct_dose_seg = self.ct_dose_seg_extractor(batch_data)
        # features = self.features_extractor(batch_data)
        labels = self.label_extractor(batch_data)
        # return ct, rtdose, segmentation_map, features, labels
        return ct_dose_seg, labels #features,

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self.inputs_labels_from_batch(batch)


class TrainDataLoaderIter(DataLoaderIter):
    def __init__(
        self, data_loader: DataLoader,
            # image_extractor: Callable,
            # ct_extractor: Callable, rtdose_extractor: Callable, segmentation_map_extractor: Callable,
            ct_dose_seg_extractor: Callable,
            # features_extractor: Callable,
            label_extractor: Callable, auto_reset: bool = True
    ) -> None:
        super().__init__(data_loader,
                         # image_extractor,
                         # ct_extractor, rtdose_extractor, segmentation_map_extractor,
                         ct_dose_seg_extractor,
                        #  features_extractor,
                         label_extractor)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            # ct, rtdose, segmentation_map, features, labels = self.inputs_labels_from_batch(batch)
            ct_dose_seg,  labels = self.inputs_labels_from_batch(batch) # features, 
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            # ct, rtdose, segmentation_map, features, labels = self.inputs_labels_from_batch(batch)
            ct_dose_seg, labels = self.inputs_labels_from_batch(batch) # features, 

        # return ct, rtdose, segmentation_map, features, labels
        return ct_dose_seg, labels # features, 


class ValDataLoaderIter(DataLoaderIter):
    """This iterator will reset itself **only** when it is acquired by
    the syntax of normal `iterator`. That is, this iterator just works
    like a `torch.data.DataLoader`. If you want to restart it, you
    should use it like:

        ```
        loader_iter = ValDataLoaderIter(data_loader)
        for batch in loader_iter:
            ...

        # `loader_iter` should run out of values now, you can restart it by:
        # 1. the way we use a `torch.data.DataLoader`
        for batch in loader_iter:        # __iter__ is called implicitly
            ...

        # 2. passing it into `iter()` manually
        loader_iter = iter(loader_iter)  # __iter__ is called by `iter()`
        ```
    """

    def __init__(self, data_loader: DataLoader,
                 # image_extractor: Callable,
                 # ct_extractor: Callable, rtdose_extractor: Callable, segmentation_map_extractor: Callable,
                 ct_dose_seg_extractor: Callable,
                #  features_extractor: Callable,
                 label_extractor: Callable) -> None:
        super().__init__(data_loader,
                         # image_extractor,
                         # ct_extractor, rtdose_extractor, segmentation_map_extractor,
                         ct_dose_seg_extractor,
                        #  features_extractor,
                         label_extractor)
        self.run_limit = len(self.data_loader)
        self.run_counter = 0

    def __iter__(self):
        if self.run_counter >= self.run_limit:
            self._iterator = iter(self.data_loader)
            self.run_counter = 0
        return self

    def __next__(self):
        self.run_counter += 1
        return super().__next__()


def default_image_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting ct from batch data.
    """
    out: torch.Tensor = x['image'] if isinstance(x, dict) else x[5]
    return out


def default_ct_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting ct from batch data.
    """
    out: torch.Tensor = x['ct'] if isinstance(x, dict) else x[0]
    return out


def default_rtdose_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting rtdose from batch data.
    """
    out: torch.Tensor = x['rtdose'] if isinstance(x, dict) else x[1]
    return out


def default_segmentation_map_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting segmentation_map from batch data.
    """
    out: torch.Tensor = x['segmentation_map'] if isinstance(x, dict) else x[2]
    return out


def default_ct_dose_seg_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting ct_dose_seg from batch data.
    """
    out: torch.Tensor = x['ct_dose_seg'] if isinstance(x, dict) else x[0]
    return out


def default_features_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting features from batch data.
    """
    out: torch.Tensor = x['features'] if isinstance(x, dict) else x[3]
    return out


def default_label_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting label from batch data.
    """
    out: torch.Tensor = x['pos'] if isinstance(x, dict) else x[4]
    return out

def default_fixed_moving_extractor(x: Any) -> torch.Tensor:
    """
    Default callable for getting fixed_moving from batch data.
    """
    out: torch.Tensor = x['fixed_moving'] if isinstance(x, dict) else x[0]
    return out


class LearningRateFinder:
    """
    Learning rate range test.

    The learning rate range test increases the learning rate in a pre-training run
    between two boundaries in a linear or exponential manner. It provides valuable
    information on how well the network can be trained over a range of learning rates
    and what is the optimal learning rate.
    References:
    Modified from: https://github.com/davidtvs/pytorch-lr-finder.
    Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        tp: dict,
        criterion: torch.nn.Module,
        logger,
        memory_cache: bool = True,
        cache_dir: Optional[str] = None,
        amp: bool = False,
        pickle_module=pickle,
        pickle_protocol: int = DEFAULT_PROTOCOL,
        verbose: bool = True,
    ) -> None:
        """
        Constructor.
        """
        # Check if the optimizer is already attached to a scheduler
        self.optimizer = optimizer
        self.optimizer_name = tp['optimizer_name']
        self._check_for_scheduler()
        self.tp = tp
        self.model = model
        self.criterion = criterion
        self.perform_augmix = tp['perform_augmix']
        self.mixture_width = tp['mixture_width']
        self.mixture_depth = tp['mixture_depth']
        self.augmix_strength = tp['augmix_strength']
        self.grad_max_norm = tp['grad_max_norm']
        self.history: Dict[str, list] = {'lr': [], 'loss': []}
        self.memory_cache = memory_cache
        self.cache_dir = cache_dir
        self.amp = amp
        self.verbose = verbose
        self.logger = logger
        device = tp['device']

        # Save the original state of the model and optimizer so they can be restored if
        # needed
        self.model_device = next(self.model.parameters()).device
        self.state_cacher = StateCacher(
            in_memory=memory_cache, cache_dir=cache_dir
        )  # , pickle_module=pickle_module, pickle_protocol=pickle_protocol)
        self.state_cacher.store('model', self.model.state_dict())
        self.state_cacher.store('optimizer', self.optimizer.state_dict())

        # If device is None, use the same as the model
        self.device = device if device else self.model_device

    def reset(self) -> None:
        """Restores the model and optimizer to their initial states."""

        self.model.load_state_dict(self.state_cacher.retrieve('model'))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer'))
        self.model.to(self.model_device)

    def range_test(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # image_extractor: Callable = default_image_extractor,
        # ct_extractor: Callable = default_ct_extractor,
        # rtdose_extractor: Callable = default_rtdose_extractor,
        # segmentation_map_extractor: Callable = default_segmentation_map_extractor,
        ct_dose_seg_extractor: Callable = default_fixed_moving_extractor,
        # features_extractor: Callable = default_features_extractor,
        label_extractor: Callable = default_label_extractor,
        start_lr: Optional[float] = None,
        end_lr: int = 10,
        num_iter: int = 100,
        step_mode: str = 'exp',
        smooth_f: float = 0.05,
        diverge_th: int = 5,
        accumulation_steps: int = 1,
        non_blocking_transfer: bool = True,
        auto_reset: bool = True,
        am_obj: object = None,
    ) -> None:
        """
        Performs the learning rate range test.
        """

        # Reset test results
        self.history = {'lr': [], 'loss': []}
        best_loss = -float('inf')

        # Move the model to the proper device
        self.model.to(self.device)

        # Check if the optimizer is already attached to a scheduler
        self._check_for_scheduler()

        # Set the starting learning rate
        if start_lr:
            self._set_learning_rate(start_lr)

        # Check number of iterations
        if num_iter <= 1:
            raise ValueError('`num_iter` must be larger than 1.')

        # Initialize the proper learning rate policy
        lr_schedule: Union[ExponentialLR, LinearLR]
        if step_mode.lower() == 'exp':
            lr_schedule = ExponentialLR(self.optimizer, end_lr, num_iter)
        elif step_mode.lower() == 'linear':
            lr_schedule = LinearLR(self.optimizer, end_lr, num_iter)
        else:
            raise ValueError(f'Expected one of (exp, linear), got {step_mode}.')

        if smooth_f < 0 or smooth_f >= 1:
            raise ValueError('Smooth_f is outside the range [0, 1].')

        # Create an iterator to get data batch by batch
        train_iter = TrainDataLoaderIter(train_loader,
                                         # image_extractor,
                                         # ct_extractor, rtdose_extractor, segmentation_map_extractor,
                                         ct_dose_seg_extractor,
                                          label_extractor) # features_extractor,
        if val_loader:
            val_iter = ValDataLoaderIter(val_loader,
                                         # image_extractor,
                                         # ct_extractor, rtdose_extractor, segmentation_map_extractor,
                                         ct_dose_seg_extractor,
                                         label_extractor) # features_extractor, 

        trange: Union[partial[tqdm.trange], Type[range]]
        if self.verbose and has_tqdm:
            trange = partial(tqdm.trange, desc='Computing optimal learning rate.')
            tprint = tqdm.tqdm.write
        else:
            trange = range
            tprint = print

        for iteration in trange(num_iter):
            if self.verbose and not has_tqdm:
                print(f'Computing optimal learning rate, iteration {iteration + 1}/{num_iter}.')

            # Train on batch and retrieve loss
            loss = self._train_batch(train_iter, accumulation_steps, am_obj, self.tp, non_blocking_transfer=non_blocking_transfer)
            if val_loader:
                loss = self._validate(val_iter, non_blocking_transfer=non_blocking_transfer)

            # Update the learning rate
            self.history['lr'].append(lr_schedule.get_lr()[0])
            lr_schedule.step()

            # Track the best loss and smooth it if smooth_f is specified
            if iteration == 0:
                best_loss = loss
            else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * self.history['loss'][-1]
                if loss < best_loss:
                    best_loss = loss

            # Check if the ; if it has, stop the test
            self.history['loss'].append(loss)
            if loss > diverge_th * best_loss:
                self.logger.my_print('Loss has diverged: stop the test.')
                self.logger.my_print('Loss: {}.'.format(loss))
                self.logger.my_print('Best_loss: {}.'.format(best_loss))
                if self.verbose:
                    tprint('Stopping early, the loss has diverged.')
                break

        if auto_reset:
            if self.verbose:
                self.logger.my_print('Resetting model and optimizer.')
            self.reset()

    def _set_learning_rate(self, new_lrs: Union[float, list]) -> None:
        """
        Set learning rate(s) for optimizer.
        """
        if not isinstance(new_lrs, list):
            new_lrs = [new_lrs] * len(self.optimizer.param_groups)
        if len(new_lrs) != len(self.optimizer.param_groups):
            raise ValueError(
                'Length of `new_lrs` is not equal to the number of parameter groups in the given optimizer.'
            )

        for param_group, new_lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = new_lr

    def _check_for_scheduler(self):
        """
        Make sure that optimizer does not already have scheduler.
        """
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                raise RuntimeError('Optimizer already has a scheduler attached to it.')


    def _train_batch(self, train_iter, accumulation_steps: int, am_obj, tp, non_blocking_transfer: bool = True) -> float:
        self.model.train()
        total_loss = 0

        # Zero the parameter gradients
        self.optimizer.zero_grad(set_to_none=True)
        for i in range(accumulation_steps):
            ct_dose_seg, labels = next(train_iter) # features,
            inputs, labels = ( #features, 
                ct_dose_seg.to(self.device), labels.to(self.device) #  features.to(self.device),
            )


            # NOTE: ADD AUGMIX LATER
            if self.perform_augmix:
                for b in range(len(inputs)):
                    # Generate a random 32-bit integer
                    seed_b = random.getrandbits(32)
                    inputs[b] = am_obj.aug_mix(arr=inputs[b],
                                                     mixture_width=self.mixture_width,
                                                     mixture_depth=self.mixture_depth,
                                                     augmix_strength=self.augmix_strength,
                                                     seed=seed_b, tp=tp)

            # Forward pass
            outputs = self.model(x_fixed= inputs['fixed'], x_moving=['moving']) #features=features)
            try:
                # Cross-Entropy, Rank, Custom
                loss = self.criterion(outputs, labels)
            except:
                # BCE
                loss = self.criterion(outputs, torch.reshape(labels, outputs.shape).to(outputs.dtype))

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            if self.amp and hasattr(self.optimizer, '_amp_stash'):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with torch.cuda.amp.scale_loss(loss, self.optimizer, delay_unscale=delay_unscale) as scaled_loss:  # type: ignore
                    scaled_loss.backward()
            elif self.optimizer_name in ['ada_hessian']:
                # https://github.com/pytorch/pytorch/issues/4661
                # https://discuss.pytorch.org/t/how-to-backward-the-derivative/17662?u=bpetruzzo
                # Using backward() with create_graph=True will create a reference cycle between the parameter and its gradient
                # which can cause a memory leak. We recommend using autograd.grad when creating the graph to avoid this.
                # However, if we do use backward(create_graph=True), then we have to make sure to reset the
                # .grad fields of our parameters to None after use to break the cycle and avoid the leak.
                loss.backward(create_graph=True)
                # torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            else:
                loss.backward()

            total_loss += loss.item()

        # Perform gradient clipping
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        if self.grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.grad_max_norm)

        self.optimizer.step()

        # Reset the .grad fields of your parameters to None after use to break the cycle and avoid the memory leak
        if self.optimizer_name in ['ada_hessian']:
            for p in self.model.parameters():
                p.grad = None

        return total_loss

    def _validate(self, val_iter: ValDataLoaderIter, non_blocking_transfer: bool = True) -> float:
        # Set model to evaluation mode and disable gradient computation
        running_loss = 0
        with eval_mode(self.model):
            # for ct, rtdose, segmentation_map, features, labels in val_iter:
            for ct_dose_seg,  labels in val_iter: # features,
                # ct, rtdose, segmentation_map, features, labels = copy_to_device(
                #     [ct, rtdose, segmentation_map, features, labels],
                #     device=self.device, non_blocking=non_blocking_transfer)
                # ct, rtdose, segmentation_map, features, labels = (
                #     ct.to(self.device), rtdose.to(self.device), segmentation_map.to(self.device),
                #     features.to(self.device), labels.to(self.device)
                # )
                inputs,  labels = ( # features,
                    ct_dose_seg.to(self.device),  labels.to(self.device) # features.to(self.device),
                )

                # # Preprocess inputs and features
                # inputs = process_data.preprocess_inputs(inputs)
                # features = process_data.preprocess_features(features)

                # Forward pass and loss computation
                outputs = self.model(x_fixed= inputs['fixed'], x_moving=['moving'])
                try:
                    # Cross-Entropy, Rank, Custom
                    loss = self.criterion(outputs, labels)
                except:
                    # BCE
                    loss = self.criterion(outputs, torch.reshape(labels, outputs.shape).to(outputs.dtype))
                running_loss += loss.item() * len(labels)

        return running_loss / len(val_iter.dataset)

    def get_lrs_and_losses(self, skip_start: int = 0, skip_end: int = 0) -> Tuple[list, list]:
        """
        Get learning rates and their corresponding losses.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the end.
        """
        if skip_start < 0:
            raise ValueError('Skip_start cannot be negative.')
        if skip_end < 0:
            raise ValueError('Skip_end cannot be negative.')

        lrs = self.history['lr']
        losses = self.history['loss']
        end_idx = len(lrs) - skip_end - 1
        lrs = lrs[skip_start:end_idx]
        losses = losses[skip_start:end_idx]

        return lrs, losses

    def get_steepest_gradient(
        self, skip_start: int = 0, skip_end: int = 0
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        """
        Get learning rate which has steepest gradient and its corresponding loss.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the end.

        Returns:
            Learning rate which has steepest gradient and its corresponding loss
        """
        lrs, losses = self.get_lrs_and_losses(skip_start, skip_end)

        # Determine global minimum: only compute steepest gradient to the left of the global minimum
        # Note: if global_min occurs multiple times in losses, then .index() will consider the most-left index, so
        # .index() is doing what we want
        global_min = min(losses)
        global_min_idx = losses.index(global_min)
        losses = losses[:global_min_idx]
        lrs = lrs[:global_min_idx]

        try:
            min_grad_idx = np.gradient(np.array(losses)).argmin()
            return lrs[min_grad_idx], losses[min_grad_idx]
        except ValueError:
            self.logger.my_print('Failed to compute the gradients, there might not be enough points.')
            return None, None

    def plot(self, skip_start: int = 0, skip_end: int = 0, log_lr: bool = True, ax=None, steepest_lr: bool = True):
        """
        Plots the learning rate range test.

        Args:
            skip_start: number of batches to trim from the start.
            skip_end: number of batches to trim from the start.
            log_lr: True to plot the learning rate in a logarithmic
                scale; otherwise, plotted in a linear scale.
            ax: the plot is created in the specified matplotlib axes object and the
                figure is not be shown. If `None`, then the figure and axes object are
                created in this method and the figure is shown.
            steepest_lr: plot the learning rate which had the steepest gradient.

        Returns:
            The `matplotlib.axes.Axes` object that contains the plot. Returns `None` if
            `matplotlib` is not installed.
        """
        if not has_matplotlib:
            warnings.warn('Matplotlib is missing, cannot plot result.')
            return None

        lrs, losses = self.get_lrs_and_losses(skip_start, skip_end)

        # Create the figure and axes object if axes was not already given
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Plot loss as a function of the learning rate
        ax.plot(lrs, losses)

        # Plot the LR with steepest gradient
        if steepest_lr:
            lr_at_steepest_grad, loss_at_steepest_grad = self.get_steepest_gradient(skip_start, skip_end)
            if lr_at_steepest_grad is not None:
                ax.scatter(
                    lr_at_steepest_grad,
                    loss_at_steepest_grad,
                    s=75,
                    marker='o',
                    color='red',
                    zorder=3,
                    label='steepest gradient',
                )
                ax.legend()

        if log_lr:
            ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')

        # Show only if the figure was created internally
        if fig is not None:
            plt.show()

        return ax


class LearningRateWarmUp(object):
    """
    From https://github.com/developer0hye/Learning-Rate-WarmUp.
    Source: https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean

    Note: for LearningRateWarmUp we need scheduler.step() to be applied after every batch in main.py, otherwise
        the warmup will not be effective.
    """

    def __init__(self, optimizer, tp, after_scheduler=None):
        self.optimizer = optimizer
        
        self.warmup_batches = tp['warmup_batches']
        self.num_batches_per_epoch = tp['batch_size']
        self.target_lr = tp['target_lr']
        self.after_scheduler = after_scheduler

        self.warmup_epochs = self.warmup_batches / self.num_batches_per_epoch  # as float
        self.cur_iteration = 0
        self.step()

    def warmup_learning_rate(self):
        warmup_lr = self.target_lr * float(self.cur_iteration) / float(self.warmup_batches)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_epoch=None):
        if cur_epoch is None:
            # E.g. cyclic scheduler: scheduler.step()
            self.cur_iteration += 1
        else:
            # E.g. cosine/exponential/step scheduler: scheduler.step(epoch + (i + 1) / train_num_iterations)
            self.cur_iteration = cur_epoch * self.num_batches_per_epoch

        if self.cur_iteration <= self.warmup_batches:
            self.warmup_learning_rate()
        else:
            self.after_scheduler.step(cur_epoch)

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)