"""
Explanation: This module is for implementing augmix in the code.
"""

import torch
import random
import numpy as np
from monai.transforms import (
    Compose,
    Rand3DElastic,
    RandAdjustContrast,
    RandFlip,
    RandGaussianNoise,
    RandAffine,
    RandRotate,
    RandShiftIntensity,
    ScaleIntensityRange,
    NormalizeIntensity,
)


class AugMix():
    
    def __init__(self):
        self.aug_list = [self.translate, self.rotate, self.scale]
    
    def flip(self, arr, mode, strength, seed):
        augmenter = RandFlip(prob=1.0, spatial_axis=-1)
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)


    def translate(self, arr, mode, strength, seed):
        # augmenter = RandAffine(prob=1.0, translate_range=(7 * strength, 7 * strength, 7 * strength),
        #                        padding_mode='border', mode=mode),  # 3D: (num_channels, H, W[, D])
        augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                                translate_range=(round(7 * strength), round(7 * strength), round(7 * strength)),
                                padding_mode='border', mode=mode)
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)


    def rotate(self, arr, mode, strength, seed):
        # augmenter = RandRotate(prob=1.0, range_x=(np.pi / 24) * strength, align_corners=True,
        #                        padding_mode='border', mode=mode),
        augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                                rotate_range=((np.pi / 24) * strength, (np.pi / 24) * strength, (np.pi / 24) * strength),
                                padding_mode='border', mode=mode)
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)


    def scale(self, arr, mode, strength, seed):
        # augmenter = RandAffine(prob=1.0, scale_range=(0.07 * strength, 0.07 * strength, 0.07 * strength),
        #                        padding_mode='border', mode=mode),  # 3D: (num_channels, H, W[, D])
        augmenter = Rand3DElastic(prob=1.0, sigma_range=(5, 8), magnitude_range=(0, 1),
                                scale_range=(0.07 * strength, 0.07 * strength, 0.07 * strength),
                                padding_mode='border', mode=mode)
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)


    def gaussian_noise(self, arr, mode, strength, seed):
        augmenter = RandGaussianNoise(prob=1.0, mean=0.0, std=0.02)
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)


    def intensity(self, arr, mode, strength, seed):
        # augmenter = RandShiftIntensity(prob=1.0, offsets=(0, 0.05 * strength))
        augmenter = RandAdjustContrast(prob=1.0, gamma=(0.9, 1))
        augmenter.set_random_state(seed=seed)
        return augmenter(arr)





    def aug_mix(self, arr, mixture_width, mixture_depth, augmix_strength, device, seed, tp):
        """
        Perform AugMix augmentations and compute mixture.

        Source: https://github.com/google-research/augmix/blob/master/augment_and_mix.py

        Args:
            arr: (preprocessed) Numpy array of size (3, depth, height, width)
            mixture_width: width of augmentation chain (cf. number of parallel augmentations)
            mixture_depth: range of depths of augmentation chain (cf. number of consecutive augmentations)
                OLD: -1 enables stochastic depth uniformly from [1, 3]
            augmix_strength:
            device:
            seed:

        Returns:
            mixed: Augmented and mixed image.
        """
        ws = torch.tensor(np.random.dirichlet([1] * mixture_width), dtype=torch.float32)
        m = torch.tensor(np.random.beta(1, 1), dtype=torch.float32)
        # m = np.random.uniform(0, 0.25)

        mix = torch.zeros_like(arr, device=device)
        for i in range(mixture_width):
            image_aug = arr.clone()

            # OLD
            # depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
            depth = np.random.randint(mixture_depth[0], mixture_depth[1] + 1)
            for _ in range(depth):
                # op = np.random.choice(aug_list)
                idx = random.randint(0, len(self.aug_list) - 1)
                op = self.aug_list[idx]
                seed_i = random.getrandbits(32)

                # CT Fixed
                image_aug[0] = op(arr=image_aug[0], mode=tp['ct_interpol_mode_2d'], strength=augmix_strength, seed=seed_i)
                # CT Moving
                # image_aug[1] = op(arr=image_aug[1], mode=tp['ct_interpol_mode_2d'], strength=augmix_strength,
                #                 seed=seed_i)
                # ############# Changing for WeeklyCTs #############
                # Segmentation
                # image_aug[2] = op(arr=image_aug[2], mode=tp['segmentation_interpol_mode_2d'], strength=augmix_strength,
                #                 seed=seed_i)

                # # WeeklyCT
                # image_aug[2] = op(arr=image_aug[2], mode=config.weeklyct_interpol_mode_2d, strength=augmix_strength, seed=seed_i)

            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * image_aug

        mixed = (1 - m) * arr + m * mix

        return mixed



    