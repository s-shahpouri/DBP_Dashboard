import random
import numpy as np
import itk
import torch
from monai.transforms import (
    MapTransform, Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    Spacingd, SpatialPadd, CenterSpatialCropd, EnsureTyped, RandWeightedCropd, Rand3DElasticd
)
from monai.transforms import ToDeviced
from monai.data.image_reader import ITKReader
from monai.data import CacheDataset, DataLoader

class Logger:
    def my_print(self, message):
        print(message)

class TransformGenerator:

    def call_generic_transform_1(self, trial_parameters):
        return Compose([
            LoadImaged(keys=trial_parameters['image_keys'][:2], reader=ITKReader()),
            EnsureChannelFirstd(keys=trial_parameters['image_keys'][:2]),
            NormalizeIntensityd(keys=trial_parameters['image_keys'][:2]),
            Spacingd(keys=trial_parameters['image_keys'][:2], pixdim=trial_parameters['spacing'], mode=trial_parameters['ct_interpol_mode_3d']),
            SpatialPadd(keys=trial_parameters['image_keys'][:2], spatial_size=trial_parameters['output_size'], mode='constant'),
            CenterSpatialCropd(keys=trial_parameters['image_keys'][:2], roi_size=trial_parameters['output_size']),
        ])

    def call_aug_transform_1(self, trial_parameters):
        weighted_crop_transform = Compose([
            CreateSingleWeightMapd(keys=trial_parameters['image_keys'][:2], w_key='weight_map'),  # a simple label for random cropping
            EnsureTyped(keys=trial_parameters['image_keys']),
            RandWeightedCropd(
                keys=trial_parameters['image_keys'][:2],
                w_key='weight_map',
                spatial_size=trial_parameters['cropping_dim'],
                num_samples=trial_parameters['num_sample']),  # Random cropping
            SpatialPadd(keys=trial_parameters['image_keys'][:2], spatial_size=trial_parameters['output_size'], mode='constant'),
        ])

        aug_transforms = Compose([
            RandomCoordinationTransform(
                keys=trial_parameters['image_keys'][:2],
                coord_key='pos',
                num_samples=trial_parameters['num_coordinations'],
                low=-15, high=15,
                pixdim=trial_parameters['spacing'],
                prob=trial_parameters.get('coord_prob', 0.5)  # Random translocation generator
            ),
            Rand3DElasticd(
                keys=trial_parameters['image_keys'][:2],
                sigma_range=trial_parameters['sigma_elastic'],
                magnitude_range=trial_parameters['magnitude_elastic'],
                shear_range=None,
                mode='nearest', padding_mode='zeros',
                prob=trial_parameters.get('elastic_prob', 0.5)),  # Random elastic deformation
            ConditionalTransform(
                keys=trial_parameters['image_keys'][:2],
                transform=weighted_crop_transform,
                prob=trial_parameters.get('crop_prob', 0.5)),  # Creating probability conditional for random cropping
        ])
        return aug_transforms

    def get_general_transforms(self, trial_parameters):
        """
        Generates the general (basic) transformations.
        """
        func_name = trial_parameters['generic_transformation']
        print(func_name)
        try:
            func = getattr(self, func_name)
            gt = func(trial_parameters)
        except AttributeError:
            print(f'Method {func_name} not found in TransformGenerator.')
        except Exception as e:
            print(f'An error occurred while calling {func_name}: {e}')
        return gt        

    def get_augmentation_transforms(self, trial_parameters):
        """
        Generates augmentation transformations based on the trial parameters.
        """
        func_name = trial_parameters['augmentation_function']
        try:
            func = getattr(self, func_name)
            aug_transforms = func(trial_parameters)
        except AttributeError:
            print(f'Method {func_name} not found in TransformGenerator.')
        except Exception as e:
            print(f'An error occurred while calling {func_name}: {e}')
        return aug_transforms

    def generate_transforms(self, trial_parameters, logger):
        """
        Generates both general and augmentation transformations and returns them.
        """
        logger.my_print(f"To_device: {trial_parameters['to_device']}.")

        # Get general transforms
        generic_transforms = self.get_general_transforms(trial_parameters)

        # Get augmentation transforms
        aug_transforms = self.get_augmentation_transforms(trial_parameters)

        # Validation transforms: only generic
        val_transforms = generic_transforms.flatten()

        # Training transforms: generic + augmentation
        train_transforms = Compose([generic_transforms, aug_transforms]).flatten()

        # Apply device transformation if needed
        if trial_parameters['to_device']:
            train_transforms = Compose([
                train_transforms,
                ToDeviced(keys=trial_parameters['image_keys'][:2] + ['weight_map'], device=trial_parameters['device']),
            ]).flatten()
            val_transforms = Compose([
                val_transforms,
                ToDeviced(keys=trial_parameters['image_keys'][:2] + ['weight_map'], device=trial_parameters['device']),
            ]).flatten()
            pass

        logger.my_print('Transformers have been made successfully.')

        return train_transforms, val_transforms




from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    Spacingd, SpatialPadd, CenterSpatialCropd, RandWeightedCropd,
    EnsureTyped, MapTransform, Rand3DElasticd
)
from monai.data import CacheDataset, DataLoader, ITKReader
import numpy as np
import torch
import random

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, 
    Spacingd, SpatialPadd, CenterSpatialCropd, RandWeightedCropd,
    EnsureTyped, MapTransform, Rand3DElasticd
)
from monai.data import CacheDataset, DataLoader, ITKReader
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import itk



def trans(img):
    return np.rot90(np.flipud(img), k=1)

class ConditionalTransform(MapTransform):
    def __init__(self, keys, transform, prob, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.transform = transform
        self.prob = prob

    def __call__(self, data):
        if random.random() < self.prob:
            data = self.transform(data)
        return data


class CreateSingleWeightMapd(MapTransform):
    def __init__(self, keys, w_key, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.w_key = w_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            weight_map = np.zeros_like(img)

            w = 20
            # Custom weight map logic
            x1 = int((img.shape[1]/2) - w)
            x2 = int((img.shape[1]/2) + w)
            weight_map[:, x1:x2, x1:x2] = 1.0

            d[self.w_key] = weight_map
        return d


    def create_weight_map(self, image):
        # Example of creating a weight map
        # This should be replaced with the actual logic to create a weight map
        weight_map = np.ones_like(image, dtype=np.float32)
        return weight_map


import numpy as np
import itk
import torch
from monai.transforms import MapTransform

class RandomCoordinationTransform(MapTransform):
    def __init__(self, keys, coord_key='pos', num_samples=4, low=-15, high=15, pixdim=(2.0, 2.0, 2.0), prob=0.5, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.coord_key = coord_key
        self.num_samples = num_samples
        self.low = low
        self.high = high
        self.pixdim = pixdim
        self.prob = prob

    def generate_random_translation_vector(self, size):
        return np.random.uniform(self.low, self.high, size)

    def __call__(self, data):
        d = dict(data)
        new_samples = []

        if random.random() > self.prob:
            return [d]

        for i in range(self.num_samples):
            new_data = {key: d[key] for key in self.keys}
            coordination = np.array(d[self.coord_key])
            new_translation = self.generate_random_translation_vector(coordination.size)
            updated_translation = coordination - new_translation

            fixed_CT_array = d["plan"].numpy().squeeze()  # Convert tensor to numpy array and remove channel dimension
            moving_CT_array = d["repeat"].numpy().squeeze()  # Convert tensor to numpy array and remove channel dimension

            # Perform transformation
            transformed_moving_CT_array_updated = transform_moving_ct(
                moving_CT_array, fixed_CT_array, updated_translation, self.pixdim
            )

            new_data["repeat"] = torch.tensor(transformed_moving_CT_array_updated[np.newaxis, ...])  # Convert numpy array back to tensor and add channel dimension
            new_data["plan"] = torch.tensor(fixed_CT_array[np.newaxis, ...])  # Convert numpy array back to tensor and add channel dimension
            new_data[self.coord_key] = new_translation
            
            new_samples.append(new_data)

        return new_samples

def transform_moving_ct(moving_CT_array, fixed_CT_array, coordination, pixdim):
    if len(coordination) != 3:
        raise ValueError(f"Expected coordination of length 3, but got {len(coordination)}")

    # Convert NumPy arrays to ITK images
    fixed_CT_image = itk.image_from_array(fixed_CT_array)
    moving_CT_image = itk.image_from_array(moving_CT_array)
    fixed_CT_image.SetSpacing(pixdim)
    moving_CT_image.SetSpacing(pixdim)

    translation_updated = itk.TranslationTransform[itk.D, 3].New()
    translation_updated.SetOffset(np.array([coordination[2], coordination[1], coordination[0]], dtype=np.float64))

    resampler_updated = itk.ResampleImageFilter.New(Input=moving_CT_image, Transform=translation_updated, UseReferenceImage=True, ReferenceImage=fixed_CT_image)
    resampler_updated.SetInterpolator(itk.LinearInterpolateImageFunction.New(fixed_CT_image))

    resampler_updated.Update()
    transformed_moving_CT_image_updated = resampler_updated.GetOutput()
    transformed_moving_CT_array_updated = itk.array_view_from_image(transformed_moving_CT_image_updated)

    return transformed_moving_CT_array_updated





