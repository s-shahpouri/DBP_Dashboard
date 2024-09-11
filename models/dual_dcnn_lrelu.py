"""
Same as DCNN, but where LeakyReLU activation has been applied whenever possible.

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
import math
import torch
import torch
import numpy as np
from functools import reduce
from operator import __add__
import os


class Output(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Output, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.fc = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x):
        out = self.fc(x)
        return out 
    

class conv3d_padding_same(torch.nn.Module):
    """
    Padding so that the next Conv3d layer outputs an array with the same dimension as the input.
    Depth, height and width are the kernel dimensions.

    Example:
    import torch
    from functools import reduce
    from operator import __add__

    batch_size = 8
    in_channels = 3
    out_channel = 16
    kernel_size = (2, 3, 5)
    stride = 1  # could also be 2, or 3, etc.
    pad_value = 0
    conv = torch.nn.Conv3d(in_channels, out_channel, kernel_size, stride=stride)

    x = torch.empty(batch_size, in_channels, 100, 100, 100)
    conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
    y = torch.nn.functional.pad(x, conv_padding, 'constant', pad_value)

    out = conv(y)
    print(out.shape): torch.Size([8, 16, 100, 100, 100])

    Source: https://stackoverflow.com/questions/58307036/is-there-really-no-padding-same-option-for-pytorchs-conv2d
    Source: https://pytorch.org/docs/master/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    """

    def __init__(self, depth, height, width, pad_value):
        super(conv3d_padding_same, self).__init__()
        self.kernel_size = (depth, height, width)
        self.pad_value = pad_value

    def forward(self, x):
        # Determine amount of padding
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes.
        conv_padding = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]])
        # conv_padding  = tuple((np.array(self.kernel_size) - 1) // 2)
        # print('HIIIIIIIIIIIIIIIIIIIIII', self.pad_value,conv_padding, self.kernel_size, x.shape)
        x_padded = torch.nn.functional.pad(x, conv_padding, 'constant', self.pad_value)

        return x_padded


class conv_block_one(torch.nn.Module):
    """
    Type: pytorch class
    inputs: in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False
    Explanation: This conv block contains conv3d with padding the same, instanceNorm3d and Leakyrelu as the activation layer.
                 Also, it contains two repeated layers.
    """
    def __init__(self, in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False):
        super(conv_block_one, self).__init__()

        if ((type(kernel_size) == list) or (type(kernel_size) == tuple)) and (len(kernel_size) == 3):
            kernel_depth = kernel_size[0]
            kernel_height = kernel_size[1]
            kernel_width = kernel_size[2]
        elif type(kernel_size) == int:
            kernel_depth = kernel_size
            kernel_height = kernel_size
            kernel_width = kernel_size
        else:
            raise ValueError("Kernel_size is invalid:", kernel_size)

        self.pad = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=pad_value)
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=use_bias)
        self.norm1 = torch.nn.InstanceNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, stride=1,
                                     bias=use_bias)
        self.norm2 = torch.nn.InstanceNorm3d(filters)
        self.use_activation = use_activation
        self.activation2 = torch.nn.LeakyReLU(negative_slope=lrelu_alpha)

    def forward(self, x):

        x = self.pad(x)
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.activation1(x)
        x = self.pad(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_activation:
            x = self.activation2(x)
        return x


class conv_block_two(torch.nn.Module):
    """
    Type: pytorch class
    inputs: in_channels, filters, kernel_size, strides, pad_value, lrelu_alpha, use_activation,
                 use_bias=False
    Explanation: This conv block contains conv3d with padding the same, BatchNormalization, Leakyrelu as the activation layer, and 3dMaxPool.
                 Also, it contains two repeated layers.
                 NOTE:In this block,The kernel size, filters and strides are the same for two Conc blocks.
                 
    reference: https://www.nature.com/articles/s41598-021-81044-7
    """
    def __init__(self, in_channels_list, tp, i):      
        super(conv_block_two, self).__init__()

        # Extract the proper hyperparameters for this layer
        in_channels = in_channels_list[i]
        filters = tp['filters'][i]
        strides = tp['strides'][i]
        kernel_size = tp['kernel_sizes'][i]

        # Validate and extract kernel dimensions
        if isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 3:
            kernel_depth, kernel_height, kernel_width = kernel_size
        elif isinstance(kernel_size, int):
            kernel_depth = kernel_height = kernel_width = kernel_size
        else:
            raise ValueError(f"Invalid kernel_size: {kernel_size}")
        
        # conv_padding  = tuple((np.array(kernel_size) - 1) // 2)
        
        
        # First layer
        self.pad1 = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=tp['pad_value'])
        self.conv1 = torch.nn.Conv3d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size,
                                     stride=strides, bias=tp['use_bias'])
        self.norm1 = torch.nn.BatchNorm3d(filters)
        self.activation1 = torch.nn.LeakyReLU(negative_slope=tp['lrelu_alpha'])
        self.use_activation1 = tp['perform_activation'][0]
        self.use_max_pool1 = tp['perform_max_pool'][0]

        # Second layer
        self.pad2 = conv3d_padding_same(depth=kernel_depth, height=kernel_height, width=kernel_width,
                                       pad_value=tp['pad_value'])
        self.conv2 = torch.nn.Conv3d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                                     stride=1, bias=tp['use_bias']) # NOTE:I changed the stride of the second layer to 1 !!! for now it is alright.
        self.norm2 = torch.nn.BatchNorm3d(filters)
        self.activation2 = torch.nn.LeakyReLU(negative_slope=tp['lrelu_alpha'])

        # Initialize Max Pooling
        self.max_pooling = torch.nn.MaxPool3d(kernel_size=tp['kernel_size_pool'], stride=tp['stride_pool'], padding=tp['padding_pool'], 
                                              dilation=1, return_indices=False, ceil_mode=False)
        
        self.use_activation2 = tp['perform_activation'][1]
        self.use_max_pool2 = tp['perform_max_pool'][1]

    def forward(self, x):
        
        # first layer
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        
        if self.use_activation1:
            x = self.activation1(x)

        # if self.use_max_pool1:
        #     x = self.max_pooling(x)
                    
        # Seconf layer 
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        if self.use_activation2:
            x = self.activation2(x)

        # if self.use_max_pool2:
        #     x = self.max_pooling(x) 
        
        return x
    

class Dual_DCNN_LReLU(torch.nn.Module):
    """
    Simple Dual Deep CNN
    NOTE: it is NOT a multi-model architecture. If you want to add 
    """

    def __init__(self, trial_parameters): ############ Some input variables that we have them in the config file
        super(Dual_DCNN_LReLU, self).__init__()
        self.tp = trial_parameters
                
        # Determine number of downsampling blocks
        n_down_blocks = [sum([x[0] == 2 for x in self.tp['strides']]), 
                         sum([x[1] == 2 for x in self.tp['strides']]),
                         sum([x[2] == 2 for x in self.tp['strides']])]
        
           
        # Determine Linear input channel size ############# Make this using config file
        end_height  = math.ceil(self.tp['output_size'][0] / (2 ** n_down_blocks[0]))
        end_depth = math.ceil(self.tp['output_size'][1] / (2 ** n_down_blocks[1]))
        end_width = math.ceil(self.tp['output_size'][2] / (2 ** n_down_blocks[2]))

        # Initialize conv blocks ############  Learn about in channels (Why do we have the input channels as the first filter)
        in_channels = [self.tp['n_input_channels']] + self.tp['filters'][:-1]
        
        self.conv_blocks_fixed = torch.nn.ModuleList()
        self.conv_blocks_moving = torch.nn.ModuleList()
        
        # Initialize the blocks of fixed 
        for i in range(len(in_channels)):
            self.conv_blocks_fixed.add_module(f'fixed_conv_block{i}',
                                        conv_block_two(in_channels, self.tp, i))
            self.conv_blocks_moving.add_module(f'moving_conv_block{i}',
                                        conv_block_two(in_channels, self.tp, i))

        # Initialize pooling conv
        if self.tp['perform_pooling']:
            self.pool_fixed = torch.nn.AdaptiveAvgPool3d(kernel_size=(end_depth, end_height, end_width))
            self.pool_moving = torch.nn.AdaptiveAvgPool3d(kernel_size=(end_depth, end_height, end_width))
            end_depth, end_height, end_width = 1, 1, 1

        # Initialize flatten layer
        # self.flatten = torch.nn.Flatten()

        # print('HIIIIIIIIIIIIIIIIIIIIIIIIIII', end_depth, end_height, end_width)
        # Initialize linear layers 
        self.linear_layers = torch.nn.ModuleList()
        # NOTE: WHEN WE HAVE DUAL MODEL THE FIRST INPUT LAYER OF LINEAR UNITS SHOULD BE MULTIPLY TO 2 (2 ARM)
        linear_units = [2 * end_depth * end_height * end_width * self.tp['filters'][-1]] + self.tp['linear_units'] 
        
        # Assemble each linear layer 
        # NOTE: I do NOT want to have any activation layer after my linear units since I want to preserve my negative values.
        for i in range(len(linear_units) - 1):
            self.linear_layers.add_module(f'dropout{i}', torch.nn.Dropout(self.tp['dropout_p_j'][i]))
            self.linear_layers.add_module(f'linear{i}',
                                          torch.nn.Linear(in_features=linear_units[i], out_features=linear_units[i+1],
                                                          bias=self.tp['use_bias']))
            # self.linear_layers.add_module(f'lrelu{i}', torch.nn.LeakyReLU(negative_slope=self.tp['lrelu_alpha']))

        self.n_sublayers_per_linear_layer = len(self.linear_layers) / (len(linear_units) - 1)
        
        # Initialize output layer
        self.out_layer = Output(in_features=linear_units[-1], out_features=3,
                                    bias=self.tp['use_bias'])
        

    def forward(self, x_fixed, x_moving):
        
        # Conv Blocks
        for block_fixed, block_moving in zip(self.conv_blocks_fixed, self.conv_blocks_moving):
            print(x_fixed.shape, x_moving.shape)
            x_fixed = block_fixed(x_fixed)
            x_moving = block_moving(x_moving)
            # block_path = '//zkh/appdata/RTDicom/Projectline_HNC_modelling/Users/Hooman Bahrdo/Checkers'
            # torch.save(x, os.path.join(block_path, f'block{counter}.pt'))
            
        # Pooling layers
        if self.tp['perform_pooling']:
            x_fixed = self.pool_fixed(x_fixed)
            x_moving = self.pool_moving(x_moving)

        # Combine two blocks with each other
        x_fixed = x_fixed.view(x_fixed.size(0), -1)
        x_moving = x_moving.view(x_moving.size(0), -1)
        print(x_fixed.shape, x_moving.shape)
        x = torch.cat((x_fixed, x_moving), dim=1)        

        print(x.shape)
        # Linear layers
        for layer in self.linear_layers:
            x = layer(x)

        # Output
        x = self.out_layer(x)

        return x
    
    
    
if __name__ == '__main__':
    import json
    json_file_name = 'main_parameters.json'
    with open (json_file_name, 'r') as config_name:
        config_file = json.load(config_name)

    param_dict = config_file
    model = Dual_DCNN_LReLU(param_dict)
