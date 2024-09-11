"""
Explanation: This module contains all the loss functions used n this project.
NOTE: Go through the following link to find moe loss functions.
https://medium.com/@mlblogging.k/14-loss-functions-you-can-use-for-regression-b24db8dff987

Author: Hooman Bahrdo
Last Revision: 06/14/2024
"""
import torch
from torch.nn.functional import pairwise_distance


class SeparatedChannelMAELoss(torch.nn.Module):
    def __init__(self):
        super(SeparatedChannelMAELoss, self).__init__()

    def forward(self, pred, target):
        # Ensure the input tensors have the correct shape
        assert pred.shape == target.shape, "Pred and Target shapes must match"
        assert pred.shape[-1] == 3, "Last dimension of Pred and Target must be 3 (x, y, z)"
        
        # Separate the channels
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        
        # Compute the absolute differences for each channel
        loss_x = torch.abs(pred_x - target_x)
        loss_y = torch.abs(pred_y - target_y)
        loss_z = torch.abs(pred_z - target_z)
        
        # Combine the losses
        total_loss = torch.mean(loss_x) + torch.mean(loss_y) + torch.mean(loss_z)
        
        return total_loss


class SeparatedChannelMSELoss(torch.nn.Module):
    def __init__(self):
        super(SeparatedChannelMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shapes must match"
        assert pred.shape[-1] == 3, "Last dimension of Pred and Target must be 3 (x, y, z)"
        
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        
        loss_x = (pred_x - target_x) ** 2
        loss_y = (pred_y - target_y) ** 2
        loss_z = (pred_z - target_z) ** 2
        
        total_loss = torch.mean(loss_x) + torch.mean(loss_y) + torch.mean(loss_z)
        
        return total_loss


class SeparatedChannelChamferLoss(torch.nn.Module):
    def __init__(self):
        super(SeparatedChannelChamferLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shapes must match"
        assert pred.shape[-1] == 3, "Last dimension of Pred and Target must be 3 (x, y, z)"
        
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        
        def chamfer_distance_1d(a, b):
            a_expanded = a.unsqueeze(1)
            b_expanded = b.unsqueeze(0)
            dist_matrix = torch.cdist(a_expanded, b_expanded, p=2)
            min_dist_a_to_b = torch.min(dist_matrix, dim=1)[0]
            min_dist_b_to_a = torch.min(dist_matrix, dim=0)[0]
            return torch.mean(min_dist_a_to_b) + torch.mean(min_dist_b_to_a)
        
        loss_x = chamfer_distance_1d(pred_x, target_x)
        loss_y = chamfer_distance_1d(pred_y, target_y)
        loss_z = chamfer_distance_1d(pred_z, target_z)
        
        total_loss = loss_x + loss_y + loss_z
        
        return total_loss

class SeparatedChannelWeightedMSELoss(torch.nn.Module):
    def __init__(self, weight_x=1.0, weight_y=1.0, weight_z=1.0):
        super(SeparatedChannelWeightedMSELoss, self).__init__()
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_z = weight_z

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shapes must match"
        assert pred.shape[-1] == 3, "Last dimension of Pred and Target must be 3 (x, y, z)"
        
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        
        loss_x = torch.abs(pred_x - target_x) ** 2
        loss_y = torch.abs(pred_y - target_y) ** 2
        loss_z = torch.abs(pred_z - target_z) ** 2
        
        total_loss = (self.weight_x * torch.mean(loss_x) +
                      self.weight_y * torch.mean(loss_y) +
                      self.weight_z * torch.mean(loss_z))
        
        return total_loss

class SeparatedChannelWeightedMAELoss(torch.nn.Module):
    def __init__(self, weight_x=1.0, weight_y=1.0, weight_z=1.0):
        super(SeparatedChannelWeightedMAELoss, self).__init__()
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_z = weight_z

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Pred and Target shapes must match"
        assert pred.shape[-1] == 3, "Last dimension of Pred and Target must be 3 (x, y, z)"
        
        pred_x, pred_y, pred_z = pred[:, 0], pred[:, 1], pred[:, 2]
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        
        loss_x = torch.abs(pred_x - target_x)
        loss_y = torch.abs(pred_y - target_y)
        loss_z = torch.abs(pred_z - target_z)
        
        total_loss = (self.weight_x * torch.mean(loss_x) +
                      self.weight_y * torch.mean(loss_y) +
                      self.weight_z * torch.mean(loss_z))
        
        return total_loss


class PyTorchLossFunctions():
    """
    Source: https://neptune.ai/blog/pytorch-loss-functions
    """
    
    def l1_loss(self):
        """
        Applicarion: Regression problems, especially when the distribution of the target variable has outliers, 
                     such as small or big values that are a great distance from the mean value. It is considered 
                     to be more robust to outliers.
        """
        return torch.nn.L1Loss()

    def mse_loss(self):
        """
        Application: MSE is the default loss function for most Pytorch regression problems.
        """
        return torch.nn.MSELoss()
    
    def log_likelihood(self): 
        """
        Application: Multi-class classification problems
        """
        return torch.nn.NLLLoss()
    
    def cross_entropy(self):
        """
        Application: Binary classification tasks, for which it's the default loss function in Pytorch.
                     Creating confident models—the prediction will be accurate and with a higher probability.
        """
        return torch.nn.CrossEntropyLoss()
    

    
    
    