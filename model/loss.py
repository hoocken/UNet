import torch
import torch.nn as nn

class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, x, ground):
        """
        Calculate generalized dice loss based of "Generalised Dice Overlap as a 
        Deep Learning Loss Function for Highly Unbalanced Segmentations"

        Parameters:
            x: Tensor of size (N, C, H, W)
            ground: Ground truth labels of (N, C, H, W) with C the number of classes
        """
        ground_sum = torch.sum(ground, (2, 3)) # (N, C)
        weights = 1 / torch.pow(ground_sum, 2)
        intersection =  torch.sum(x * ground, (2, 3))
        union = torch.sum(x, (2, 3)) + ground_sum
        loss = 1 - 2 * torch.sum(weights * intersection, 1) / torch.sum(weights * union , 1)

        avg_loss = torch.mean(loss)
        return avg_loss
    
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, x, ground):
        """
        Calculate binary cross entropy for each class.

        Parameters:
            x: Tensor of size (N, C, H, W)
            ground: Ground truth labels of (N, C, H, W) with C the number of classes
        """
        return self.loss(x, ground)
    

class UNetLoss(nn.Module):
    def __init__(self):
        super(UNetLoss, self).__init__()
        self.dsc = GeneralizedDiceLoss()
        self.bce = BinaryCrossEntropyLoss()

    def forward(self, x, ground):
        return self.dsc(x, ground) + self.bce(x, ground)

