import torch
import torch.nn as nn
import torch.nn.functional as F
from tsalib import get_dim_vars

from .utils import gram_matrix

B, C, H, W = get_dim_vars('B C H W')


class ContentLoss(nn.Module):
    def __init__(self, target: (B, C, H, W)):
        """
        Module that calculates the content loss.

        Args:
            target (torch.Tensor): The tensor wrt to which the loss has to be
            calculated, usually the feature map of the content image.
        """

        super(ContentLoss, self).__init__()

        # detaching the target so that gradients are not backproped along 
        # it.
        self.target = target.detach()

        # defining variable to hold the calculated loss
        self.loss: torch.Tensor = torch.Tensor()

    def forward(self, x: (B, C, H, W)) -> (B, C, H, W):
        # the if condition is required because while creating the `StyleTransferer`,
        # a forward pass with the style image is performed in order to compute the targets for
        # the style loss. The shape of the style image and the content image can 
        # differ, and if it does, the MSE operation below will crash. Not computing (or computing)
        # the content loss does not affect the construction of the targets for the
        # style loss. 
        # Such a situation only occurs while constructing the targets for the style
        # loss while initializing the model, it does not occur during training as
        # the input image during training has the same shape of the content image.
        if self.target.shape == x.shape:
            # calculating MSE loss between input and target
            self.loss = F.mse_loss(x, self.target)

        # returning input
        return x


class StyleLoss(nn.Module):
    def __init__(self, target: (B, C, H, W)):
        """
        Module that calculates the style loss.

        Args:
            target (torch.Tensor): The tensor wrt to which the loss has to be
            calculated, usually it is the feature map of the style image.
        """

        super(StyleLoss, self).__init__()

        # calculating gram matrix of the target and detaching it
        # detaching the target so that gradients are not backproped along 
        # it.
        self.target: (C, C) = gram_matrix(target).detach()

        # defining variable for loss
        self.loss: torch.Tensor = torch.Tensor()

    def forward(self, x: (B, C, H, W)) -> (B, C, H, W):
        # calculating MSE loss between gram matrix of input and
        # gram matrix of target
        self.loss = F.mse_loss(gram_matrix(x), self.target)

        # returning input
        return x
