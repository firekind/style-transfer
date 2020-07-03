from typing import List

import torch
import torch.nn as nn
from tsalib import get_dim_vars

B, C, H, W = get_dim_vars('B C H W')


def gram_matrix(x: (B, C, H, W)) -> (C, C):
    """
    Computes the gram matrix of the given tensor.
    """

    b, c, h, w = x.shape

    # reshaping F_XL to \hat F_XL, which is a KxN matrix where K is the number
    # of feature maps at layer L, and N is the length of the vectorized feature
    # map
    features: (B, C, H * W) = x.view(b, c, h * w)

    # computing gram matrix
    G: (C, C) = torch.bmm(features, features.transpose(1, 2))

    # normalizing the values of the gram matrix by dividing each value in the 
    # matrix by the total number of elements in the feature map
    G.div_(h * w)
    
    return G


class ModelTargets:
    def __init__(self, content_layers: List[str], style_layers: List[str], content_image: (B, C, H, W),
                 style_image: (B, C, H, W), model: nn.Module):
        """
        Computes and stores the required targets.

        Args:
            content_layers (List[str]): The layers that will be used to calculate the content loss
            style_layers (List[str]): The layers that will be used to calculate the style loss
            content_image ((B, C, H, W)): The image that will be used as the content. A tensor.
            style_image ((B, C, H, W)): The image that will used as the style image. A tensor.
            model (nn.Module): The model to be used to compute the targets.
        """

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_image = content_image
        self.style_image = style_image
        self.model = model

    def compute(self):
        """
        Computes the targets.
        """

        # setting the model to eval mode
        self.model.eval()

        # forward proping
        with torch.no_grad():
            content_res = self.model(self.content_image)
            style_res = self.model(self.style_image)

        # extracting content targets and assigning them to self.
        for name in self.content_layers:
            full_name = f"content_{name}"
            setattr(self, full_name, getattr(content_res, full_name).detach())

        # extracting style targets and assigning them to self
        for name in self.style_layers:
            full_name = f"style_{name}"
            setattr(self, full_name, getattr(style_res, full_name).detach())

        # setting model to train mode
        self.model.train()
