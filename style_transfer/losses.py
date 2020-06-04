import torch
import torch.nn.functional as F
from tsalib import get_dim_vars

from .utils import gram_matrix

B, C, H, W = get_dim_vars('B C H W')


def style_loss(x: (B, C, H, W), target: (B, C, H, W)) -> torch.Tensor:
    """
    Calculates the style loss between `x` and `target`.

    Args:
        x ((B, C, H, W)): The predicted value from the model.
        target ((B, C, H, W)): The label to predict against.
        
    Returns:
        torch.Tensor: A scalar loss.
    """

    # calculating MSE loss between gram matrix of input and
    # gram matrix of target
    return F.mse_loss(gram_matrix(x), gram_matrix(target).detach())


def content_loss(x: (B, C, H, W), target: (B, C, H, W)) -> torch.Tensor:
    """
    Calculates the content loss between `x` and `target`.

    Args:
        x ((B, C, H, W)): The predicted value from the model.
        target ((B, C, H, W)): The label to predict against.
        
    Returns:
        torch.Tensor: A scalar loss.
    """
    # noinspection PyTypeChecker
    return F.mse_loss(x, target.detach())
