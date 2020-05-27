import torch

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
    features: (B * C, H * W) = x.view(b * c, h * w)

    # computing gram matrix
    G: (C, C) = torch.mm(features, features.t())

    # normalizing the values of the gram matrix by dividing each value in the 
    # matrix by the total number of elements in the feature map
    return G.div(b * c * h * w)
