from typing import Tuple

import torch
from torchvision import transforms


def get_processing_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Gets the transforms for pre-processing and postprocessing

    Args:
        image_size (int): The size to resize the image to.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple consisting of a pre-processing transform
        and post processing transform.
    """

    # pre-processing transform
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # post-processing transform
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: torch.clamp(x, min=0, max=1)),
        transforms.ToPILImage()
    ])

    return preprocess, postprocess
