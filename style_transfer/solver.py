from typing import List

import torch
import torch.optim as optim
from PIL import Image
from pkbar import Kbar
from torchvision.models import vgg19
from tsalib import get_dim_vars

from .data import get_processing_transforms
from .model import StyleTransferer

B, C, H, W = get_dim_vars("B C H W")


def transfer_style(
        content_image_path: str,
        style_image_path: str,
        cuda: bool = True,
        image_size: int = 512,
        content_as_input: bool = True,
        epochs: int = 16,
        content_weight: float = 1,
        style_weight: float = 1000000,
        extractor_mean: List[float] = (0.485, 0.456, 0.406),
        extractor_std: List[float] = (0.229, 0.224, 0.225)
) -> Image.Image:
    """
    Transfers the style from the style image to the content image.`

    Args:
        content_image_path (str): The path to the content image.
        style_image_path (str): The path to the style image
        cuda (bool, optional): If True, will use cuda. Defaults to True.
        image_size (int, optional): The size to resize to while training. Defaults to 512.
        content_as_input (bool, optional): Use the content image as input while training. Defaults to True.
        epochs (int, optional): Number of epochs to train for. Defaults to 16.
        content_weight (float, optional): The weight for the content loss. Defaults to 1.
        style_weight (float, optional): The weight for the style loss. Defaults to 1000000.
        extractor_mean (List[float], optional): The mean used to normalize the channels during the training of the 
        model which will be used to extract features (VGG19 by default). Defaults to [0.485, 0.456, 0.406].
        extractor_std (List[float], optional): The std used to normalize the channels during the training of the 
        model which will be used to extract features (VGG19 by default). Defaults to [0.229, 0.224, 0.225].

    Returns:
        Image.Image: The resultant image.
    """

    # getting device to be used
    device: str = "cuda:0" if torch.cuda.is_available() and cuda else "cpu"

    # getting the pre-processing and post processing transforms
    preprocess, postprocess = get_processing_transforms(image_size)

    # getting the content image
    content_image: (C, H, W) = preprocess(Image.open(content_image_path)).to(device)
    content_image: (B, C, H, W) = content_image.unsqueeze(0)

    # getting the style image
    style_image: (C, H, W) = preprocess(Image.open(style_image_path)).to(device)
    style_image: (B, C, H, W) = style_image.unsqueeze(0)

    # getting / constructing the image to be used as input to the model
    input_image: (B, C, H, W) = content_image.clone() if content_as_input else torch.randn(
        content_image.shape, device=device
    )

    # constructing model
    model = StyleTransferer(
        content_layers=["conv_4"],
        style_layers=["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"],
        content_image=content_image,
        style_image=style_image,
        feature_extractor=vgg19(pretrained=True).features.to(device).eval(),
        mean=torch.tensor(extractor_mean).to(device),
        std=torch.tensor(extractor_std).to(device),
    ).to(device)

    # creating optimizer that optimizes the input image
    optimizer = optim.LBFGS([input_image.requires_grad_()])

    # defining variable
    current_epoch: int = 0

    # creating progress bar
    prog_bar = Kbar(target=epochs, unit_name="epoch")

    # training
    while current_epoch < epochs:
        # defining variables
        content_loss_sum = 0.0
        style_loss_sum = 0.0
        count = 0

        def closure():
            # referring to non local variables defined above (crashes without it)
            nonlocal content_loss_sum, style_loss_sum, count

            # clamping image without needing to calculate gradient
            with torch.no_grad():
                input_image.clamp_(0, 1)

            # zero the optimizer
            optimizer.zero_grad()

            # forward pass
            model(input_image)

            # calculating the total content loss
            content_score = model.content_score()
            content_loss_sum += content_score

            # calculating the total style loss
            style_score = model.style_score()
            style_loss_sum += style_score

            # calculating the total loss
            loss = content_weight * content_score + style_weight * style_score

            # computing gradients
            loss.backward()

            # incrementing count
            count += 1

            return loss

        # stepping optimizer
        optimizer.step(closure)

        current_epoch += 1
        prog_bar.update(current_epoch,
                        values=[("content loss", content_loss_sum / count), ("style loss", style_loss_sum / count)])

    return postprocess(input_image.squeeze().cpu())
