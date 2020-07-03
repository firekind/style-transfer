from typing import List, Union

import torch
import torch.optim as optim
from PIL import Image
from pkbar import Kbar
from torchvision.models import vgg19
from tsalib import get_dim_vars

from .data import get_processing_transforms
from .losses import content_loss, style_loss
from .model import StyleTransferer
from .utils import ModelTargets

B, C, H, W = get_dim_vars("B C H W")


def transfer_style(
        content_image_path: str,
        style_image_path: str,
        cuda: bool = True,
        image_size: int = 512,
        epochs: int = 16,
        content_weight: Union[float, List[float]] = 1,
        style_weight: Union[float, List[float]] = 1000000,
        extractor_mean: List[float] = (0.40760392, 0.45795686, 0.48501961),
        extractor_std: List[float] = (1, 1, 1),
        content_layers: List[str] = ("conv_4_2",),
        style_layers: List[str] = ("conv_1_1", "conv_2_1", "conv_3_1", "conv_4_1", "conv_5_1"),
        save_path: str = None
) -> Image.Image:
    """
    Transfers the style from the style image to the content image.`

    Args:
        content_image_path (str): The path to the content image.
        style_image_path (str): The path to the style image
        cuda (bool, optional): If True, will use cuda. Defaults to True.
        image_size (int, optional): The size to resize to while training. Defaults to 512.
        epochs (int, optional): Number of epochs to train for. Defaults to 16.
        content_weight (Union[float, List[float], optional): The weight(s) for the content loss. If a list, number of weights given should
        match the number of layers given for `content_layers`. Defaults to 1.
        style_weight (Union[float, List[float], optional): The weight(s) for the style loss. If a list, number of weights given should
        match the number of layers given for `style_layers`.Defaults to 1000000.
        extractor_mean (List[float], optional): The mean used to normalize the channels during the training of the 
        model which will be used to extract features (VGG19). Defaults to [0.485, 0.456, 0.406].
        extractor_std (List[float], optional): The std used to normalize the channels during the training of the 
        model which will be used to extract features (VGG19). Defaults to [0.229, 0.224, 0.225].
        content_layers (List[str]): The layers used to calculate content loss.
        style_layers (List[str]): The layers used to calculate style loss.
        save_path (str): Path to save the output image (as JPEG) to.

    Raises:
        RuntimeError: If the number of weights given are invalid (if they are a list)

    Returns:
        Image.Image: The resultant image.
    """

    # getting device to be used
    device: str = "cuda:0" if torch.cuda.is_available() and cuda else "cpu"

    # parsing loss weights
    if not isinstance(content_weight, (list, tuple)):
        content_weight = [content_weight for _ in range(len(content_layers))]
    if not isinstance(style_weight, (list, tuple)):
        style_weight = [style_weight for _ in range(len(style_layers))]
    
    # checking if the losses are correctly given
    if len(content_weight) != len(content_layers) or len(style_weight) != len(style_layers):
        raise RuntimeError("invalid number of weights.")

    # getting the pre-processing and post processing transforms
    preprocess, postprocess = get_processing_transforms(image_size)

    # getting the content image
    processed_content: (C, H, W) = preprocess(Image.open(content_image_path)).to(device)
    processed_content: (B, C, H, W) = processed_content.unsqueeze(0)

    # getting the style image
    processed_style: (C, H, W) = preprocess(Image.open(style_image_path)).to(device)
    processed_style: (B, C, H, W) = processed_style.unsqueeze(0)

    # getting / constructing the image to be used as input to the model
    input_image: (B, C, H, W) = processed_content.clone()

    # constructing model
    model = StyleTransferer(
        content_layers,
        style_layers,
        vgg19(pretrained=True).features.to(device).eval(),
        mean=torch.tensor(extractor_mean).to(device),
        std=torch.tensor(extractor_std).to(device),
    ).to(device)

    # constructing targets for style and content loss
    targets = ModelTargets(
        content_layers,
        style_layers,
        processed_content,
        processed_style,
        model
    )
    targets.compute()

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
            content_score = torch.tensor(0.0, requires_grad=False).to(device)
            style_score = torch.tensor(0.0, requires_grad=False).to(device)

            # clamping image without needing to calculate gradient
            with torch.no_grad():
                input_image.clamp_(0, 1)

            # zero the optimizer
            optimizer.zero_grad()

            # forward pass
            res = model(input_image)

            # calculating the total content loss
            for i, name in enumerate(content_layers):
                full_name = f"content_{name}"
                content_score += content_weight[i] * content_loss(getattr(res, full_name), getattr(targets, full_name))
            content_loss_sum += content_score

            # calculating the total style loss
            for i, name in enumerate(style_layers):
                full_name = f"style_{name}"
                style_score += style_weight[i] * style_loss(getattr(res, full_name), getattr(targets, full_name))
            style_loss_sum += style_score

            # calculating the total loss
            loss = content_score + style_score

            # computing gradients
            loss.backward()

            # incrementing count
            count += 1

            return loss

        # stepping optimizer
        optimizer.step(closure)

        # updating required variables
        current_epoch += 1
        prog_bar.update(current_epoch,
                        values=[("content loss", content_loss_sum / count), ("style loss", style_loss_sum / count)])

    output_image = postprocess(input_image.squeeze().cpu())
    if save_path is not None:
        output_image.save(save_path, "JPEG")
        print("saved output to: %s" % save_path)

    return output_image
