from typing import List

import torch
import torch.nn as nn
from tsalib import get_dim_vars

from .losses import ContentLoss, StyleLoss

B, C, H, W = get_dim_vars('B C H W')


class Normalization(nn.Module):
    def __init__(self, mean: (C,), std: (C,)):
        """
        Module that normalizes the input using the given mean and std.

        Args:
            mean (torch.Tensor): The mean. shape: (C)
            std (torch.Tensor): The standard deviation. shape (C)
        """

        super(Normalization, self).__init__()

        self.mean: (C, 1, 1) = mean.view(-1, 1, 1)
        self.std: (C, 1, 1) = std.view(-1, 1, 1)

    def forward(self, x: (B, C, H, W)) -> (B, C, H, W):
        return (x - self.mean) / self.std


class StyleTransferer(nn.Module):
    def __init__(
            self,
            content_layers: List[str],
            style_layers: List[str],
            content_image: (B, C, H, W),
            style_image: (B, C, H, W),
            feature_extractor: nn.Sequential,
            mean: (C,),
            std: (C,)
    ):
        """
        Takes an image and reproduce it with a new artistic style.

        Args:
            content_layers (List[str]): The layers that are used to calculate the content loss.
            style_layers (List[str]): The layers that are used to calculate the style loss.
            content_image ((B, C, H, W)): The tensor depicting the content of the output.
            style_image ((B, C, H, W)): The tensor whose style needs to be transferred.
            feature_extractor (nn.Sequential): The module that is used to generate feature maps.
            mean ((C)): The mean used to normalize the channels during the training of the 
            feature extractor. A tensor.
            std ((C)): The std used to normalize the channels during the training of the 
            feature extractor. A tensor.

        Raises:
            RuntimeError: When an unrecognized layer is found in the `feature_extractor`.
        """

        super(StyleTransferer, self).__init__()

        # defining variables
        self.layers = nn.Sequential(Normalization(mean, std))
        self.content_losses: List[ContentLoss] = []
        self.style_losses: List[StyleLoss] = []

        count: int = 0
        last_layer_count = max(len(content_layers), len(style_layers))

        iterator = iter(feature_extractor)

        while count < last_layer_count:
            # getting the next layer from the feature extractor
            layer = next(iterator)

            # setting the name (and layer) depending on the type of 
            # layer
            if isinstance(layer, nn.Conv2d):
                count += 1
                name = f"conv_{count}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{count}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{count}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"batchnorm_{count}"
            else:
                raise RuntimeError(f"Unknown layer: {layer}")

            # adding the layer to the model
            self.layers.add_module(name, layer)

            # adding `ContentLoss` module
            if name in content_layers:
                # getting feature map of content image generated upto this layer
                # noinspection PyTypeChecker
                feature_map: (B, C, H, W) = self.layers(content_image)

                # constructing `ContentLoss` module and adding it `self.content_losses`
                module = ContentLoss(feature_map)
                self.content_losses.append(module)

                # adding the `ContentLoss` module
                self.layers.add_module(
                    f"content_loss_{count}",
                    module
                )

            # adding `StyleLoss` module
            if name in style_layers:
                # getting feature map of content image generated upto this layer
                # noinspection PyTypeChecker
                feature_map: (B, C, H, W) = self.layers(style_image)

                # constructing `StyleLoss` module and adding it `self.style_losses`
                module = StyleLoss(feature_map)
                self.style_losses.append(module)

                # constructing and adding `StyleLoss` module
                self.layers.add_module(
                    f"style_loss_{count}",
                    module
                )

    def forward(self, x: (B, C, H, W)) -> (B, C, H, W):
        return self.layers(x)

    def content_score(self) -> torch.Tensor:
        """
        Calculates the total content loss in the model.

        Returns:
            float: The content loss.
        """

        return sum([c.loss for c in self.content_losses])

    def style_score(self) -> torch.Tensor:
        """
        Calculates the total style loss in the model.

        Returns:
            float: The content loss.
        """

        return sum([s.loss for s in self.style_losses])
