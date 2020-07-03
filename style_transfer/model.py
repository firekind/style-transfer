from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
from tsalib import get_dim_vars

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
    def __init__(self, content_layers: List[str], style_layers: List[str], feature_extractor: nn.Module, mean: (C,),
                 std: (C,)):
        """
        Takes an image and reproduce it with a new artistic style.

        Args:
            content_layers (List[str]): The layers used to calculate content loss.
            style_layers (List[str]): The layers used to calculate style loss.
            feature_extractor (nn.Module): The model used to extract the features.
            mean ((C,)): The mean of the images used to train the extractor. A 1 dimensional tensor.
            std ((C,)): The std of the images used to train the extractor. A 1 dimensional tensor.
        """

        super(StyleTransferer, self).__init__()

        self.norm = Normalization(mean, std)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.feature_extractor = nn.Sequential()
        self.result_tuple = namedtuple(
            "TransferResult", [
                *[f"content_{name}" for name in content_layers],
                *[f"style_{name}" for name in style_layers],
            ])

        # reconstructing feature extractor to remove inplace ReLU operations
        # (inplace ReLU operation messes with output)
        conv_layer_idx = 1
        conv_count = 0

        for name, layer in feature_extractor.named_children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            if isinstance(layer, nn.MaxPool2d):
                conv_layer_idx += 1
                conv_count = 0
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
                name = f"conv_{conv_layer_idx}_{conv_count}"

            self.feature_extractor.add_module(name, layer)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: (B, C, H, W)) -> Tuple:
        result = {}
        content_layer_name = None
        style_layer_name = None

        # forward proping
        x = self.norm(x)
        for name, layer in self.feature_extractor.named_children():
            x: (B, C, H, W) = layer(x)

            if content_layer_name is not None:
                result[f"content_{content_layer_name}"] = x
                content_layer_name = None

            if style_layer_name is not None:
                result[f"style_{style_layer_name}"] = x
                style_layer_name = None

            if name in self.content_layers:
                content_layer_name = name

            if name in self.style_layers:
                style_layer_name = name

        # noinspection PyArgumentList
        return self.result_tuple(**result)
