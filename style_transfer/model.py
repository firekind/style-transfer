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
        self.max_layer_count = max(len(content_layers), len(style_layers))
        self.result_tuple = namedtuple(
            "TransferResult", [
                *[f"content_{name}" for name in content_layers],
                *[f"style_{name}" for name in style_layers],
            ])

        # reconstructing feature extractor to remove inplace ReLU operations
        # (inplace ReLU operation messes with output)
        for name, layer in feature_extractor.named_children():
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            self.feature_extractor.add_module(name, layer)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x: (B, C, H, W)) -> Tuple:
        # constructing iterator
        iterator = iter(self.feature_extractor)

        # initializing variables
        count = 0
        result = {}

        # forward proping
        x = self.norm(x)
        while count < self.max_layer_count:
            layer: nn.Module = next(iterator)
            x: (B, C, H, W) = layer(x)

            if isinstance(layer, nn.Conv2d):
                count += 1
                layer_name = f"conv_{count}"

                if layer_name in self.content_layers:
                    result[f"content_{layer_name}"] = x

                if layer_name in self.style_layers:
                    result[f"style_{layer_name}"] = x

        # noinspection PyArgumentList
        return self.result_tuple(**result)
