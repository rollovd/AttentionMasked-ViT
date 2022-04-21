import math
import numpy as np
from PIL import Image
from typing import Callable

import torch
import torch.nn as nn
import torchvision.transforms as transforms


def square_crop(image):
    image = np.asarray(image)
    shape = image.shape
    if shape[-1] == 4:
        image = image[..., :-1]

    square = min(shape[:-1])

    x1_borders = (shape[0] - square) // 2
    x2_borders = (shape[1] - square) // 2

    if not x1_borders:
        image = image[:, x2_borders:-x2_borders]
    else:
        image = image[x1_borders:-x1_borders, :]

    return Image.fromarray(image)


def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight

class ImagePadding:

    def __init__(self,
                 patch_size=16,
                 padding_type=None,
                 square_image_size=256):

        self.patch_size = patch_size
        self.padding_type = padding_type
        self.square_image_size = square_image_size
        self.to_tensor = transforms.ToTensor()
        self._preferable_values = torch.tensor([192, 144, 384, 288])
        self.side_patches = square_image_size // patch_size

        assert self.padding_type in ['corner', 'both_sides', 'one_side']

    def _create_mask(self, diff, vertical=True):
        with torch.no_grad():
            extra_pixels = diff // self.patch_size
            relevant_pixels = self.side_patches - extra_pixels

            base = torch.ones((self.side_patches, relevant_pixels), dtype=torch.bool)
            extra = torch.zeros((self.side_patches, extra_pixels), dtype=torch.bool)
            mask = torch.cat((base, extra), dim=1)

            return ~mask if vertical else ~mask.T

    def _aspect_ratio_resize(self, image_path):
        image = Image.open(image_path)

        H, W, C = np.asarray(image).shape
        max_value = max(H, W)
        proportion = round(H / W, 3)
        partition = max_value / self.square_image_size

        if max_value == H and max_value != W:
            H_new, W_new = self.square_image_size, math.ceil(W / partition)
            mask = torch.isclose(torch.tensor([W_new]), self._preferable_values, rtol=1e-16, atol=2)
            wish_value = self._preferable_values[mask]
            W_new = wish_value

            diff = abs(H_new - W_new).item()
            mask = self._create_mask(diff, vertical=True)

        elif max_value == W and max_value != H:
            H_new, W_new = math.ceil(H / partition), self.square_image_size
            mask = torch.isclose(torch.tensor([H_new]), self._preferable_values, rtol=1e-16, atol=2)
            wish_value = self._preferable_values[mask]
            H_new = wish_value

            diff = abs(W_new - H_new).item()
            mask = self._create_mask(diff, vertical=False)

        else:
            H_new, W_new = self.square_image_size, self.square_image_size

        image = image.resize((W_new, H_new), Image.NEAREST)

        return image, mask.flatten(), C

    def _pad_image(self, image):
        with torch.no_grad():
            C, H, W = image.shape
            max_value = max(H, W)

            if self.padding_type == 'one_side':
                H_pad, W_pad = abs(H - max_value), abs(W - max_value)
                image = nn.functional.pad(image, (0, W_pad, 0, H_pad, 0, 0))

            elif self.padding_type == 'both_sides':
                H_pad, W_pad = abs(H - max_value) // 2, abs(W - max_value) // 2
                image = nn.functional.pad(image, (W_pad, W_pad, H_pad, H_pad, 0, 0))

            return image

    def __call__(self, image_path):
        resize_image, mask, proportion = self._aspect_ratio_resize(image_path)
        image = self.to_tensor(resize_image)
        image = self._pad_image(image)

        return image, mask, proportion