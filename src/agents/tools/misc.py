from typing import List, Tuple
import PIL

import numpy as np

import torch
import torchvision.transforms as transforms

from smolagents import tool


@tool
def masking_frame(frame: PIL.Image, mask: PIL.Image) -> PIL.Image:
    """
        Mask an RGB frame according to a mask
        Args:
            frame: the RGB image
            mask: the binary mask
        Returns:
            masked_frame
    """
    background: PIL.Image = PIL.Image.new("RGB", frame.size, (0, 0, 0))

    return PIL.Image.composite(frame, background, mask)