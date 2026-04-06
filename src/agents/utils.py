from typing import List, Any
import PIL

import numpy as np

from smolagents import tool, Tool

from matplotlib import pyplot as plt


@tool
def export_masks(masks: List[Any]) -> List[Any]:
    """A tool that returns a mask for each modality
    Args:
        masks: A list of masks, each of which is an PILLOW Image
    
    """
    # for mask in mask:
    #     plt.imshow(mask)
    #     plt.axis("off")
    #     plt.show()
    return masks


@tool
def extract_by_mask(image: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    """A tool that extract depth image according to a mask
    Args:
        image: depth image
        mask: binary image
    """
    if image.size != mask.size: return mask
    
    return mask


@tool
def mask_from_reid(image: PIL.Image.Image, bbox: list) -> PIL.Image.Image:
    """
    Mask the entire input image except the region corresponding
    to the re-identified object.

    Args:
        image: Input image.
        bbox: Bounding box [x1, y1, x2, y2] of the re-identified object.

    Returns:
        Masked image where only the identified object region remains visible.
    """

    x1, y1, x2, y2 = map(int, bbox)

    image_np = np.array(image)
    h, w = image_np.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1

    masked = image_np.copy()
    masked[mask == 0] = 0

    return PIL.Image.fromarray(masked)

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer

    def __init__(self, *args, **kwargs):
        self.is_initialized = False


class MyFinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer
    
    def __init__(self, *args, **kwargs):
        self.is_initialized = False

