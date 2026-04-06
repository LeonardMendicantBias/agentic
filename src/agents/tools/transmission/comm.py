from typing import List

import PIL

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder

from smolagents import Tool


class EncodingTool(Tool):
    name = "encoding_tool"
    description = """
    Extract visual features from an input image and quantize the features into discrete codewords.
    The output is a 2D grid of integer indices representing the quantized visual tokens.
    """
    inputs = {
        "frame": {
            "type": "image",
            "description": "Input RGB or binary frame to be encoded.",
        }
    }
    output_type = "array"

    def __init__(self,
        ckpt,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.ckpt = ckpt
        self.device = device

    def setup(self):
        self.model: Encoder = load_model(self.ckpt, self.device)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.unsqueeze(0)),  # add a dimension at dim=0
            map_pixels
        ])
        super().setup()

    def forward(self, frame: PIL.Image.Image) -> List[List[int]]:
        if frame.mode == "L":
            frame = frame.convert("RGB")
        # black_mask = PIL.Image.new("RGB", frame.size, (0, 0, 0))
        
        inp_frame = self.transforms(frame)  # (B, 3, H, W)
        # black_mask = self.transforms(black_mask)  # (B, 3, H, W)
        # z_logits = self.model(torch.cat([inp_frame, black_mask], dim=0).to(self.device))
        
        z_logits = self.model(inp_frame.to(self.device))
        z = torch.argmax(z_logits, axis=1)
        # comm_mask = torch.ne(z[0], z[1])

        return z


class DecodingTool(Tool):

    name = "decoding_tool"
    description = """
        This is a tool that decode codewords back to RGB frame.
        It returns an image.
    """
    inputs = {
        "codewords": {
            "type": "array",
            "description": "The encoded codewords of the input frame",
        },
        # "comm_mask": {
        #     "type": "array",
        #     "description": "The binary mask image that indicates the location that visual features were not transmitted",
        # },
        "is_grayscale": {
            "type": "boolean",
            "description": "The encoded codewords of the input frame",
            "nullable": True
        }
    }
    output_type = "image"
    
    def __init__(self,
        ckpt,
        # dec_ckpt,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        # self.enc_ckpt = enc_ckpt
        self.ckpt = ckpt
        self.device = device

    def setup(self):
        # self.enc: Decoder = load_model(self.enc_ckpt, self.device)
        self.model: Decoder = load_model(self.ckpt, self.device)

        self.transforms = T.Compose([
            unmap_pixels
        ])
        super().setup()

    def forward(self,
        codewords: List[List[int]],
        # comm_mask: List[List[int]],
        is_grayscale: bool=False
    ) -> PIL.Image.Image:
        # black_mask = PIL.Image.new("RGB", frame.size, (0, 0, 0))

        z = F.one_hot(codewords, num_classes=self.model.vocab_size).permute(0, 3, 1, 2).float().to(self.device)

        x_stats = self.model(z).float()
        
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

        x_rec = T.ToPILImage(mode='L' if is_grayscale else "RGB")(x_rec[0])

        return x_rec
