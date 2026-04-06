import torch
import torch.nn as nn

from transformers import Trainer

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder


class ArcTrainer(Trainer):

    def __init__(self,
        ckpt_path: str,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    #     self.enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", device)
    #     self.dec: Decoder = load_model(f"{ckpt_path}/decoder.pkl", device)

    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")

        # outputs = model(**inputs)
        # logits = outputs.logits

        # loss_fn = nn.MSELoss()
        # loss = loss_fn(logits, labels)

        # return (loss, outputs) if return_outputs else loss

        print(inputs)

        return None

    # def 
