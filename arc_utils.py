import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
from transformers import MllamaProcessor

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder

from matplotlib import pyplot as plt


class ArcProcessor(MllamaProcessor):

	def __init__(self, vq_enc, image_processor, tokenizer, **kwargs):
		self.vq_enc = vq_enc
		
		clip_segm_name = "CIDAS/clipseg-rd64-refined"
		self.clip_segm_processor = CLIPSegProcessor.from_pretrained(clip_segm_name, use_fast=True)
		self.clip_segm_model = CLIPSegForImageSegmentation.from_pretrained(clip_segm_name)#.to(device)
		self.size = (352, 352)

		super().__init__(
			image_processor=image_processor,
			tokenizer=tokenizer,
			**kwargs
		)
	
	@staticmethod
	def preprocess(img: Image.Image) -> torch.Tensor:
		img = torch.unsqueeze(T.ToTensor()(img), 0)
		return map_pixels(img)  # (1 - 2 * 0.1) * x + 0.1

	def __call__(self, prompts, images=None, text=None, **kwargs):
		if text is None:
			text = ""
		if images is not None:
			print(len(images))
			with torch.inference_mode():
				masks = []
				_text = ""
				for image in images:
					inputs = self.clip_segm_processor(
						text=prompts, images=[image]*len(prompts),
						padding=True, return_tensors="pt"
					)
					outputs = self.clip_segm_model(**inputs)
					mask = outputs.logits.amax(0).sigmoid().float().cpu().numpy()
					mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(self.size)
					masks.append(mask.resize(image.size))

					x = self.preprocess(mask.convert("RGB"))#.to(dev)
					z_logits = self.vq_enc(x)
					z = torch.argmax(z_logits, axis=1)

					# " Given above image, the mask tokens for the image are: " + \
					# "<|image|>" +\
					_z = z[0].cpu().numpy()
					_text += "<begin_mask>" + "".join([
						f"<vq_{_z[i, j]}>"
						for i in range(_z.shape[0])
						for j in range(_z.shape[1])
					]) + "<end_mask>"
				text += _text


		# if mask is not None:
		#     mask_s = self.vq_enc(self.preprocess(mask))
		
		return super().__call__(
			images=images,
			text=text,
			**kwargs
		)
	
	@classmethod
	def from_llama_vision(cls, ckpt_path: str, model_id: str):        
		# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		vq_enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", "cpu")
		new_tokens = [
			f"<vq_{i}>" for i in range(vq_enc.blocks[-1].conv.w.shape[0])
		] + ["<begin_mask>", "<end_mask>"]

		processor = MllamaProcessor.from_pretrained(model_id)
		processor.tokenizer.add_tokens(new_tokens)

		return cls(
			vq_enc,
			image_processor=processor.image_processor,
			tokenizer=processor.tokenizer
		)