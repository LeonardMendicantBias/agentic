import numpy as np
from PIL import Image

import torch
from torch import nn
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


class ExtendedLMHead(nn.Module):

	def __init__(self, base_head, extra_tokens, hidden_size=4096):
		super().__init__()
		self.base_head = base_head
		self.extra_head = nn.Linear(
			hidden_size,
			extra_tokens,
			bias=False
		)

		# freeze original head
		self.base_head.weight.requires_grad = False

	def forward(self, x):
		old_logits = self.base_head(x)
		new_logits = self.extra_head(x)
		return torch.cat([old_logits, new_logits], dim=-1)
	
	@classmethod
	def from_llama(cls, ckpt_path, model):
		vq_enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", "cpu")
		return cls(
			base_head=model.lm_head,
			extra_tokens=vq_enc.vocab_size+2,
			hidden_size=model.lm_head.in_features,
		)

class ExtendEmbedding(nn.Module):

	def __init__(self, base_embedding: nn.Embedding, extra_tokens: int):
		super().__init__()

		self.base_embedding = base_embedding

		self.base_vocab_size = base_embedding.num_embeddings
		self.hidden_size = base_embedding.embedding_dim

		self.extra_embedding = nn.Embedding(
			extra_tokens,
			self.hidden_size,
			padding_idx=None,
		)

	def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
		orig_shape = input_ids.shape
		flat_ids = input_ids.view(-1)

		is_extra = flat_ids >= self.base_vocab_size

		# map ids to valid ranges
		base_ids = flat_ids.clamp(max=self.base_vocab_size - 1)
		extra_ids = flat_ids - self.base_vocab_size

		# lookup both
		base_out = self.base_embedding(base_ids)
		extra_out = self.extra_embedding(extra_ids.clamp(min=0))

		# select output
		out = torch.where(
			is_extra.unsqueeze(-1),
			extra_out,
			base_out
		)

		return out.view(*orig_shape, self.hidden_size)
	
	@classmethod
	def from_llama(cls, ckpt_path, model):
		vq_enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", "cpu")
		return cls(
			base_embedding=model.language_model.embed_tokens,
			extra_tokens=vq_enc.vocab_size+2,
		)