from typing import List
import copy

import numpy as np
from PIL import Image

import torch
from torch import nn
import torchvision.transforms as T

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
from transformers import MllamaProcessor, PreTrainedTokenizerFast

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
						return_tensors="pt"
					)
					outputs = self.clip_segm_model(**inputs)
					mask = outputs.logits.amax(0).sigmoid().float().cpu().numpy()
					mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(self.size)
					masks.append(mask.resize(image.size))

					x = self.preprocess(mask.convert("RGB"))#.to(dev)
					z_logits = self.vq_enc(x)
					z = torch.argmax(z_logits.detach(), axis=1)

					# " Given above image, the mask tokens for the image are: " + \
					# "<|image|>" +\
					_z = z[0].cpu().numpy()
					_text += "<begin_mask>" + "".join([
						f"<|vq_{_z[i, j]}>"
						for i in range(_z.shape[0])
						for j in range(_z.shape[1])
					]) + "<|end_mask|>"
				text += "<|begin_of_text|>" + _text + "<|end_of_text|>"

		# if mask is not None:
		#     mask_s = self.vq_enc(self.preprocess(mask))
		
		return super().__call__(
			images=images[0],
			text=text,
			**kwargs
		)
	
	@classmethod
	def from_llama_vision(cls, ckpt_path: str, model_id: str):        
		# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		vq_enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", "cpu")
		new_tokens = [
			f"<vq_{i}>" for i in range(vq_enc.blocks[-1].conv.w.shape[0])
		] + ["<begin_mask>", "<end_mask>", "<|image|>"]

		processor = MllamaProcessor.from_pretrained(model_id)
		# print(len(processor.tokenizer))

		tokenizer = PreTrainedTokenizerFast(
			tokenizer_file=f"{ckpt_path}/LLaMa-3.2-Vision.json"
		)
		tokenizer.bos_token = "<|begin_of_text|>"
		tokenizer.eos_token = "<|end_of_text|>"
		tokenizer.pad_token = "<|finetune_right_pad_id|>"
		# print(len(processor.tokenizer), len(new_tokens))
		tokenizer.add_tokens(new_tokens)

		processor.tokenizer = tokenizer

		# print(len(processor.tokenizer))
		return cls(
			vq_enc,
			image_processor=processor.image_processor,
			tokenizer=processor.tokenizer
		)



class ARCCollator:

	def __init__(self,
		processor,
		tokenizer,
		vq_enc,
		image_token="<|image|>",
		ignore_index=-100,
		device="cpu",
	):
		self.processor = processor
		self.tokenizer = tokenizer
		self.vq_enc = vq_enc
		self.vq_enc.eval()
		for param in self.vq_enc.parameters(): param.requires_grad = False
		self.ignore_index = ignore_index
		self.device = device

		self.image_token = image_token
		self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
		self.eos_token_id = tokenizer.eos_token_id
		self.pad_token_id = tokenizer.pad_token_id
		
		clip_segm_name = "CIDAS/clipseg-rd64-refined"
		self.clip_segm_processor = CLIPSegProcessor.from_pretrained(clip_segm_name, use_fast=True)
		self.clip_segm_model = CLIPSegForImageSegmentation.from_pretrained(clip_segm_name)#.to(device)
		self.clip_segm_model.eval()
		for param in self.clip_segm_model.parameters(): param.requires_grad = False
		# self.size = (352, 352)
		self.size = (176, 176)

	@staticmethod
	def preprocess(img: Image.Image) -> torch.Tensor:
		img = torch.unsqueeze(T.ToTensor()(img), 0)
		return map_pixels(img)  # (1 - 2 * 0.1) * x + 0.1
	
	@torch.no_grad()
	def convert_mask_to_text(self, masks: List[Image.Image]) -> List:
		# print(masks)
		xs = torch.concat([self.preprocess(mask.convert("RGB")) for mask in masks])  # (B, 3, H, W)

		z_logits = self.vq_enc(xs)  # (B, C, H', W')
		zs = torch.argmax(z_logits.detach(), axis=1)  # (B, H', W')
		texts = [
			# self.image_token + 
			# self.tokenizer.bom_token
			"<|begin_of_mask|>" + "".join([
				f"<|vq_{_z[i, j]}|>"
				for i in range(_z.shape[0])
				for j in range(_z.shape[1])
			]) + "<|end_of_mask|>"  # self.tokenizer.eom_token
			for _z in zs
		]
		return texts
	
	def find_index_of_bom(self, tokens):
		temp = self.tokenizer("<|begin_of_mask|>", add_special_tokens=False)
		bom_ids = temp['input_ids']
		
		n, m = len(tokens)-1, len(bom_ids)-1

		ids = []
		for i in range(n - m + 1):
			if tokens[i:i+m].tolist() == bom_ids[:-1]:
				ids.append(i)
		return ids

	@torch.no_grad()
	def _get_mask(self, images, prompts) -> List[Image.Image]:
		masks = []
		for image in images:
			inputs = self.clip_segm_processor(
				text=prompts, images=[image]*len(prompts),
				return_tensors="pt"
			)
			outputs = self.clip_segm_model(**inputs)  # (C, H, W)
			# combined_mask from all classes
			mask = outputs.logits.amax(0).sigmoid().float().cpu().numpy()
			mask = Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(self.size)
			masks.append(mask)
		return masks

	def __call__(self, batch, prompts, application):
		# print(batch)
		texts, images = [], []
		for sample in batch:
			_imgs = sample["rgb"]
			images.append([_img.copy() for _img in _imgs])
			_masks = self._get_mask(_imgs, prompts)
			_mask_texts = self.convert_mask_to_text(_masks)

			messages = [{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": (
							"You are a critic that selects sensing regions likely to contain "
							"salient information not already visible in the current image. "
							f"You should request sensing data regions for the application {application}. "
							"The generated mask tokens must be enclosed within "
							"<|begin_of_mask|> and <|end_of_mask|>."
						)
					}
				]
			}]
			for mask in _mask_texts:
				messages.extend([
					{
						"role": "user",
						"content": [
							{"type": "image"},
							{
								"type": "text",
								"text": f"Generate mask tokens for this image."
							}
						]
					},
					{
						"role": "assistant",
						"content": [
							{"type": "text", "text": mask}
						]
					},
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": f"Evaluation score: 98%."
							}
						]
					}
				])

			text = self.processor.apply_chat_template(
				messages,
				add_generation_prompt=False
			)
			texts.append(text)

		batch = self.processor(
			text=texts,
			images=images,
			return_tensors="pt",
			padding=True,
			truncation=False,
			add_special_tokens=False
		)

		labels = batch["input_ids"].clone()
		labels[labels == self.pad_token_id] = self.ignore_index
		labels[labels == self.image_token_id] = self.ignore_index

		bom_id = self.tokenizer.convert_tokens_to_ids("<|begin_of_mask|>")
		for i in range(labels.shape[0]):
			idx = (labels[i] == bom_id).nonzero(as_tuple=True)[0]
			if len(idx) > 0:
				labels[i][:idx[0]] = self.ignore_index

		batch["labels"] = labels
		return batch
	
	
class ExtendedLMHead(nn.Module):

	def __init__(self, base_head, extra_tokens, hidden_size=4096):
		super().__init__()
		self.base_head = base_head
		self.extra_head = nn.Linear(
			hidden_size,
			extra_tokens,  # accounting for the <|image|> token
			bias=False
		)

	def forward(self, x):
		old_logits = self.base_head(x)
		new_logits = self.extra_head(x)
		return torch.cat([old_logits, new_logits], dim=-1)
	
	@classmethod
	def from_llama(cls, ckpt_path, model):
		vq_enc: Encoder = load_model(f"{ckpt_path}/encoder.pkl", "cpu")
		return cls(
			base_head=copy.deepcopy(model.lm_head),
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
		base_ids = flat_ids.clamp(max=self.base_vocab_size-1)
		extra_ids = flat_ids - self.base_vocab_size

		# lookup both
		base_out = self.base_embedding(base_ids)#.detach()
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
			base_embedding=copy.deepcopy(model.language_model.embed_tokens),
			extra_tokens=vq_enc.vocab_size+2,
		)
