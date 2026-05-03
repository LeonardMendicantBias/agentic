from typing import List
import copy
import json

import numpy as np
from PIL import Image
import cv2

import torch
from torch import nn
import torchvision.transforms as T
from torchmetrics import JaccardIndex

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegImageSegmentationOutput
from transformers import MllamaProcessor, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoImageProcessor

from dall_e import map_pixels, unmap_pixels, load_model
from dall_e import Encoder, Decoder

from matplotlib import pyplot as plt

from src.agents.tools.inpainting import InpaintingTool


class ArcProcessor(MllamaProcessor):
	
	@classmethod
	def from_data(cls, model_id: str, path: str):
		vq_enc: Encoder = load_model(f"{path}/encoder.pkl", "cpu")
		vq_enc.eval()

		new_tokens = [
			f"<|vq_{i}|>" for i in range(vq_enc.blocks[-1].conv.w.shape[0])
		] + ["<|begin_of_mask|>", "<|end_of_mask|>", "<|image|>"]

		tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
		tokenizer.bom_token = "<|begin_of_mask|>"
		tokenizer.eom_token = "<|end_of_mask|>"
		tokenizer.add_tokens(new_tokens)

		del vq_enc

		with open("./checkpoints/chat_template.json", "r", encoding="utf-8") as f:
			chat_template = json.load(f)
			
		return cls(
			AutoImageProcessor.from_pretrained(model_id),
			tokenizer,
			chat_template=chat_template["chat_template"],
		)


class ArcCollator:

	def __init__(self,
		processor,
		tokenizer,
		path,
		image_token="<|image|>",
		ignore_index=-100,
		device="cpu",
	):
		self.processor = processor
		self.tokenizer = tokenizer
		self.vq_enc: Encoder = load_model(f"{path}/encoder.pkl", "cpu")
		self.vq_enc.eval()
		for param in self.vq_enc.parameters(): param.requires_grad = False
		self.ignore_index = ignore_index
		self.device = device

		self.image_token = image_token
		self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
		
		clip_segm_name = "CIDAS/clipseg-rd64-refined"
		self.clip_segm_processor = CLIPSegProcessor.from_pretrained(clip_segm_name, use_fast=True)
		self.clip_segm_model = CLIPSegForImageSegmentation.from_pretrained(clip_segm_name)#.to(device)
		self.clip_segm_model.eval()
		for param in self.clip_segm_model.parameters(): param.requires_grad = False
		# self.size = (352, 352)
		self.size = (176, 176)
		self.patch_size = 8

		self.inpainting = InpaintingTool(ckpt=f"{path}/dstt.pth")

		self.jaccard = JaccardIndex(num_classes=2, task="binary")

	@staticmethod
	def preprocess(img: Image.Image) -> torch.Tensor:
		img = torch.unsqueeze(T.ToTensor()(img), 0)
		return map_pixels(img)  # (1 - 2 * 0.1) * x + 0.1
	
	@torch.no_grad()
	def convert_mask_to_text(self, mask: Image.Image):
		x = self.preprocess(mask.convert("RGB"))  # (B, 3, H, W)

		z_logits = self.vq_enc(x)  # (1, C, H', W')
		z = torch.argmax(z_logits.detach(), axis=1)  # (1, H', W')
		z = z[0]
		text = self.tokenizer.bom_token + "".join([
			f"<|vq_{z[i, j]}|>"
			for i in range(z.shape[0])
			for j in range(z.shape[1])
		]) + self.tokenizer.eom_token
		return text
	
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
	def _get_sem_mask(self, image, prompts, drop_p: float=0.6) -> Image.Image:
		inputs = self.clip_segm_processor(
			text=prompts, images=[image]*len(prompts),
			return_tensors="pt"
		)
		outputs = self.clip_segm_model(**inputs)  # (C, H, W)
		# combined_mask from all classes
		_m = (np.random.random(outputs.logits.shape[0]) >= drop_p).astype(int)
		mask = outputs.logits.sigmoid().float().cpu().numpy() * _m[:, None, None]
		mask = mask.max(0)
		return Image.fromarray(255 * (mask > 0.5).astype(np.uint8)).resize(image.size)
	
	def _get_random_mask(self, image, patch_size: int=8, threshold: float=0.2) -> Image.Image:
		h, w = image.size
		grid = (np.random.random((w // patch_size, h // patch_size)) < threshold).astype(np.uint8)
		return Image.fromarray(grid.repeat(patch_size, axis=0).repeat(patch_size, axis=1)*255, mode="L")

	def _get_mask(self, images, prompts) -> List[Image.Image]:
		gt_masks, sem_masks, masks = [], [], []
		for image in images:
			sem_mask = self._get_sem_mask(image, prompts)
			ran_mask = self._get_random_mask(image)

			mask = Image.composite(sem_mask, ran_mask, sem_mask)
			sem_masks.append(sem_mask)
		return gt_masks, sem_masks, masks
	
	def _get_patch_laplacian(self, img):
		gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
		laplacian = cv2.Laplacian(gray, cv2.CV_64F)
		
		h, w = gray.shape
		ph, pw = self.patch_size, self.patch_size
		
		# Trim image so it divides evenly
		h_trim = (h // ph) * ph
		w_trim = (w // pw) * pw
		
		# Reshape into patches
		lap_patches = laplacian.reshape(h_trim//ph, ph, w_trim//pw, pw)
		lap_patches = lap_patches.transpose(0, 2, 1, 3)  # (num_h, num_w, ph, pw)
		
		# Compute variance per patch
		patch_var = lap_patches.reshape(-1, ph*pw).var(axis=1)
		patch_var = patch_var.reshape(h_trim//ph, w_trim//pw)
		patch_var = np.log(patch_var + 1e-8)
		return (patch_var - patch_var.min()) / (patch_var.max() - patch_var.min() + 1e-8)

	def __call__(self, batch, prompts, application):
		# print(batch)
		texts, images = [], []
		for sample in batch:
			self.inpainting.reset()
			_imgs = sample["rgb"]
			images.append([_img.copy() for _img in _imgs[:-1]])
			
			gt_masks = [self._get_sem_mask(img, prompts, drop_p=0.) for img in _imgs]
			sem_masks = [self._get_sem_mask(img, prompts) for img in _imgs]

			rand_masks = [self._get_random_mask(img, patch_size=self.patch_size) for img in _imgs]

			masks = [
				Image.composite(sem_mask, rand_mask, sem_mask)
				for sem_mask, rand_mask in zip(sem_masks, rand_masks)
			]
			mask_imgs = [
				Image.composite(img, Image.new("RGB", img.size, 0), mask.resize(img.size))
				for img, mask in zip(_imgs, masks)
			]
			recon_imgs = [self.inpainting(mask_img, mask) for mask_img, mask in zip(mask_imgs, masks)]
			laps = [
				self._get_patch_laplacian(img)
				for img in recon_imgs
			]
			
			messages = [{
				"role": "system",
				"content": [
					{
						"type": "text",
						"text": (
							"You are a critic that requests regions in sensing data that is likely to contain salient information not already visible in the current image. "
							f"You should request sensing data regions spatially relevant for the application {application}. "
							f"Otherwise, you should request sensing data regions that temporally uncertaint. "
							f"Note that you should focus primarily on requesting relevant sensing data regions before uncertaint regions. "
							f"The inputs RGB image and binary mask of size ({_imgs[0].size[0]}, {_imgs[0].size[1]}) with patch of size ({self.patch_size}, {self.patch_size}). "
							f"The generated mask tokens must be enclosed within {self.tokenizer.bom_token} and {self.tokenizer.eom_token}."
						)
					}
				]
			}]

			for i, j in zip(range(len(_imgs)-1), range(1, len(_imgs))):
				mask = masks[i]
				_mask_text = self.convert_mask_to_text(mask)

				sem_score = self.jaccard(
					torch.tensor(np.array(masks[i]) // 255),
					torch.tensor(np.array(rand_masks[i]) // 255)
				)

				_mask = np.array(masks[j])
				h, w = _mask.shape
				rows, cols = h // 8, w // 8
				mask_patches = _mask[:rows * self.patch_size, :cols * self.patch_size]
				mask_patches = mask_patches.reshape(rows, self.patch_size, cols, self.patch_size)
				mask_patches = mask_patches.mean(axis=(1, 3))
				cur_score = (laps[j] - laps[i]) * (mask_patches / 255)
				cur_score = (cur_score / (mask_patches / 255).sum()).sum()

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
							{"type": "text", "text": _mask_text}
						]
					},
					{
						"role": "user",
						"content": [
							{
								"type": "text",
								"text": f"Semantic score as {sem_score:.4f} with curiousity score as {cur_score:.4f}"
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
		labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
		labels[labels == self.image_token_id] = self.ignore_index

		# bom_id = self.tokenizer.convert_tokens_to_ids("<|begin_of_mask|>")
		# for i in range(labels.shape[0]):
		# 	idx = (labels[i] == bom_id).nonzero(as_tuple=True)[0]
		# 	if len(idx) > 0:
		# 		labels[i][:idx[0]] = self.ignore_index

		batch["labels"] = labels
		return batch, texts
	
	
class ExtendedLMHead(nn.Module):

	def __init__(self, base_head, extra_tokens, hidden_size=4096):
		super().__init__()
		self.base_head = base_head
		self.extra_head = nn.Linear(
			hidden_size,
			extra_tokens,  # accounting for the <|image|> token
			bias=False
		).to(base_head.weight.device)

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
		).to(base_embedding.weight.device)

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
