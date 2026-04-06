from typing import List, Tuple
import PIL

import numpy as np

import torch
import torch.nn.functional as F

from smolagents import Tool

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.models.sam2.modeling_sam2 import Sam2ImageSegmentationOutput as SegmentationOutput


class SemanticSegmentationTool(Tool):
	name = "semantic_segmentation_tool"
	description = """
	Segment an input RGB frame according to a text prompt describing the object class of interest.
	Returns a binary mask image where the target object pixels are 1 and the background is 0.
	"""
	inputs = {
		"frame": {
			"type": "image",
            "description": "Input RGB frame as a PIL image.",
		},
		"prompt": {
			"type": "string",
            "description": "Text description of the object class to segment.",
		},
	}
	output_type = "image"
	
	def __init__(self,
		model_name,
		device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	):
		super().__init__()
		self.model_name, self.device = model_name, device
		self.size = (352, 352)

	def setup(self):
		self.processor = CLIPSegProcessor.from_pretrained(self.model_name, use_fast=True)
		self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
		for p in self.model.parameters(): p.requires_grad_(False)
		self.model.eval()
		super().setup()
	
	def forward(self, frame: PIL.Image.Image, prompt: List[str]) -> PIL.Image.Image:
		inputs = self.processor(
			images=[frame]*len(prompt),
			text=prompt,
			padding=True,
			truncation=True,
			return_tensors="pt"
		).to(self.device)
		outputs = self.model(**inputs)
		
		mask = outputs.logits.amax(0).sigmoid().float().cpu().numpy()
		
		mask = PIL.Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(self.size)
		return mask.resize(frame.size)

	# # def forward(self, frame: List[PIL.Image], prompt: List[str]) -> PIL.Image:
	# def forward(self, frame: PIL.Image.Image, prompt: List[str]) -> PIL.Image.Image:
	# 	ratio = (round(frame.size[0]/self.size[0]+0.1), round(frame.size[1]/self.size[1]+0.1))
	# 	n_cells = ratio[0]*ratio[1]

	# 	resized_frame = frame.resize((ratio[0]*self.size[0], ratio[1]*self.size[1]))
	# 	patch_frame = [
	# 		resized_frame.crop((x, y, x + self.size[1], y + self.size[0]))
	# 		for y in range(0, resized_frame.height, self.size[0])
	# 		for x in range(0, resized_frame.width, self.size[1])
	# 	]

	# 	inputs = self.processor(text=n_cells*[prompt], images=patch_frame, return_tensors="pt").to(self.device)
	# 	outputs = self.model(**inputs)
		
	# 	masks = outputs.logits.sigmoid().float().cpu().numpy()
	# 	sem_mask = PIL.Image.new("1", (ratio[0]*self.size[0], ratio[1]*self.size[1]))
		
	# 	for i, mask in enumerate(masks):
	# 		x = (i % ratio[0]) * self.size[0]
	# 		y = (i // ratio[0]) * self.size[1]
	# 		mask = PIL.Image.fromarray((mask > 0.5).astype(np.uint8) * 255).resize(self.size)
	# 		sem_mask.paste(mask, (x, y, x + self.size[0], y + self.size[1]))

	# 	return sem_mask.resize(frame.size)

if __name__ == "__main__":
	from refiner.kitti_tracking import KittiDataset

	root_dir = "E:/KittiTracking"
	n_steps, n_pred_steps = 16, 3

	train_ds = KittiDataset(root_dir, "train", n_steps, n_pred_steps)

	semantic_segmentation_tool = SemanticSegmentationTool(model_name="CIDAS/clipseg-rd64-refined")

	prompts = ['vehicles', 'pedestrians', 'cyclists', 'road areas']
	for sample in train_ds:
		frames, _ = sample
		
		for frame in frames:
			mask = semantic_segmentation_tool(frame, prompts)
			break

		break
	
