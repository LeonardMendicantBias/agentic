from typing import List, Tuple
import PIL

import numpy as np

import torch
import torchvision.transforms as transforms

from smolagents import Tool

from .DSTT_OM import InpaintGenerator as DSTT_OM
from .DSTT_R import InpaintGenerator as DSTT_R


class InpaintingTool(Tool):
	name = "inpainting_tool"
	description = """
	Reconstruct missing RGB regions in a masked video frame using temporal information from previous frames.
	The tool fills the masked regions and returns a reconstructed RGB frame.
	"""
	inputs = {
		"masked_frame": {
			"type": "image",
			"description": "RGB frame with missing regions to be reconstructed.",
		},
		"mask": {
			"type": "image",
			"description": "Binary mask indicating missing regions (1 = hole, 0 = valid pixel).",
		}
	}
	output_type = "image"
	
	def __init__(self, 
		ckpt: str,
		ref_step: int=10, n_refs: int=10,
		n_neighbors: int=4,
		refine_window: int=4, online_window: int=3,
		device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	):
		super().__init__()
		self.ckpt = ckpt
		self.device = device

		self.ref_step, self.n_refs = ref_step, n_refs
		self.n_neighbors = n_neighbors
		self.refine_window, self.online_window = refine_window, online_window
		self.refine_length = 8

		self._h, self._w = 240, 432  # (200, 360) | (40, 72)

	def reset(self):
		self._frame_idx = 0
		# self.inpainting_memory = torch.Tensor().to(self.device)
		self.masked_frames = torch.Tensor().to(self.device)
		self.refine_stack = torch.Tensor().to(self.device)
		self.online_stack = torch.Tensor().to(self.device)

	def setup(self):
		self.online_model = DSTT_OM().to(self.device)
		data = torch.load(self.ckpt, map_location=self.device)
		self.online_model.load_state_dict(data['netG'])
		self.online_model.eval()
		for p in self.online_model.parameters(): p.requires_grad = False

		self.refine_model = DSTT_R().to(self.device)
		data = torch.load(self.ckpt, map_location=self.device)
		self.refine_model.load_state_dict(data['netG'])
		self.refine_model.eval()
		for p in self.refine_model.parameters(): p.requires_grad = False
		
		self.transforms = transforms.Compose([
			transforms.Resize((self._h, self._w)),
			transforms.ToTensor(),
			transforms.Lambda(lambda x: 2*x-1)
		])
		self.reset()
		super().setup()

	def get_ref_index(self, neighbor_ids):
		ref_index = []
		i = self._frame_idx - 1
		while(i >= 0 and len(ref_index) < self.n_refs):
			if not i in neighbor_ids and not i%self.ref_step:
				# if comp is not None:
				# 	if comp[i]:
				# 		ref_index.append(i)
				# else:
				ref_index.append(i)
			i -= 1
		return ref_index[::-1]

	def forward(self,
		masked_frame: PIL.Image.Image,  # (W, H)
		mask: PIL.Image.Image,  # binary image
	) -> PIL.Image.Image:
		resized_frame = masked_frame.resize((self._w, self._h))
		inp_frame = self.transforms(resized_frame).unsqueeze(0).unsqueeze(1).to(self.device)
		mask_np = np.array(mask.resize((self._w, self._h))) / 255

		_frame = inp_frame*(torch.tensor(1-mask_np, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0))

		# add to list
		self.masked_frames = torch.cat((self.masked_frames, _frame), dim=1)
		if self.masked_frames.shape[1] > self.refine_length+1:
			self.masked_frames = self.masked_frames[:, -self.refine_length-1:]

		if self._frame_idx > self.refine_window:			
			attn = self.refine_model(self.masked_frames)
			attn = attn.view(8, 1, -1, 720, 512).permute(2, 0, 1, 3, 4)
			if len(self.refine_stack) == 0:
				self.refine_stack = attn
			else:
				# print(refine_stack.shape, attn.shape)
				self.refine_stack = 0.5 * self.refine_stack + 0.5 * attn[:-1]
				self.refine_stack = torch.cat([self.refine_stack, attn[-1:]])
				
				if len(self.refine_stack) > self.refine_length:
					self.refine_stack = self.refine_stack[-self.refine_length:]

		neighbor_ids = [
			i-max(0, self._frame_idx-self.refine_length)
			for i in range(max(0, self._frame_idx-10), self._frame_idx)
			if i > self.refine_window
		][-self.n_neighbors:]
		ref_ids = [
			i for i in range(0, len(self.refine_stack), self.ref_step) 
			if i not in neighbor_ids and self._frame_idx > i
		]
		ids = neighbor_ids + ref_ids

		refine_attn = self.refine_stack[ids]

		# Prediction using information from the refine memory and the current online memory
		pred_img, attn = self.online_model(_frame, torch.cat((self.online_stack[-self.online_window:], refine_attn)))
		# Memory update
		self.online_stack = torch.cat((self.online_stack, attn.unsqueeze(0)))
		# to prevent memory creep
		if len(self.online_stack) > self.online_window:
			self.online_stack = self.online_stack[-self.online_window:]

		pred_img = (pred_img + 1) / 2
		pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
		
		frame_np = np.array(resized_frame).astype(np.uint8)
		# binary_mask = (np.array(mask_np[0, 0].permute(1, 2, 0).cpu()) != 0).astype(np.uint8)
		binary_mask = (mask_np < 0.5).astype(np.uint8)
		img = np.array(pred_img[0]).astype(np.uint8)*(1-np.expand_dims(binary_mask, axis=-1)) + frame_np*(np.expand_dims(binary_mask, axis=-1))

		self._frame_idx += 1

		_img = PIL.Image.fromarray(np.array(img).astype(np.uint8))
		return _img.resize(masked_frame.size)
	
if __name__ == "__main__":
	from matplotlib import pyplot as plt
