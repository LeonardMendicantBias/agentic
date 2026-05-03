# %%
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
		Sequentially inpaint masked regions in a stream of images one frame at a time.
		Maintains temporal state across calls. Reset between sequences.
	"""
	inputs = {
		"image": {
			"type": "image",
			"description": "RGB PIL image.",
		},
		"mask": {
			"type": "image",
			"description": "Binary PIL image (L mode), white=keep, black=inpaint.",
		},
	}
	output_type = "image"

	def __init__(self,
		ckpt: str,
		ref_step: int = 10, n_refs: int = 10,
		n_neighbors: int = 4,
		refine_window: int = 4, online_window: int = 3,
		device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
	):
		super().__init__()
		self.ckpt = ckpt
		self.device = device
		self.ref_step, self.n_refs = ref_step, n_refs
		self.n_neighbors = n_neighbors
		self.refine_window, self.online_window = refine_window, online_window
		self.refine_length = 8
		self._h, self._w = 240, 432

	def reset(self):
		self._frame_idx = 0
		self.masked_frames = torch.Tensor().to(self.device)
		self.refine_stack = torch.Tensor().to(self.device)
		self.online_stack = torch.Tensor().to(self.device)

	def setup(self):
		data = torch.load(self.ckpt, map_location=self.device)

		self.online_model = DSTT_OM().to(self.device)
		self.online_model.load_state_dict(data['netG'])
		self.online_model.eval()
		for p in self.online_model.parameters(): p.requires_grad = False

		self.refine_model = DSTT_R().to(self.device)
		self.refine_model.load_state_dict(data['netG'])
		self.refine_model.eval()
		for p in self.refine_model.parameters(): p.requires_grad = False

		self._to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Lambda(lambda x: 2 * x - 1),
		])
		self.reset()
		super().setup()

	def get_ref_index(self, neighbor_ids):
		ref_index = []
		i = self._frame_idx - 1
		while i >= 0 and len(ref_index) < self.n_refs:
			if i not in neighbor_ids and not i % self.ref_step:
				ref_index.append(i)
			i -= 1
		return ref_index[::-1]

	def forward(self,
		image: PIL.Image.Image,
		mask: PIL.Image.Image
	) -> PIL.Image.Image:
		"""
		image : PIL RGB image
		mask  : PIL binary image (L mode), white (255)=keep, black (0)=inpaint
		returns PIL RGB image at original resolution
		"""
		orig_size = image.size  # (W, H)

		# Resize to model resolution
		resized = image.resize((self._w, self._h))
		keep_r = np.array(mask.resize((self._w, self._h), PIL.Image.NEAREST)) / 255.0  # (h,w) in [0,1]

		keep_t = torch.tensor(keep_r, dtype=torch.float32, device=self.device)

		# Frame tensor: zero out inpaint regions (keep * frame)
		frame_t = self._to_tensor(resized).to(self.device)  # (3, h, w) in [-1,1]
		masked_t = (frame_t * keep_t).unsqueeze(0).unsqueeze(0)  # (1,1,3,h,w)

		# Accumulate masked frames buffer
		self.masked_frames = torch.cat((self.masked_frames, masked_t), dim=1)
		if self.masked_frames.shape[1] > self.refine_length + 1:
			self.masked_frames = self.masked_frames[:, -self.refine_length - 1:]

		# Refinement step (needs refine_window frames)
		if self._frame_idx >= self.refine_window:
			with torch.no_grad():
				attn = self.refine_model(self.masked_frames)
			attn = attn.view(8, 1, -1, 720, 512).permute(2, 0, 1, 3, 4)
			if len(self.refine_stack) == 0:
				self.refine_stack = attn
			else:
				self.refine_stack = 0.5 * self.refine_stack + 0.5 * attn[:-1]
				self.refine_stack = torch.cat([self.refine_stack, attn[-1:]])
				if len(self.refine_stack) > self.refine_length:
					self.refine_stack = self.refine_stack[-self.refine_length:]

		# Online model: select neighbor and reference attention indices
		offset = max(0, self._frame_idx - self.refine_length)
		neighbor_ids = [
			i - offset
			for i in range(max(0, self._frame_idx - 10), self._frame_idx)
			if i >= self.refine_window
		][-self.n_neighbors:]
		ref_ids = [
			i for i in range(0, len(self.refine_stack), self.ref_step)
			if i not in neighbor_ids and self._frame_idx > i
		]
		ids = neighbor_ids + ref_ids

		refine_attn = self.refine_stack[ids] if ids else torch.Tensor().to(self.device)
		combined_mem = torch.cat((self.online_stack[-self.online_window:], refine_attn))

		with torch.no_grad():
			pred_img, attn = self.online_model(masked_t, combined_mem)

		self.online_stack = torch.cat((self.online_stack, attn.unsqueeze(0)))
		if len(self.online_stack) > self.online_window:
			self.online_stack = self.online_stack[-self.online_window:]

		# Decode and composite: use prediction only where mask==0 (inpaint region)
		pred = (pred_img + 1) / 2  # [-1,1] → [0,1]
		pred = pred.cpu().squeeze(0).permute(1, 2, 0).numpy()  # (h, w, 3)
		pred = (pred * 255).clip(0, 255).astype(np.uint8)

		orig_resized = np.array(resized).astype(np.uint8)
		keep_hw = keep_r[..., np.newaxis]  # (h, w, 1) broadcast over channels
		composite = (pred * (1 - keep_hw) + orig_resized * keep_hw).astype(np.uint8)

		self._frame_idx += 1

		return PIL.Image.fromarray(composite).resize(orig_size)


if __name__ == "__main__":
	from matplotlib import pyplot as plt
	from src.agents.refiner.kitti_tracking import KittiDataset

	root_dir = "E:/KittiTracking"
	n_steps, n_pred_steps = 16, 3

	train_ds = KittiDataset(root_dir, "train", n_steps, n_pred_steps)

	inpainting_tool = InpaintingTool(model_name="../../../checkpoints/dstt.pth")

	for sample in train_ds:
		frames, _ = sample
		break
		
	for frame in frames:
		mask = inpainting_tool(frame)
		break
