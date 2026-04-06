# %%
from typing import List, Tuple
import PIL

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchreid
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

from smolagents import Tool


class ReIDTool(Tool):
	name = "reidentification_tool"
	description = """
	Determine whether a person or object in the input RGB image matches  a previously observed identity.
	The tool compares the visual features of the detected instance with stored identity embeddings and returns the matched identity label."
	"""
	inputs = {
		"image": {
			"type": "image",
			"description": "Input RGB image containing the person or object to identify.",
		},
	}
	output_type = "image"
		
	def __init__(self,
		threshold: float,
		model_name,
		device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	):
		super().__init__()
		self.threshold = threshold
		self.model_name, self.device = model_name, device
		self.size = (256, 128)

	def reset(self):
		self.embedding_db = []
		self.id_db = []
		self.target_features = []

	def setup(self):
		self.yolo = YOLO("yolov8n.pt")
		self.yolo.eval()

		self.model = torchreid.models.build_model(
			name=self.model_name,
			num_classes=1000,
			loss="softmax",
			pretrained=True
		).to(self.device)
		for p in self.model.parameters(): p.requires_grad_(False)
		self.model.eval()

		self.transform = transforms.Compose([
			transforms.Resize((256, 128)),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		])

		self.reset()
		super().setup()

	def extract_embedding(self, images: PIL.Image.Image) -> PIL.Image.Image:
		images = torch.stack([
			self.transform(img)
			for img in images
		], dim=0).to(self.device)

		with torch.no_grad():
			emb = self.model(images)
			emb = F.normalize(emb, dim=1)

		return emb  # shape: (1, D)

	def forward(self, image: PIL.Image.Image):
		w, h = image.size
		center = (w/2, h/2)
		mask = PIL.Image.new("L", (w, h), 0)
		
		results = self.yolo(image, verbose=False)
		bboxes = [
			bbox
			for bbox in results[0].boxes
			if bbox.cls == 0 and bbox.conf > 0.6 # human
		]
		if len(bboxes) == 0: return mask
		
		crop_imgs = [
			image.crop(*bbox.xyxy.cpu().tolist())
			for bbox in bboxes
		]

		# get best match "idx"
		if len(self.target_features) == 0:
			dist = [
				abs( 
					((bbox.xyxy[0, 0]+bbox.xyxy[0, 2])/2-center[0]) + ((bbox.xyxy[0, 1]+bbox.xyxy[0, 3])/2-center[1])
				) for bbox in bboxes
			]
			idx = dist.index(min(dist))
		else:
			vis_features = self.extract_embedding(crop_imgs)

			sim = cosine_similarity(vis_features.cpu().numpy(), self.target_features.cpu().numpy())  # shape: (N, K)
			idx = sim.mean(-1).argmax(axis=-1).item()
		
		if len(self.target_features) == 0:
			self.target_features = self.extract_embedding([crop_imgs[idx]])  # shape: (1, D)
		else:
			self.target_features = torch.cat([self.target_features, vis_features[idx:idx+1]])
			if len(self.target_features) > 10:
				self.target_features = self.target_features[-10:]

		# draw bounding box mask
		sel_bbox = bboxes[idx]
		draw = PIL.ImageDraw.Draw(mask)
		x1, y1, x2, y2 = map(int, sel_bbox.xyxy[0].tolist())
		x1, y1 = max(0, x1), max(0, y1)
		x2, y2 = min(w, x2), min(h, y2)
		draw.rectangle([x1, y1, x2, y2], fill=1)

		return mask
	

if __name__ == "__main__":
	from refiner.kitti_tracking import KittiDataset
	from matplotlib import pyplot as plt

	model_name = "osnet_x0_25"
	reid_tool = ReIDTool(0.5, model_name)
	reid_tool.setup()

	root_dir = "E:/KittiTracking"
	train_ds = KittiDataset(root_dir, "train", 8, 3)

	for sample in train_ds:
		frames, _ = sample

		for frame_idx, frame in enumerate(frames):
			mask = reid_tool.forward(frame)
			
			plt.imshow(mask, cmap="grey")
			plt.axis("off")
			plt.show()

			if frame_idx == 5: break
		break

# %%
