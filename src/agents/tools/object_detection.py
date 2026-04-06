# %% Begin of code
from typing import List
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput as DetectionOutput
from accelerate import Accelerator

from smolagents import Tool


class ObjectDetectionTool(Tool):
	name = "object_detection_tool"
	description = """
	This is a tool that performs object detection given specific classes.
	It returns a string summarize the number of bounding boxes and the pixel location of each bounding box in 'xyxy' format.
	"""
	inputs = {
		"class": {
			"type": "string",
			"description": "the object class",
		}
	}
	# output_type = "string"
	output_type = "array"

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.model_id = "IDEA-Research/grounding-dino-tiny"
		self.device = Accelerator().device

		self.processor = AutoProcessor.from_pretrained(self.model_id)
		self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)
		
		for p in self.model.parameters(): p.requires_grad_(False)
		self.model.eval()
		self.model.to(self.device)
	
	@torch.autocast(device_type="cuda")
	def forward(self, image: Image, text: List[str]):
		inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
		outputs: DetectionOutput = self.model(**inputs)
		results: List = self.processor.post_process_grounded_object_detection(
			outputs,
			inputs.input_ids,
			# box_threshold=0.3,
			text_threshold=0.3,
			target_sizes=[image.size[::-1]]  # (H, W)
		)
		# boxes = results[0]["boxes"]
		# return f"Found {len(boxes)} bounding boxes: {'-'.join(bbox for bbox in boxes)}"
		return results
	
if __name__ == "__main__":
	from refiner.kitti_tracking import KittiDataset
	
	object_detection_tool = ObjectDetectionTool()
