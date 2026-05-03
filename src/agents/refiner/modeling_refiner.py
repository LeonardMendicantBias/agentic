from typing import Optional, Union, List, Any
from dataclasses import dataclass
import PIL
import copy
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from smolagents import Tool

from transformers.utils import ModelOutput
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from transformers.models.clipseg.modeling_clipseg import CLIPSegModel, CLIPSegDecoder, CLIPSegDecoderLayer
from transformers.models.clipseg.modeling_clipseg import CLIPSegOutput, CLIPSegDecoderOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.agents.refiner.configuration_refiner import RefinerConfig

from transformers import BatchEncoding


@dataclass
class RefinerOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Binary cross entropy loss for segmentation.
    logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        Classification scores for each pixel.
    conditional_embeddings (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
        Conditional embeddings used for segmentation.
    pooled_output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`):
        Pooled output of the [`CLIPSegVisionModel`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPSegVisionModel`].
    decoder_output (`CLIPSegDecoderOutput`):
        The output of the [`CLIPSegDecoder`].
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    values: Optional[torch.FloatTensor] = None
    conditional_embeddings: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    vision_model_output: BaseModelOutputWithPooling = None
    decoder_output: CLIPSegDecoderOutput = None
    critic_output: CLIPSegDecoderOutput = None

    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["vision_model_output", "decoder_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
    

@dataclass
class RefinerDecoderOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, height, width)`):
        Classification scores for each pixel.
    """

    logits: Optional[torch.FloatTensor] = None
    values: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None

class MyProcessor(CLIPSegProcessor):
    def __call__(
        self,
        task: Optional[Union[str, List[str]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        images: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> BatchEncoding:
        if task is not None and isinstance(task, str):
            task = [task]

        if text is not None and isinstance(text, str):
            text = [text]

        if images is not None and isinstance(images, PIL.Image.Image):
            images = [images]

        return super().__call__(task=task, text=text, images=images, return_tensors=return_tensors, **kwargs)


class RefinerDecoder(CLIPSegDecoder):
    
    def __init__(self, config: RefinerConfig):
        super().__init__(config)

        self.conditional_layer = config.conditional_layer

        self.film_mul = nn.Linear(config.projection_dim, config.reduce_dim)
        self.film_add = nn.Linear(config.projection_dim, config.reduce_dim)

        if config.use_complex_transposed_convolution:
            transposed_kernels = (config.vision_config.patch_size // 4, config.vision_config.patch_size // 4)

            self.transposed_convolution = nn.Sequential(
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                # NEW: dimension output is action_channels
                nn.ConvTranspose2d(
                    config.reduce_dim // 2, config.action_channels, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
            self.value_transposed_convolution = nn.Sequential(
                nn.Conv2d(config.reduce_dim, config.reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    config.reduce_dim,
                    config.reduce_dim // 2,
                    kernel_size=transposed_kernels[0],
                    stride=transposed_kernels[0],
                ),
                nn.ReLU(),
                # NEW: dimension output is action_channels
                nn.ConvTranspose2d(
                    config.reduce_dim // 2, 1, kernel_size=transposed_kernels[1], stride=transposed_kernels[1]
                ),
            )
        else:
            # NEW: dimension output is action_channels

            self.transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, config.action_channels, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )
            self.value_transposed_convolution = nn.ConvTranspose2d(
                config.reduce_dim, 1, config.vision_config.patch_size, stride=config.vision_config.patch_size
            )

        depth = len(config.extract_layers)
        self.reduces = nn.ModuleList(
            [nn.Linear(config.vision_config.hidden_size, config.reduce_dim) for _ in range(depth)]
        )

        decoder_config = copy.deepcopy(config.vision_config)
        decoder_config.hidden_size = config.reduce_dim
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.hidden_act = "relu"
        self.layers = nn.ModuleList([CLIPSegDecoderLayer(decoder_config) for _ in range(len(config.extract_layers))])

    def forward(
        self,
        hidden_states: tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> RefinerDecoderOutput:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1)  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.view(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)
        values = self.value_transposed_convolution(output).squeeze(1)

        if not return_dict:
            return tuple(v for v in [logits, values, all_hidden_states, all_attentions] if v is not None)

        return RefinerDecoderOutput(
            logits=logits,
            values=values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    
class Refiner(CLIPSegForImageSegmentation):
    config: RefinerConfig

    def __init__(self, config: RefinerConfig):
        super().__init__(config)

        self.config = config

        self.clip = CLIPSegModel(config)
        self.extract_layers = config.extract_layers

        self.decoder = RefinerDecoder(config)  # agent

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        conditional_pixel_values: Optional[torch.FloatTensor] = None,
        conditional_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = True,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CLIPSegOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the query images through the frozen CLIP vision encoder
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                interpolate_pos_encoding=interpolate_pos_encoding,
                return_dict=return_dict,
            )
            pooled_output = self.clip.visual_projection(vision_outputs[1])

            hidden_states = vision_outputs.hidden_states if return_dict else vision_outputs[2]
            # we add +1 here as the hidden states also include the initial embeddings
            activations = [hidden_states[i + 1] for i in self.extract_layers]

            # update vision_outputs
            if return_dict:
                vision_outputs = BaseModelOutputWithPooling(
                    last_hidden_state=vision_outputs.last_hidden_state,
                    pooler_output=vision_outputs.pooler_output,
                    hidden_states=vision_outputs.hidden_states if output_hidden_states else None,
                    attentions=vision_outputs.attentions,
                )
            else:
                vision_outputs = (
                    vision_outputs[:2] + vision_outputs[3:] if not output_hidden_states else vision_outputs
                )

        # step 2: compute conditional embeddings, either from text, images or an own provided embedding
        if conditional_embeddings is None:
            conditional_embeddings = self.get_conditional_embeddings(
                batch_size=pixel_values.shape[0],
                # task_input_ids=task_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                conditional_pixel_values=conditional_pixel_values,
            )
        else:
            if conditional_embeddings.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Make sure to pass as many conditional embeddings as there are query images in the batch"
                )
            if conditional_embeddings.shape[1] != self.config.projection_dim:
                raise ValueError(
                    "Make sure that the feature dimension of the conditional embeddings matches"
                    " `config.projection_dim`."
                )

        # step 3: forward both the pooled output and the activations through the lightweight decoder to predict masks
        decoder_outputs = self.decoder(
            activations,
            conditional_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        values = decoder_outputs.values if return_dict else decoder_outputs[1]

        loss = None
        if labels is not None:
            # move labels to the correct device to enable PP
            labels = labels.to(logits.device)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        if not return_dict:
            output = (logits, conditional_embeddings, pooled_output, vision_outputs, decoder_outputs)
            return ((loss,) + output) if loss is not None else output

        return RefinerOutput(
            loss=loss,
            logits=logits,
            values=values,
            conditional_embeddings=conditional_embeddings,
            pooled_output=pooled_output,
            vision_model_output=vision_outputs,
            decoder_output=decoder_outputs,
        )

__all__ = [
    "Refiner",
]


if __name__ == "__main__":
    model_name = "CIDAS/CLIPSeg-ViT-B-16"
    processor = CLIPSegProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPSegForImageSegmentation.from_pretrained(model_name).to("cuda:0")
    model.eval()