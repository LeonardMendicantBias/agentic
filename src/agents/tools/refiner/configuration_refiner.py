from transformers.utils import logging

from transformers.models.clipseg.configuration_clipseg import CLIPSegConfig


logger = logging.get_logger(__name__)


class RefinerConfig(CLIPSegConfig):
    
    def __init__(self, action_channels, **kwargs):
        super().__init__(**kwargs)

        self.action_channels = action_channels


__all__ = ["RefinerConfig"]
