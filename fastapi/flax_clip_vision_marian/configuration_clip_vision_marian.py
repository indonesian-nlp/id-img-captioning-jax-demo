import copy

from transformers import CLIPVisionConfig, MarianConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CLIPVisionMarianConfig(PretrainedConfig):

    model_type = "clip-vision-marian"
    is_composition = True
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        if "marian_config" not in kwargs:
            raise ValueError("`marian_config_dict` can not be `None`.")

        if "clip_vision_config" not in kwargs:
            raise ValueError("`clip_vision_config_dict` can not be `None`.")

        marian_config = kwargs.pop("marian_config")
        clip_vision_config = kwargs.pop("clip_vision_config")
        
        self.marian_config = MarianConfig(**marian_config)

        self.clip_vision_config = CLIPVisionConfig(**clip_vision_config)

        self.is_encoder_decoder = True

    @classmethod
    def from_clip_vision_marian_configs(
        cls,
        clip_vision_config: PretrainedConfig,
        marian_config: PretrainedConfig,
        **kwargs
    ):
        return cls(
            clip_vision_config=clip_vision_config.to_dict(),
            marian_config=marian_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["clip_vision_config"] = self.clip_vision_config.to_dict()
        output["marian_config"] = self.marian_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
