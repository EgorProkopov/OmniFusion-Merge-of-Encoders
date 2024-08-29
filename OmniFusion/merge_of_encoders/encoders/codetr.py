from PIL import Image

import torch
from torch import nn
from transformers import ConditionalDetrImageProcessor, ConditionalDetrModel, ConditionalDetrConfig


def expand2square(pil_img, background_color=0):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class CustomCoDetrImageProcessor(ConditionalDetrImageProcessor):
    def __call__(self, images=None, return_tensors=None, **kwargs):
        return super().__call__(expand2square(images).resize((640, 640)),
                                return_tensors=return_tensors,
                                do_center_crop=False, **kwargs)


class CoDETRVisionTower(nn.Module):
    def __init__(self, vision_tower, delay_load=False, patch='cls_patch'):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_feature = patch

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ConditionalDetrConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CustomCoDetrImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = ConditionalDetrModel.from_pretrained(self.vision_tower_name)
        self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.dtype)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.encoder_last_hidden_state
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                self.vision_tower = self.vision_tower.to(device=self.device, dtype=self.dtype)
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            self.vision_tower=self.vision_tower.to(device=self.device, dtype=self.dtype)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
