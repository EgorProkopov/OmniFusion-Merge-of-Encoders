import torch
from torch import nn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TrOCRConfig

class TrOCRVisionTower(nn.Module):
    def __init__(self, pth="microsoft/trocr-base-handwritten", select_feature='patch'):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained(pth).encoder
        self.image_processor = TrOCRProcessor.from_pretrained(pth).image_processor
        self.cfg_only = TrOCRConfig.from_pretrained(pth)
        self.select_feature = select_feature

    @torch.no_grad()
    def forward(self, x):
        x = x.to(device=self.device, dtype=self.dtype)
        out = self.model(x)
        selected_features = self.feature_select(out)
        return selected_features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.last_hidden_state
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

if __name__ == "__main__":
    image = torch.randint(0, 255, (1, 3, 256, 256))
    print(f"Input batch shpe is {image.shape}")
    trocr = TrOCRVisionTower()

    image = trocr.image_processor(image, return_tensors="pt").pixel_values
    result = trocr.forward(image)
    print(f"Result shape is {result.shape}")
