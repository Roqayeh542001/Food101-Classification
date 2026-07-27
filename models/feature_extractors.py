import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super().__init__()

        self.model = model
        self.features = None

        layer = dict(self.model.named_modules())[layer_name]

        layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output

    def forward(self, x):
        _ = self.model(x)

        return torch.flatten(self.features, 1)