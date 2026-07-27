import torch
from torch import nn

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    googlenet,
    GoogLeNet_Weights,
    resnet50,
    ResNet50_Weights
)

from models.feature_extractors import FeatureExtractor


class FoodNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        alex = alexnet(
            weights=AlexNet_Weights.DEFAULT
        )

        google = googlenet(
            weights=GoogLeNet_Weights.DEFAULT
        )

        resnet = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        for param in alex.parameters():
            param.requires_grad = False

        for param in google.parameters():
            param.requires_grad = False

        for param in resnet.parameters():
            param.requires_grad = False


        # Create Feature Extractor
        self.alex = FeatureExtractor(
            alex,
            "classifier.5"
        )

        self.google = FeatureExtractor(
            google,
            "avgpool"
        )

        self.resnet = FeatureExtractor(
            resnet,
            "avgpool"
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096+1024+2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        fa = self.alex(x)
        #print("Alex: ", fa.shape)

        fg = self.google(x)
        #print("Google: ", fg.shape)

        fr = self.resnet(x)
        #print("ResNet: ", fr.shape)

        x = torch.cat(
            [fa, fg, fr],
            dim=1
        )

        x = self.classifier(x)

        return x
    
if __name__ == "__main__":
    model = FoodNet(num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("output shape: ", y.shape)