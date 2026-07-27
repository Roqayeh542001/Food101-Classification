from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchvision.models import (
    alexnet,
    googlenet,
    resnet50,
    convnext_tiny
)
from models.foodnet import FoodNet

from data.data_setup import class_names

import torch



def create_models():
    # AlexNet
    alex_model = alexnet(weights=None)
    alex_model.classifier[6] = torch.nn.Linear(
        4096,
        len(class_names)
    )

    # GoogLeNet
    googlenet_model = googlenet(weights=None, aux_logits=False)
    googlenet_model.fc = torch.nn.Linear(
        1024,
        len(class_names)
    )

    # ResNet50
    resnet_model = resnet50(weights=None)
    resnet_model.fc = torch.nn.Linear(
        2048,
        len(class_names)
    )

    # FoodNet
    foodnet_model = FoodNet(
        num_classes=len(class_names)
    )

    # ConvNeXt_Tiny
    convnext_model = convnext_tiny(weights=None)

    convnext_model.classifier[2] = torch.nn.Linear(768, len(class_names))

    models = {
        "AlexNet": {
            "model": alex_model,
            "checkpoint": ROOT / "checkpoints" / "alexnet_best.pth"
        },

        "GoogLeNet": {
            "model": googlenet_model,
            "checkpoint": ROOT / "checkpoints" / "googlenet_best.pth"
        },

        "ResNet50": {
            "model": resnet_model,
            "checkpoint": ROOT / "checkpoints" / "resnet50_best.pth"
        },

        "FoodNet": {
            "model": foodnet_model,
            "checkpoint": ROOT / "checkpoints" / "foodnet_best.pth"
        },

        "ConvNeXt": {
            "model": convnext_model,
            "checkpoint": ROOT / "checkpoints" / "convnext_best.pth"
        }
    }

    return models