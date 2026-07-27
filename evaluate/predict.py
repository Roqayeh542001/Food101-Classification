from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from config import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD
)

from torchvision import transforms

from torchvision.models import (
    alexnet,
    googlenet,
    resnet50
)
from models.foodnet import FoodNet

from data.histogram_equalization import HistogramEqualization

from data.data_setup import class_names

from models.model_factory import create_models

test_transform = transforms.Compose([
    HistogramEqualization(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])



# Create models
models = create_models()

for info in models.values():
    checkpoint = torch.load(info["checkpoint"], map_location=DEVICE)

    info["model"].load_state_dict(checkpoint["model_state_dict"])

    info["model"].to(DEVICE)

    info["model"].eval()


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    x = test_transform(image)

    x = x.unsqueeze(0).to(DEVICE)

    results = {}

    with torch.inference_mode():
        for model_name, info in models.items():
            model = info["model"]
            
            output = model(x)
            probs = torch.softmax(output, dim=1)

            confidence, pred = torch.max(probs, dim=1)
            confidence = confidence.item() * 100

            if confidence < 70:
                prediction = f"{class_names[pred.item()]} ?"
            else:
                prediction = class_names[pred.item()]


            results[model_name] = (
                prediction,
                confidence
            )

    return results
