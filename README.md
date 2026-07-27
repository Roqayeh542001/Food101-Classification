# Food Image Classification using Deep Learning

### Overview

This project compares several deep convolutional neural networks for food image classification on the Food101 dataset.

The following architectures are implemented and evaluated:

- AlexNet
- GoogLeNet
- ResNet50
- ConvNeXt-Tiny
- FoodNet (Ensemble Model)

The project also compares these models in terms of:

- Test Accuracy
- Training Loss
- Test Loss
- Number of Parameters
- Training Time
- Confusion Matrix
- Classification Report
- Real Image Prediction
- Dataset

#### Dataset:

Food101

#### Classes used in this project:

- Pizza
- Steak
- Sushi

#### Project Structure
```text
Food101-Classification/
│
├── checkpoints/
├── data/
├── evaluate/
├── models/
├── train/
├── utils/
├── config.py
└── README.md
```

**Note:**
After running the training scripts, the following files will be generated automatically:
```text
results/
├── figures/
├── history/
├── predictions/
├── reports/
├── model_test_metrics.csv
└── model_comparison.csv

checkpoints/
├── alexnet_best.pth
├── convenext_best.pth
├── foodnet_best.pth
├── googlenet_best.pth
└── resnet50_best.pth
```

Models
| Model |	Description |
|-------|-------------|
| AlexNet |	Classical CNN architecture |
| GoogLeNet	| Inception architecture |
| ResNet50	| Residual Network |
| ConvNeXt-Tiny |	Modern CNN architecture |
| FoodNet |	Ensemble model using AlexNet + GoogLeNet + ResNet50 features |

#### Training Strategy

All pretrained models use ImageNet pretrained weights.

Two transfer learning strategies were used:

1. Feature Extraction
   
   Freeze all backbone layers
   
   Train only the final classifier

3. Fine-Tuning
   
   Unfreeze the last stage of ConvNeXt
   
   Train the classifier together with the last feature extraction stage

#### FoodNet Architecture

FoodNet combines feature vectors extracted from three pretrained models.

- AlexNet  -> 4096 features
- GoogLeNet ->1024 features
- ResNet50 ->2048 features

Concatenate

↓

Fully Connected

↓

Food Prediction

Total concatenated feature vector:

4096 + 1024 + 2048 = 7168

Classifier:

7168
 ↓
1024
 ↓
3 classes
Evaluation

#### The following metrics are reported:

- Accuracy
- Loss
- Confusion Matrix
- Classification Report
- Prediction Confidence
- Results

#### Example comparison:

| Model |	Best Accuracy |
|-------|---------------|
| AlexNet	| 90.27% |
| GoogLeNet	| 77.20% |
| ResNet50 |	96.40% |
| ConvNeXt	| 98.40% |
| FoodNet |	96.13% |


#### Requirements
- Python 3.14.2
- PyTorch
- Torchvision
- Matplotlib
- Pandas
- Scikit-learn
- tqdm
- Pillow
- OpenCV

Install:
pip install -r requirements.txt

#### Run

**Train AlexNet**

python train/train_alexnet.py

**Train GoogLeNet**

python train/train_googlenet.py

**Train ResNet50**

python train/train_resnet50.py

**Train ConvNeXt**

python train/train_convnext.py

**Train FoodNet**

python train/train_foodnet.py

**Evaluate**

python evaluate/test_metrics.py

**Predict Real Images**

python evaluate/predict_real_image.py
Author

**Roqayeh**

**Deep Learning Course Project**
