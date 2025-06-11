# CIFAR-10 Image Classification with PyTorch

This repository contains Jupyter notebooks implementing deep convolutional neural networks for image classification on the CIFAR-10 dataset using PyTorch.

## Project Overview

The project focuses on training and evaluating models for classifying images from the CIFAR-10 dataset into 10 different categories: airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

Two main approaches are implemented:
1. **Transfer Learning with Feature Extraction** (`cifar_classifier.ipynb`): Using a pre-trained ResNet18 model where only the final classification layers are trained
2. **Full Fine-Tuning** (`resnet18_finetuning.ipynb`): Fine-tuning the entire ResNet18 model for the CIFAR-10 dataset

## Dataset

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of:
- 60,000 32x32 color images
- 10 classes with 6,000 images per class
- 50,000 training images and 10,000 test images

## Implementation Details

### Data Augmentation
Several data augmentation techniques are applied to improve model generalization:
- Random horizontal flips
- Random rotations
- Random crops with padding
- Color jitter (in the fine-tuning notebook)

### Model Architecture
- **Base Architecture**: ResNet18 pre-trained on ImageNet
- **Modifications**: Custom fully connected layers added to the end of the network
- **Feature Extraction**: Only training the new fully connected layers while freezing pre-trained layers (in `cifar_classifier.ipynb`)
- **Fine-Tuning**: Training the entire network (in `resnet18_finetuning.ipynb`)

### Training Process
- **Optimizer**: SGD with momentum and Adam
- **Learning Rate Scheduling**: ReduceLROnPlateau to reduce learning rate when validation loss plateaus
- **Loss Function**: Cross-Entropy Loss
- **Hardware Acceleration**: Support for CUDA (NVIDIA GPUs) and MPS (Apple Silicon)

## Results

The notebooks include:
- Training and validation accuracy/loss curves
- Confusion matrices for model evaluation
- Per-class accuracy metrics
- Visualization of model predictions

## Requirements

- Python 3.8+
- PyTorch 1.7+
- torchvision
- matplotlib
- numpy
- scikit-learn
- seaborn
- tqdm
- pandas

## Usage

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt  # Not included, you may need to create this
```

3. Open and run the Jupyter notebooks:
```bash
jupyter notebook
```

4. The notebooks will download the CIFAR-10 dataset automatically when run

## Future Improvements

Potential areas for improvement include:
- Implementing more advanced architectures (ResNet50, EfficientNet, Vision Transformer)
- Exploring different learning rate schedules
- Implementing more sophisticated data augmentation techniques
- Ensemble methods to combine multiple models

## License

[MIT License](LICENSE)  # Add your preferred license

## Acknowledgments

- The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- ResNet architecture was developed by Kaiming He et al. 