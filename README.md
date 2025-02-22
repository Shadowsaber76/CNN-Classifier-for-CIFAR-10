# CNN Classifier for CIFAR-10

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The model is trained with data augmentation and evaluated using accuracy and F1-score.

## What is a CNN?
A Convolutional Neural Network (CNN) is a deep learning model designed for image recognition tasks. It consists of convolutional layers that extract features, pooling layers that reduce dimensionality, and fully connected layers that classify the output. CNNs are widely used in computer vision due to their ability to learn spatial hierarchies of features.

## Features
- **Dataset**: CIFAR-10
- **Data Augmentation**: Random horizontal flip, random cropping
- **CNN Architecture**:
  - Three convolutional layers with increasing filters (64, 128, 256)
  - Max pooling(Downsampling) and ReLU activation(For faster training and non-linearity)
  - Dropout for regularization(To prevent Overfitting)
  - Fully connected layers with softmax output
- **Optimization**:
  - Adam optimizer with weight decay
  - Learning rate scheduler (`StepLR`)
- **Metrics**: Accuracy and weighted F1-score

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install torch torchvision datasets scikit-learn matplotlib numpy
```

## Dataset
This project uses the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) loaded via the `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset("uoft-cs/cifar10")
```

## Usage
- Run the training script
- Modify the `num_epochs` parameter in the script to adjust training duration. (Genrally more the number of epochs better is the Train Accuracy but extreme number of epochs can cause overfitting leading to dip in Test Accuracy)

## Evaluation
The script evaluates the model after each epoch using accuracy and F1-score:
```python
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1
```

## Results
- Training and testing accuracy is printed during training.

![image](https://github.com/user-attachments/assets/619ed5e3-b123-4a6b-9296-c8fbd16923c9)

![image](https://github.com/user-attachments/assets/1a0f6a2f-c40d-425f-b579-5053d3ccd9e3)

![image](https://github.com/user-attachments/assets/9e0923ae-256a-4729-9ca5-5a3c45d80384)

Made with ❤️ by Gaurav
Please ignore the dataset import warning

## License
This project is open-source under the MIT License.

