# Classification_MNIST_CNN

This repository contains a Convolutional Neural Network (CNN) model for classifying the MNIST dataset.

## Overview

The `classification_MNIST_CNN.ipynb` and `classification_MNIST_CNN.py` files in this repository implement a CNN model for classifying handwritten digits from the MNIST dataset. The model is trained using the TensorFlow framework.

## Why CNN (Convolutional Neural Network) was used:
- CNNs are perfect for handling picture data because they can recognize features at many levels of abstraction, from edges to intricate textures, and they can record hierarchical spatial patterns. They use parameter sharing to reduce overfitting and enhance generalization, and they offer translation invariance, which is necessary for applications like picture classification. CNNs also automatically extract features, which eliminates the need for human feature engineering.

## Reasons for CNN Parameters:
- Convolutional Layer (32 filters, 3x3 kernel, ReLU activation): Convolutional layers are responsible for feature extraction. The choice of 32 filters with a 3x3 kernel is to capture various patterns in the images. ReLU activation is used to introduce non-linearity.

- Max-Pooling Layer (2x2 pool size): Max-pooling layers reduce the spatial dimensions of the feature maps, aiding in translation invariance and computational efficiency.

- Flatten Layer: This layer reshapes the output from the previous layers into a 1D vector, preparing it for fully connected layers.

- Dropout (0.2): Dropout is used to prevent overfitting. A dropout rate of 0.2 means that 20% of the neurons are randomly dropped out during training, which helps in generalization.

- Dense Layers (128 neurons, ReLU activation): These fully linked layers use the retrieved characteristics to conduct categorization. The 128 neurons that were selected are fairly random and may be changed based on the task's difficulty and the available processing power. Non-linearity is introduced by ReLU activation.

- Output Layer (10 neurons, Softmax activation): The output layer has 10 neurons, matching the number of classes in MNIST. Softmax activation converts the final layer's raw scores into class probabilities.

## Usage

To use the CNN model, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/Nehlr1/Classification_MNIST_CNN.git
   ```
   
2. Creating Python environment
    ```
    cd Classification_MNIST_CNN
    py -m pip install --user virtualenv
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. Open the `classification_MNIST_CNN.ipynb` notebook or run the `classification_MNIST_CNN.py` script using a Python environment.

4. Follow the instructions provided in the notebook or script to train and evaluate the CNN model on the MNIST dataset.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.