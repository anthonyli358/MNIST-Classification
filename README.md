# MNIST-Classification
Using Keras to classify images from the MNIST dataset.

## Results
### MLP (Multilayer Perceptron)
Overfitting is a common issue in deep learning models and despite using Dropout layers this is particularly evident in the loss plot. 
To minimise this the dropout should be increased or regularization layers introduced.

<p align="left">
    <img src="results/model_evaluation/mlp_loss.png" alt="mlp_loss" width="400"/> 
    <img src="results/model_evaluation/mlp_accuracy.png" alt="mlp_accuracy" width="400"/> 
</p>

### CNN (Convolutional Neural Network)
The CNN outperforms the MLP without overfitting. BatchNormalization layers are therefore not included.
<p align="left">
    <img src="results/model_evaluation/cnn_loss.png" alt="cnn_loss" width="400"/> 
    <img src="results/model_evaluation/cnn_accuracy.png" alt="cnn_accuracy" width="400"/> 
</p>

## Getting Started
- Change the ml_model variable at the bottom of [main.py](main.py) to mlp or cnn and run

## Development
- Image classification using MLP and CNN neural networks in Keras
- Model architecture plots available in the 'results/model_architectures' folder
- Evaluation metrics discussed for multi-class problems
