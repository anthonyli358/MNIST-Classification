# MNIST-Classification
Using Keras to classify images from the MNIST dataset.

## Results
### MLP (Multilayer Perceptron)
Overfitting is a common issue in deep learning models and despite using Dropout layers this is particularly evident in the loss plot. 
To minimise this the dropout should be increased or regularization layers introduced.

<p float="left">
    <img src="results/model_evaluation/mlp_loss.png" alt="mlp_loss" width="375"/> 
    <img src="results/model_evaluation/mlp_accuracy.png" alt="mlp_accuracy" width="375"/> 
</p>

### CNN (Convolutional Neural Network)
The CNN outperforms the MLP without overfitting. The BatchNormalization layers are therefore not included.
<p float="left">
    <img src="results/model_evaluation/mlp_loss.png" alt="mlp_loss" width="375"/> 
    <img src="results/model_evaluation/mlp_accuracy.png" alt="mlp_accuracy" width="375"/> 
</p>

## Getting Started
- Change the ml_model variable at the bottom of main.py to mlp or cnn and run

## Development
- Image classification using MLP and CNN neural networks in Keras
- Model architectures plots available in the 'results/models' folder
- Evaluation metrics discussed for multi-class problems
