# Pneumonia-Identification-CNN
Developed a Pneumonia Classification model using TensorFlow 2.0. Based on a picture given by a Chest X-RAY, the model is able to correctly identify 96.2% on unseen data, with a 97% accuracy on the training set.

## Breakdown
Model comprised of a 80/20% split between the training and validation datasets. Dataset comprised of images that indicate whether the X-RAY shows signs of pneumonia or not.

Model built as a Sequential Model from utilizing the TensorFlow 2.0 library.

Comprised of three convolutional layers (32, 64, 64)for feature identification, each with their own 2D Pooling and Dropout factors to help eliminate overfitting.

Layers flattened then connected to dense layers (128, 64, 16),

Utilized a Rectified Linear Unit (RELU) activation function. for each layer.

Used the ADAM optimizer along with a BINARY_CROSSENTROPY loss function.

Model saved in a .h5 format in the "pnemonia_trained.h5" for use.

## Examples from the dataset

### NORMAL

![](IM-0001-0001.jpeg)


