# Neural Style Transfer with TensorFlow
Implementation of Neural Style Transfer algorithm with TensorFlow in Python 3.
# TensorFlow Binary Image Classification using CNN's
This is a binary image classification project using Convolutional Neural Networks and TensorFlow API (no Keras) on Python 3.
[Read all story in Turkish](https://medium.com/@mubuyuk51/tensorflow-i%CC%87le-i%CC%87kili-binary-resim-s%C4%B1n%C4%B1fland%C4%B1rma-69b15085f92c).
# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Training

```python neural_style_transfer.py ```

# Notebook

Download .ipynb file from [here](https://github.com/MuhammedBuyukkinaci/My-Jupyter-Files-1/blob/master/tensorflow_binary_image_classification2.ipynb) and run:

```jupyter lab ``` or ```jupyter notebook ```

# Data
No MNIST or CIFAR-10.

This is a repository containing datasets of 6400 training images and 1243 testing images.No problematic image.

Download .rar extension version from [here](
https://www.dropbox.com/s/ezmsiz0p364shxz/datasets.rar?dl=0) or .zip extension version from [here](
https://www.dropbox.com/s/cx6f238aoxjem6j/datasets_zip.zip?dl=0).
It is 101 MB.


# Architecture

1 input layer, 4 convolution layer, 4 pooling layer, 2 fully connected layer, 2 dropout layer, 1 output layer. The architecture used is below.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/MY_ARCHITECTURE.png) 

# Results
Accuracy score reached 90 percent on CV after 50 epochs.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/accuracy.png)

Cross entropy loss is plotted below.

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Image-Classification-Convolutional-Neural-Networks/blob/master/loss.png)

# Predictions

![alt text](https://github.com/MuhammedBuyukkinaci/TensorFlow-Binary-Image-Classification-using-CNN-s/blob/master/binary_preds.png)

