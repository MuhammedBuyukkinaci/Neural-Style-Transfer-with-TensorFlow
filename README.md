# Neural Style Transfer with TensorFlow
Implementation of Neural Style Transfer algorithm with TensorFlow in Python 3.

# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Training

```python neural_style_transfer.py ```

# Notebook

Download .ipynb file from [here](https://github.com/MuhammedBuyukkinaci/My-Jupyter-Files-1/blob/master/tensorflow_binary_image_classification2.ipynb) and run:

```jupyter lab ``` or ```jupyter notebook ```

# Outputs
Content Image             |  Style Image          |  Generated Image After 2000 iterations      |GIF during 2000 iterations             
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/content_image.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/style_image.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/generated_image_first.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/gif.gif" width="200" height="200">

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

