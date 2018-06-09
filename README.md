# Neural Style Transfer with TensorFlow
Implementation of Neural Style Transfer algorithm with pretrained VGG-16 Network & TensorFlow in Python 3.

# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Pre-trained VGG-16 Network

You should download VGG-16 (491 MB) named as ```imagenet-vgg-verydeep-16```  from [this link](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification) and put it in cloned folder.

![alt text](https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/vgg16_where.png) 

# Cloning and Training on Terminal
```git clone https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow.git```

```cd ./Neural-Style-Transfer-with-TensorFlow```

To train with default content image, default style image and other default parameters, run:
```python neural_style_transfer.py ```

To train with your content image, your style image and other parameters, run a command similar to below command:

```python neural_style_transfer.py --content_image changed_content.jpg --style_image changed_style.jpg --content_weight 25 --style_weight 25 --number_iterations 1000 ```


# Notebook

Download .ipynb file from [here](https://github.com/MuhammedBuyukkinaci/My-Jupyter-Files/blob/master/neural_style_transfer.ipynb) and run:

```jupyter lab ``` or ```jupyter notebook ```

# Outputs

Content Image             |  Style Image          |  Generated Image After 2000 iterations      |GIF during 2000 iterations             
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/content_images/content_image.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/style_images/style_image.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/generated_images/generated_image.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/gifs/gif.gif" width="200" height="200">


Content Image             |  Style Image          |  Generated Image After 2000 iterations      |GIF during 2000 iterations             
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/content_images/content_image1.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/style_images/style_image1.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/generated_images/generated_image1.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/gifs/gif1.gif" width="200" height="200">


Content Image             |  Style Image          |  Generated Image After 2000 iterations      |GIF during 2000 iterations             
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/content_images/content_image2.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/style_images/style_image2.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/generated_images/generated_image2.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/gifs/gif2.gif" width="200" height="200">


Content Image             |  Style Image          |  Generated Image After 2000 iterations      |GIF during 2000 iterations             
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/content_images/content_image3.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/style_images/style_image3.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/generated_images/generated_image3.jpg" width="200" height="200">  | <img src="https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/gifs/gif3.gif" width="200" height="200">


The last content image belongs to one of my friends, [Umut Başdemir](https://www.linkedin.com/in/umutbasdemir/). He is a Business Development Specialist at Samsung.

# References

- https://www.coursera.org/specializations/deep-learning

- https://www.youtube.com/watch?v=Re2C9INXCNc&index=38&list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud

- https://gist.github.com/prajjwal1/8bc39c430c8a0303c1430b02207b09f4

- https://github.com/harsha1601/StyleTransfer/blob/master/cnn-st.ipynb

- https://github.com/nestor94/apptist/blob/master/model/style_transfer_model.py.ipynb
