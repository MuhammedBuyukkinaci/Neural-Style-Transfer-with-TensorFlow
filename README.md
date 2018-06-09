# Neural Style Transfer with TensorFlow
Implementation of Neural Style Transfer algorithm with pretrained VGG-16 Network & TensorFlow in Python 3.

# Dependencies

```pip install -r requirements.txt```

or

```pip3 install -r requirements.txt```

# Pre-trained VGG-16 Network

Download VGG-16 (491 MB) named as ```imagenet-vgg-verydeep-16```  from [here](http://www.vlfeat.org/matconvnet/pretrained/#imagenet-ilsvrc-classification) and put it in Neural-Style-Transfer-with-TensorFlow folder.

![alt text](https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow/blob/master/pictures/vgg16_where.png) 



# Cloning and Training

```
git clone https://github.com/MuhammedBuyukkinaci/Neural-Style-Transfer-with-TensorFlow.git

cd ./Neural-Style-Transfer-with-TensorFlow

python neural_style_transfer.py
```

To train with your parameters, run a command similar to below command(for more details, check out Arguments for Terminal section ):

```python neural_style_transfer.py -c_i changed_content.jpg -s_i changed_style.jpg -c_w 25 -s_w 25 -n_i 1000 ```

# Arguments for Terminal
- c_i: Name of Content image. It must be string. Example: changed_content.jpg, default_content.jpg , content_image.png .

- s_i: Name of Style image. It must be string. Example: changed_style.jpg , default_style.jpg , style_image.png .

- c_w: Weight of Content image in loss function. It must be integer. Example: 5 , 20 , 50 .

- s_w: Weight of Style image in loss function. It must be integer. Example: 10 , 20 , 40 .

- s_w: How many iterations to train. It must be integer. Example: 500 , 1000 , 2000
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
