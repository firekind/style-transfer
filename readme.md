# Style transfer using pytorch

This repository contains an implementation of [Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). 

## What it is?
Style transfer is the process of transferring the "style" of a *style image* into a *content image*. For example, rendering a picture of a building the way Vincent Van Gogh would paint it.

<p align="center"><img src="https://github.com/firekind/style-transfer/raw/master/images/combo.png" width=600/></p>

## How it works?
Transferring the style involves "extracting" the style of the style image and merging it with the content of the content image. To obtain these, A network that is trained on object recognition is used.

Such networks are able to "extract" the objects of an image irrespective of that object's location in the image. The deeper one goes down the network, more the network cares about the actual object in the image rather than the pixel values of the image. Thus, deeper layers capture high level information of the image, and higher layers deal with pixel information.

Using this, the style and content of an image can be extracted. The information regarding the content of the image is present at the deeper convolutional layers of the network, and the information regarding the style of the network can be obtained by combining the outputs (filter responses) of all the convolutional layers in the network.

The process of transferring the style can be broken down as follows:

1. Feed the style image through the network trained on object recognition (feature extractor). Obtain the outputs (style representations) of all the convolutional layers of the network. These outputs are the targets the resultant image should achieve.

2. Feed the content image through the feature extractor. Obtain the output (content representation) of the last convolutional layer. This output is the target the resultant image should achieve.

3. During training, feed the content image through the model and obtain the style representations and content representation. 
    
    - Calculate the style loss using these style representations and the style targets computed previously.
    - Calculate the content loss using these content representations and the content targets computed previously.
    - Optimize the content image by minimizing the sum of the style and content loss.

The question remains on how to calculate the style and content losses. The content loss is calculated by a simple mean squared error between the content representation and the content target, since the differences between the two representations should be taken into account. Whereas, the style loss is calculated by a mean squared error between the gram matrix of the style representations and the gram matrix of the style targets.

To understand why a gram matrix is used to calculate the style loss, consider the example of a 32x32x256 feature map which is the output of a convolutional layer. This feature map has a width and height of 32 and has 256 channels. Assume that channel 0 of the feature map activates (has a high value) when the network detects a mountain, and channel 0 of the map activates when the network detects clouds. Thus, in an image with mountains and a lot of clouds, these two channels have high activations, and are related to each other.

Thus, to get the correlations of all the channels with each other, one could do the following:<br/>
<p align="center"><img src=https://github.com/firekind/style-transfer/raw/master/images/theory/gram_matrix.jpg height=350 /></p>

The first row of the gram matrix contains the values of the first channel multiplied with every other channel, the second row contains values of the second channel multiplied with every other channel and so on. Thus the gram matrix contains the correlations of each channel with every other channel.

Mathematically, given an input image *x*, target *y* and layer *l*,

<p align="center"><img src="https://github.com/firekind/style-transfer/raw/master/images/theory/contentloss.jpg" width=300/></p>
<br/>
<p align="center"><img src="https://github.com/firekind/style-transfer/raw/master/images/theory/styleloss.jpg" width=400/></p>
<br/>
<p align="center"><img src="https://github.com/firekind/style-transfer/raw/master/images/theory/totalloss.jpg" width=200/></p>

Where *a<sub>l</sub>* is a function to obtain the activation at layer *l*. and *g* is a function that computes the gram matrix.


## Results
Here are some results from the model implemented in this repo:

### Original
<p align="center"><img src=https://github.com/firekind/style-transfer/raw/master/images/results/forest.jpg height=350/></p>

### Result
Style from `images/picasso.jpg`.

<p align="center"><img src=https://github.com/firekind/style-transfer/raw/master/images/results/forest_out.jpg height=350/></p>

<br/>

### Original
<p align="center"><img src=https://github.com/firekind/style-transfer/raw/master/images/results/sunrise.jpg height=350 /></p>

### Result
Style from `images/style.jpg`.

<p align="center"><img src=https://github.com/firekind/style-transfer/raw/master/images/results/sunrise_out.jpg height=350 /></p>

## Setup
Clone the repository and `cd` into the directory. Create the environment using anaconda

```sh
$ conda env create -f environment.yml
```

Activate the environment using
```sh
$ conda activate style-transfer
```

To run the program from command line, run `main.py` and provide the necessary arguments (make sure the environment is activated first).

```sh
$ python main.py --help
usage: main.py [-h] --content-image-path CONTENT_IMAGE_PATH --style-image-path
               STYLE_IMAGE_PATH [--cuda] [--image-size IMAGE_SIZE]
               [--epochs EPOCHS] [--content-weight CONTENT_WEIGHT]
               [--style-weight STYLE_WEIGHT] [--content-layers CONTENT_LAYERS]
               [--style-layers STYLE_LAYERS] [--save-path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --content-image-path CONTENT_IMAGE_PATH
                        path to the content image.
  --style-image-path STYLE_IMAGE_PATH
                        path to the style image.
  --cuda                use cuda.
  --image-size IMAGE_SIZE
                        size to resize to.
  --epochs EPOCHS       number of epochs to train for.
  --content-weight CONTENT_WEIGHT
                        weight for the content loss.
  --style-weight STYLE_WEIGHT
                        weights for the style loss (if list, comma separated).
  --content-layers CONTENT_LAYERS
                        layers used to obtain content representation (comma
                        separated).
  --style-layers STYLE_LAYERS
                        layers used to obtain style representation (comma
                        separated).
  --save-path SAVE_PATH
                        path to save the output image (as JPEG).
```

To run the streamlit app, activate the conda environment first and then,

```sh
$ streamlit run app.py
```

Look at `notebooks/transfer_style.ipynb` notebook to find out how to run the code from within a python program.


## References

- [Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Neural Style Transfer Tutorial - Part 1](https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f)
- [Neural Style Transfer using Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
