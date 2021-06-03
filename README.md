# Lung-Segmentation-in-TensorFlow-2.0

This repository contains the code for semantic segmentation on the Lung Segmentation dataset using TensorFlow 2.0 framework.
The following models are used:
- [UNET](https://arxiv.org/abs/1505.04597)

Models to be used in future:
- RESUNET
- DEEPLABV3+
- more...

# Dataset
The ISIC-2018 dataset is used for this for training the UNET architecture. The dataset contains the 2596 pairs of images and masks. All of these images are of different shapes and contains a variety of skin lesions.

Original Image             |  Mask Image
:-------------------------:|:-------------------------:
![](img/image.png)  |  ![](img/mask.png)
