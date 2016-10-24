# PredNet

This repository is a `Torch` implementation of the paper [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104).
It needs MNIST dataset for training and saves the trained model in location specified by `-save` option.

A example instruction for loading the dataset and visualizing the output, alongwith saving the `graphs` of model is given below.

```
qlua main.lua --datapath /media/HDD1/Datasets2/originalDatasets/MNIST/ --vis --save /media/HDD1/Models/predNet/ --disp --dev cuda
```
