# MatchNet

This repository is a `Torch` implementation of the paper [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1605.08104).
We are looking forward to further improve this network.

Folder description:

1. [train](train): Training the model
3. [utils](utils): Contains utility script such as one for preparing dataset for training
2. [visualize](visualize): Visualize the predicted output alongwith different states of the saved model

First generate dataset compatible for this repository using [utils](utils), then [train](train) the network, and then use [visualization](visualize) tool to see the trained network's representation.
