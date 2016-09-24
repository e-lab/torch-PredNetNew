# MatchNet

A model of https://coxlab.github.io/prednet/

There are 2 version V2 is lstm integrated in one main nnGraph model

the other one is convLSTM and main models are speparated.

Run 'th predNet.lua' To run predNet model it works with 2 layers and nSeq

Models are in models folder

m2Net : Is main model for predNet

convLSTM : Convolution lstm for the predNet

In predNet.lua it do top down fist and update convLSTM and get output of lstm

After top down get out put of lstm. Do down up with m2Net.
