# MatchNet

A model of https://coxlab.github.io/prednet/

Run 'th predNet.lua' To run predNet model it works with 2 layers and nSeq

Models are in models folder

m2Net : Is main model for predNet

convLSTM : Convolution lstm for the predNet

In predNet.lua it do top down fist and update convLSTM and get output of lstm

After top down get out put of lstm. Do down up with m2Net.
