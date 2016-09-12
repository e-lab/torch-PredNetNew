# MatchNet

A model of https://coxlab.github.io/prednet/

There are 3 cases 

mNetV2: Using convLSTM from nnGraph run with 2train.lua

mNet  : Using ConvLSTM from Elementary run with train.lua 

m2Net : Separate R and others run with m2train.lua 

Since we need to update R state firest with previous t-1 sequence 

And update other modules with t sequence with R's out put.

It's neccessary to separate RNN and Normal nnGraph model. 
