-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'models.convLSTM'
local c = require 'trepl.colorize'
require 'cudnn'
backend = cudnn

function mNet(nlayers,input_stride,poolsize,channels,clOpt)
local layer={}
-- P = prediction branch, A_hat in paper
-- This module creates the MatchNet network model, defined as:
-- inputs = {prevE, thisE, nextR}
-- outputs = {E , R}, E == discriminator output, R == generator output
local Mp = backend.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
local up = nn.SpatialUpSamplingNearest(poolsize)
local Re = nn.ReLU()
local St = nn.CSubTable(1)
local Jt = nn.JoinTable(1)

-- creating input and output lists:
local inputs = {}
local outputs = {}
--This is because No Err in the first Layer
inputs[1] = nn.Identity()() -- x input
for L = 1, nlayers do
   inputs[L+1] = nn.Identity()() -- previous E
   inputs[L+2] = nn.Identity()() -- previous Cell
   inputs[L+3] = nn.Identity()() -- previous Hidden
end

--Create instance of lstm
local convlstm = {}
for i = nlayers, 1, -1 do
   if 1 == 1 then
      convlstm[i] = lstm(clOpt.lstmCh[i],clOpt.cellCh[i],clOpt,false)
   else
      convlstm[i] = lstm(clOpt.lstmCh[i],clOpt.cellCh[i],clOpt,clOpt.upSize)
   end
end

local outLstm = {}
for L = nlayers, 1 , -1 do
   if L == nlayers then
      outLstm[L] = {inputs[L+1],inputs[L+2],inputs[L+3]} - convlstm[L]
   else
      upR = outLstm[L+1] - nn.SelectTable(2)
      --Conv channels is 1 step forward since it starts from 1
      --Fill up input of LSTM channels
      inR = {upR,inputs[L+1]}- Jt
      outLstm[L] = {inR, inputs[L+2],inputs[L+3]} - convlstm[L]
   end
   outputs[3*(L-1)+2] = outLstm[L] - nn.SelectTable(1)
   outputs[3*(L-1)+3] = outLstm[L] - nn.SelectTable(2)
end

for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   local cA, cP
   if L == 1 then
      cA = backend.SpatialConvolution(channels[L], channels[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
      cP = backend.SpatialConvolution(channels[L], channels[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   else
      cA = backend.SpatialConvolution(channels[L+1], channels[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
      cP = backend.SpatialConvolution(channels[L+1], channels[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   end

   local pE, A, upR, P, E

   if L == 1 then
      pE = inputs[1]
   else
      --pE previous layer E
      pE = outputs[L-1]
   end
   A = pE - cA - Re - Mp
   pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   --iR is already updated so we do second forloop
   iR = outLstm[L] - nn.SelectTable(2)
   iR:annotate{graphAttributes = {color = 'blue', fontcolor = 'green'}}
   P = iR - cP - Re
   EN = {A, P} - St  -- PReLU instead of +/-ReLU
   EP = {P, A} - St  -- PReLU instead of +/-ReLU
   E  = {EN, EP} - Jt
   E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
   -- set outputs:
   outputs[3*(L-1)+1] = E -- this layer E
end
return nn.gModule(inputs, outputs)

end
