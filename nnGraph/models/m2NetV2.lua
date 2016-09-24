-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'models.convLSTM'
local c = require 'trepl.colorize'
backend = nn

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
   inputs[3*(L-1)+2] = nn.Identity()() -- previous E
   inputs[3*(L-1)+3] = nn.Identity()() -- previous Cell
   inputs[3*(L-1)+4] = nn.Identity()() -- previous Hidden
end

--Create instance of lstm
local convlstm = {}
for L = nlayers, 1, -1 do
   if L == nlayers then
      convlstm[L] = lstm(channels[L]*2,clOpt.cellCh[L],clOpt,false)
   else
      convlstm[L] = lstm(clOpt.lstmCh[L],clOpt.cellCh[L],clOpt,false)
   end
end

--Top Down
local outLstm = {}
for L = nlayers, 1 , -1 do
   if L == nlayers then
      outLstm[L] = {inputs[3*(L-1)+2],inputs[3*(L-1)+3],inputs[3*(L-1)+4]} - convlstm[L]
   else
      upR = outLstm[L+1] - nn.SelectTable(2) - up
      --Conv channels is 1 step forward since it starts from 1
      --Fill up input of LSTM channels
      inR = {upR,inputs[3*(L-1)+2]} - Jt
      outLstm[L] = {inR, inputs[3*(L-1)+3],inputs[3*(L-1)+4]} - convlstm[L]
   end
   outputs[3*(L-1)+2] = outLstm[L] - nn.SelectTable(1)
   outputs[3*(L-1)+3] = outLstm[L] - nn.SelectTable(2)
end
--Down Up
E = {}
local cA, Ah
local pE, A, upR
for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   if L == 1 then
      Ah = backend.SpatialConvolution(clOpt.cellCh[L], 1,3, 3, input_stride, input_stride, 1, 1) -- P convolution
   else
      Ah = backend.SpatialConvolution(clOpt.cellCh[L], channels[L], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   end


   if L == 1 then
      x = inputs[1]
   else
      --pE previous layer E
      pE = outputs[3*(L-2)+1]
      cA = backend.SpatialConvolution(clOpt.cellCh[L],channels[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
      A = pE - cA - Re - Mp
      pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   end
   --iR is already updated so we do second forloop
   if L == 1 then
      print('I am in Down Top ',L)
      iR = outLstm[L] - nn.SelectTable(2)
      iR:annotate{graphAttributes = {color = 'blue', fontcolor = 'green'}}
      local P = iR -Ah - Re
      EN = {x, P} - St  -- PReLU instead of +/-ReLU
      EP = {P, x} - St  -- PReLU instead of +/-ReLU
      outputs[3*(L-1)+1] = {EN, EP} - Jt - nn.Narrow(1,1,channels[L]*2) -- this layer E
   else
      iR = outLstm[L] - nn.SelectTable(2)
      iR:annotate{graphAttributes = {color = 'blue', fontcolor = 'green'}}
      local P = iR - Ah - Re
      EN = {A, P} - St  -- PReLU instead of +/-ReLU
      EN:annotate{graphAttributes = {color = 'red', fontcolor = 'green'}}
      EP = {P, A} - St  -- PReLU instead of +/-ReLU
      EP:annotate{graphAttributes = {color = 'red', fontcolor = 'blue'}}
      E[L]  = {EN, EP} - Jt
      E[L]:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
      --outputs[3*(L-1)+1] = E[L]-- this layer E
      outputs[3*(L-1)+1] = E[L]-- this layer E
   end
   -- set outputs:
end
return nn.gModule(inputs, outputs)

end
