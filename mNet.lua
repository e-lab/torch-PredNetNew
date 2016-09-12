-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'models.ConvLSTM'
require 'models.UntiedConvLSTM'
local c = require 'trepl.colorize'
require 'cudnn'
backend = cudnn

function mNet(nlayers,input_stride,poolsize,mapss,clOpt)
local layer={}
-- P = prediction branch, A_hat in paper
-- This module creates the MatchNet network model, defined as:
-- inputs = {prevE, thisE, nextR}
-- outputs = {E , R}, E == discriminator output, R == generator output

-- creating input and output lists:
local inputs = {}
local outputs = {}
for L = 1, nlayers do
   inputs[3*L-2] = nn.Identity()() -- previous E
   inputs[3*L-1] = nn.Identity()() -- this E
   if L < nlayers then inputs[3*L] = nn.Identity()() end -- next R
end
local nSeq = clOpt.nSeq
local clStride= clOpt.stride
for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   local cA = backend.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   --local cR = backend.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent / convLSTM temp model
   local cR = nn.UntiedConvLSTM(mapss[L+1], mapss[L+1], nSeq, 3, 3, clStride)
   local cP = backend.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   local mA = backend.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
   local up = nn.SpatialUpSamplingNearest(poolsize)
   local op = nn.PReLU(mapss[L+1])

   local pE, A, upR, R, P, E

   pE = inputs[3*L-2] -- previous layer E
   pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   A = pE - cA - mA - nn.ReLU()

   if L == nlayers then
      R = inputs[3*L-1] - cR -- this E = inputs[3*L-1] in this layer!
   else
      upR = inputs[3*L] - up -- upsampling of next layer R
      R = {inputs[3*L-1], upR} - nn.CAddTable(1) - cR -- this E = inputs[3*L-1] in this layer!
   end

   P = {R} - cP - nn.ReLU()
   E = {A, P} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
   E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
   -- set outputs:
   outputs[2*L-1] = E -- this layer E
   outputs[2*L] = R -- this layer R
end

return nn.gModule(inputs, outputs)

end
