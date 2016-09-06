-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
local c = require 'trepl.colorize'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local mapss = {3, 32, 64, 128, 256} -- layer maps sizes

local layer={}
-- P = prediction branch, A_hat in paper

local nlayers = 1

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

for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   local cA = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   local cR = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent / convLSTM temp model
   local cP = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   local mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
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

-- create graph
print('Creating model:')
local model = nn.gModule(inputs, outputs)
nngraph.annotateNodes()
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


-- test:
print('Testing model:')

-- local nT = 1 -- time sequence length
local inTable = {}
-- local outTable = {}
for L = 1, nlayers do
   table.insert(inTable, torch.ones(mapss[L], insize/2^L, insize/2^L)) -- prev E
   table.insert(inTable, torch.zeros(mapss[L+1], insize/2^(L+1), insize/2^(L+1))) -- this E
   if L < nlayers then table.insert(inTable, torch.zeros(mapss[L+1], insize/2^(L+2), insize/2^(L+2))) end -- next R
end
outTable = model:forward(inTable)
print(outTable)
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
