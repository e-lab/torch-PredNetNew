-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'

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

-- creating input and output lists:
local input = nn.Identity()()
local inputs = {} -- inputs = {inputs, previousD, nextR}
local outputs = {} -- outputs = {E, R}, D == discriminator output, R == generator output
table.insert(inputs, nn.Identity()()) -- input image x
for L = 1, nlayers do
   -- {input, E, R, E, R, ...}; R index = 2*L+1; E index = 2*L
   table.insert(inputs, nn.Identity()()) -- previous E
   table.insert(inputs, nn.Identity()()) -- next R
   table.insert(outputs, nn.Identity()()) -- previous E
   table.insert(outputs, nn.Identity()()) -- next R
end

for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   local cA = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   local cR = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent / convLSTM temp model
   local cP = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   local mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
   local up = nn.SpatialUpSamplingNearest(poolsize)
   local op = nn.PReLU(mapss[L+1])

   local pE, A, nR, R, P, E

   if L == 1 then
      pE = inputs[1] -- model input (input image)
   else
      pE = outputs[2*L-1] -- previous layer E
   end
   pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   A = pE - cA - mA - nn.ReLU()
   
   if L == nlayer then
      R = E - cR
   else
      upR = outputs[2*L+2] - up -- upsampling of next layer R
      R = {E, upR} - nn.CAddTable(1) - cR
   end
   P = {R} - cP - nn.ReLU()
   E = {A, P} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
   E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
   table.insert(outputs, E) -- this layer E
   table.insert(outputs, R) -- this layer R
end
-- create graph
print('Creating model:')
local model = nn.gModule(inputs, outputs)
nngraph.annotateNodes()
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


-- test:
-- print('Testing model:')
