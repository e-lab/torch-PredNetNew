-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
insize = 64
input_stride = 1
poolsize = 2
mapss = {3, 32, 64, 128, 256} -- layer maps sizes

layer={}
-- P = prediction branch, A_hat in paper

nlayers = 1

-- define all layers function:
for L = 1, nlayers do
end

local input = nn.Identity()()
local inputs = {}
local outputs = {}
table.insert(inputs, nn.Identity()()) -- input image x
for L = 1, nlayers do
   if L > 1 then table.insert(inputs, nn.Identity()()) end-- previous E
   table.insert(inputs, nn.Identity()()) -- next R
end

for L = 1, nlayers do
   print('Creating layer:', L)

   -- define layer functions:
   local cA = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   local cR = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent
   local cP = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   local mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
   -- local up = nn.SpatialUpSamplingNearest(poolsize)
   local op = nn.PReLU(mapss[L+1])

   if L == 1 then
      pE = inputs[1]
   else
      pE = inputs[2*L+1] -- previous E
   end
   pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   A = pE - cA - mA - nn.ReLU()
   
   nR = inputs[2*L] -- next R
   -- upR = {nR} - up
   if L == 1 then
      R = nR
   else
      R = {E, nR} - nn.CAddTable() - cR
   end
   P = {R} - cP - nn.ReLU()
   E = {A, P} - nn.CSubTable() - op -- PReLU instead of +/-ReLU
   E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
   table.insert(outputs, E)
   table.insert(outputs, R)
end
-- create graph
model = nn.gModule(inputs, outputs)
nngraph.annotateNodes()
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


-- test:
testx = {}
table.insert(testx, torch.Tensor(mapss[1], insize, insize)) -- input
for L = 1, nlayers do
   if L > 1 then table.insert(testx, torch.zeros(mapss[L], insize / poolsize, insize / poolsize)) end-- previous E
   table.insert(testx, torch.zeros(mapss[L+1], insize / poolsize, insize / poolsize)) -- next R
end
out = model:forward(testx)
   -- -- graph.dot(model.fg, 'MatchNet-model','Model') -- graph the model!
-- print('output size:', out:size())
print('output:', out)
