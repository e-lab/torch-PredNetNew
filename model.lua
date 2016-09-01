-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
-- require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
insize = 64
input_stride = 1
poolsize = 2
mapss = {3, 32, 64, 128, 256} -- layer maps sizes

layer={}
-- P = prediction branch, A_hat in paper

nlayers = 2

-- sizes = {}
-- -- initialize size:
-- for L = 1, nlayers do
--    sizes[L] = {mapss[L], insize/(poolsize^(L-1)), insize/(poolsize^(L-1))} -- sizes of inputs at each layer
-- end

-- define all layers function:
for L = 1, nlayers do
   cA = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   cR = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent
   cP = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
   up = nn.SpatialUpSamplingNearest(poolsize)
end

input = nn.Identity()()
inputs = {}
outputs = {}
table.insert(inputs, nn.Identity()()) -- input image x
for L = 1, nlayers do
   table.insert(inputs, nn.Identity()()) -- previous E
   table.insert(inputs, nn.Identity()()) -- next R
end

for L = 0, nlayers do
   print('Creating layer:', L)
   if L == 0 then
      pE = inputs[1]
   else
      pE = inputs[2*L] -- previous E
   end
   pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
   A = pE - cA - mA - nn.ReLU()
   
   if L >= 1 then
      nR = inputs[2*L+1] -- next R
      upR = {nR} - up
      R = {E, upR} - nn.CAddTable(1) - cR
      P = {R} - cP - nn.ReLU()
   end
   E = {A, P} - nn.CSubTable() - nn.PReLU(mapss[L+1]) -- PReLU instead of +/-ReLU
   E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
   table.insert(outputs, E)
   table.insert(outputs, R)
end

model = nn.gModule(inputs, outputs)
nngraph.annotateNodes()
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!

-- test:
for L = 1, nlayers do
   -- x = torch.Tensor(mapss[L], insize, insize) -- input
   -- y = torch.zeros(mapss[L+1], insize / poolsize, insize / poolsize) -- Rn and Ef
   -- out = layer[L]:forward({x,y,y,y})
   -- -- graph.dot(model.fg, 'MatchNet-model','Model') -- graph the model!
   -- print('output size:', out:size())
   -- insize = insize/2
end
