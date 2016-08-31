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
A={} P={} cA={} mA={} cP={} cR={} E={} R={} up={}
-- P = prediction branch, A_hat in paper

nlayers = 2

-- sizes = {}
-- -- initialize size:
-- for L = 1, nlayers do
--    sizes[L] = {mapss[L], insize/(poolsize^(L-1)), insize/(poolsize^(L-1))} -- sizes of inputs at each layer
-- end

-- define all layers function:
for L = 1, nlayers do
   cA[L] = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
   cR[L] = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent
   cP[L] = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
   mA[L] = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
   up[L] = nn.SpatialUpSamplingNearest(poolsize)
end

-- input = nn.Identity()()
inputs = {}
outputs = {}
table.insert(inputs, nn.Identity()()) -- input image x
for L = 1, nlayers do
   -- table.insert(inputs, nn.Identity()()) -- prev E
   table.insert(inputs, nn.Identity()()) -- prev R
   -- P[L] = torch.zeros(mapss[L], insize/(poolsize^(L-1)), insize/(poolsize^(L-1)))
end

for L = 1, nlayers do
      if L == 1 then
         A[L] = inputs[1] - cA[L] - mA[L] - nn.ReLU()
      else
         A[L] = E[L-1] - cA[L] - mA[L] - nn.ReLU()
      end
      E[L] = {A[L], P[L]} - nn.CSubTable() - nn.PReLU(mapss[L+1]) -- PReLU instead of +/-ReLU
      -- table.insert(outputs, E[L])
end

for L = nlayers, 1 do
      if L == nlayers then
         R[L] = {E[L]} - cR[L]
      else
         upR = up[L] - R[L+1]
         R[L] = {E[L], upR} - nn.CAddTable(1) - cR[L]
      end
      P[L] = {R[L]} - cP[L] - nn.ReLU()
      -- table.insert(outputs, R[L])
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
