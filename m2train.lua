-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'm2Net'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local mapss = {3, 32, 64, 128, 256} -- layer maps sizes
local mapss2 = {32, 32, 64, 128, 256} -- layer maps sizes
local nlayers = 2 
local clOpt = {}
clOpt['nSeq'] = 1
clOpt['stride'] = 1
clOpt['dropOut'] = 0.5
-- create graph
print('Creating model:')
local model = mNet(nlayers,input_stride,poolsize,mapss,clOpt)
model:cuda()

-- test:
print('Testing model:')

-- local nT = 1 -- time sequence length
local inTable = {}
-- local outTable = {}
function updateR(L)
   local rnnI = {}
   for i = 1 , L do
      rnnI[i] = torch.Tensor(mapss[i+1],32,32):cuda()
   end
   return rnnI
end
rnnI = updateR(nlayers)
table.insert(inTable, torch.ones(mapss[1], 32, 32):cuda()) -- input Image
for L = 1, nlayers do
   table.insert(inTable, torch.ones(mapss[L+1], 32, 32):cuda()) -- prev E
end
--OutPut
print('inTable')
print(inTable)
outTable = model:forward(inTable)
print('outTable')
print({outTable})
