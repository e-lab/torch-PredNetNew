-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'mNet'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local mapss = {3, 32, 64, 128, 256} -- layer maps sizes
local nlayers = 1

-- create graph
print('Creating model:')
local model = mNet(nlayers,input_stride,poolsize,mapss)


-- test:
print('Testing model:')

-- local nT = 1 -- time sequence length
local inTable = {}
-- local outTable = {}
for L = 1, nlayers do
   table.insert(inTable, torch.ones(mapss[L], 64, 64)) -- prev E
   table.insert(inTable, torch.zeros(mapss[L+1], 32, 32)) -- this E
   if L < nlayers then table.insert(inTable, torch.zeros(mapss[L+1], 32, 32)) end -- next R
end
--OutPut
outTable = model:forward(inTable)
print(outTable)
