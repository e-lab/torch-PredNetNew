-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

-- require 'nn'
-- require 'nngraph'
require 'MatchNet'
local c = require 'trepl.colorize'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- model parameters:
local insize = 64
local nlayers = 1
local input_stride = 1
local poolsize = 2
local mapss = {1, 32, 64, 128, 256}
local clOpt = {}
clOpt['nSeq'] = 19
clOpt['stride'] = 1

-- instantiate MatchNet:
print('Creating model')
local model = mNet(nlayers, input_stride, poolsize, mapss, clOpt, true)
-- print({model})
-- print(model:parameters())

-- test:
print('Testing model')

local inTable = {}
for L = 1, nlayers do
   table.insert( inTable, torch.ones( mapss[L], insize/2^(L-1), insize/2^(L-1)) ) -- same layer E
   table.insert( inTable, torch.zeros( mapss[L], insize/2^(L-1), insize/2^(L-1)) ) -- previous time E
   table.insert( inTable, torch.zeros( mapss[L+1], insize/2^(L-1), insize/2^(L-1)) ) -- previous time R
   if L < nlayers then table.insert( inTable, torch.zeros(mapss[L+1], insize/2^(L), insize/2^(L)) ) end -- next R
end

local outTable = model:forward(inTable)
print('Output is: ')
print(outTable)
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
