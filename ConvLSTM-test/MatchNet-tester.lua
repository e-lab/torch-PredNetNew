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
local nlayers = 2
local insize = 64
local input_stride = 1
local poolsize = 2
local mapss = {1, 32, 32, 32}
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

-- 1st layer:
table.insert( inTable, torch.ones(1,64,64)) -- same layer E / input
table.insert( inTable, torch.zeros(1,64,64)) -- previous time E
table.insert( inTable, torch.zeros(32,64,64)) -- previous time R
if nlayers == 2 then
-- 2nd layer
table.insert( inTable, torch.zeros(32,32,32)) -- previous time E
table.insert( inTable, torch.zeros(32,32,32))  -- previous time R
end

local outTable = model:forward(inTable)
print('Output is: ')
print(outTable)
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
