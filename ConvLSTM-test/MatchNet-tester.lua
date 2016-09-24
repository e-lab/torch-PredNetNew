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
local opt = {}
local nlayers = 2
local opt.inputSizeW = 64
local input_stride = 1
local poolsize = 2
local opt.nFilters = {1, 32, 32, 32}
local clOpt = {}
clOpt['nSeq'] = 19
clOpt['stride'] = 1

-- instantiate MatchNet:
print('Creating model')
local model = mNet(nlayers, input_stride, poolsize, opt.nFilters, clOpt, true)
-- print({model})
-- print(model:parameters())

-- test:
print('Testing model')
local inTable = {}
table.insert( inTable, torch.ones(mapss[1], insize, insize)) -- input
for L=1, nlayers do
   table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)))-- previous time E
   if L==1 then 
      table.insert( inTable, torch.zeros(opt.nFilters[L+1], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
   else
      table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
   end
-- here are immediate values as a reminder:
-- table.insert( inTable, torch.ones(1,64,64)) -- input
-- table.insert( inTable, torch.zeros(1,64,64)) -- previous time E
-- table.insert( inTable, torch.zeros(32,64,64)) -- previous time R
-- if nlayers == 2 then
-- 2nd layer
-- table.insert( inTable, torch.zeros(32,32,32)) -- previous time E
-- table.insert( inTable, torch.zeros(32,32,32))  -- previous time R
end

local outTable = model:forward(inTable)
print('Output is: ')
print(outTable)
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
