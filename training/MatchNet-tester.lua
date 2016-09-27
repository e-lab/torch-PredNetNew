-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'MatchNet'
local c = require 'trepl.colorize'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- model parameters:
local opt = {}
opt.nlayers = 2
opt.inputSizeW = 64
opt.stride = 1
opt.poolsize = 2
opt.nFilters = {1, 32, 32, 32}
local clOpt = {}
clOpt['nSeq'] = 19
clOpt['stride'] = 1

-- instantiate MatchNet:
print('Creating model')
local model = mNet(opt.nlayers, opt.stride, opt.poolsize, opt.nFilters, clOpt, true)
-- print({model})
-- print(model:parameters())

-- test:
print('Testing model')
local inTable = {}
table.insert( inTable, torch.ones(opt.nFilters[1], opt.inputSizeW, opt.inputSizeW)) -- input
for L=1, opt.nlayers do
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
