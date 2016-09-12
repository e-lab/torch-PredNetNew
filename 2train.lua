-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'mNetV2'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local mapss = {3, 32, 64, 128, 256} -- layer maps sizes
local nlayers = 1
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
function reactI(L)
   x = torch.Tensor(mapss[L+1],32,32)    
   c = torch.Tensor(mapss[L+1],32,32)    
   h = torch.Tensor(mapss[L+1],32,32)    
   ht = {}                      
   lstmLayer = 1
   for i =1, lstmLayer  do        
      table.insert(ht,h:cuda()) 
      table.insert(ht,c:cuda()) 
   end                          
   rnnI = {x:cuda(),unpack(ht)}    
   return rnnI
end
for L = 1, nlayers do
   table.insert(inTable, torch.ones(mapss[L], 64, 64):cuda()) -- prev E
   rnnI = reactI(L)
   table.insert(inTable, rnnI) -- this E
   if L < nlayers then table.insert(inTable, torch.zeros(mapss[L+1], 32*L, 32*L):cuda()) end -- next R
end
--OutPut
print(inTable)
outTable = model:forward(inTable)
model.name = 'myBad'
print('outTable')
print(outTable)
