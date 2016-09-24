-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'models.m2NetV2'
require 'models.convLSTM'
local m = require 'opts'

-- option is always gloable
opt = m.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

--make model cuda
function shipGPU(pn)
   pn.main:cuda()
   for _, m in pairs(pn.lstm)do
      m = m:cuda()
   end
end
-- Option for lstm
local clOpt = {}
clOpt['nSeq'] = nSeq
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = 1
clOpt['pa'] = 1
clOpt['dropOut'] = 0
clOpt['lm'] = 1
clOpt['upSize'] = 2

-- one layer, not time dependency:
-- Option for main
local input_stride = 1
local poolsize = 2
local inputImsize = 32
local nlayers = opt.nlayers
local nSeq    = opt.nSeq

--A anc Ah size
local imSize   ={32,16}
local channels = {1, 32} -- layer maps sizes
local prevE  = {channels[1]*2,channels[2]*2}
local cellCh = {32,prevE[1]} --  Out put size of lstm -- This is same as output channels
local lstmCh = {cellCh[2]+prevE[1],prevE[2]} --  Out put size of lstm -- last chnel has no R_l+1 concat
clOpt['cellCh'] = cellCh
clOpt['lstmCh'] = lstmCh


-- create graph
print('Creating model:')
main  = mNet(nlayers,input_stride,poolsize,channels,clOpt)
print(main)
-- test:
print('Testing model:')
local x,h,c,ht = {},{},{},{}
-- Init state for top LSTM
local initState = {}
for i = nlayers , 1, -1 do
   x[i] = torch.Tensor(prevE[i],imSize[i],imSize[i]):zero()
   c[i] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
   h[i] = c[i]:clone()
   ht[i] = {}
   --Here only lstm layer 1
   table.insert(ht[i],c[i])
   table.insert(ht[i],h[i])
end
for i = nlayers, 1, -1 do
   initState[i] = {x[i],unpack(ht[i])}
end
--Test initState
print('InitState of lstm')
print(initState)


--Top Down update convLSTM
--lstmState 1 layer LSTM {input, cell, output}
local lstmState,rnnI,inT = {} , {} , {}
for i = 1, nSeq do
   print(i,' step processing')
   if i == 1 then
      lstmState[0] = initState
   else
      lstmState[i-1] = initState
   end
   -- local nT = 1 -- time sequence length
   local inTable = {}
   table.insert(inTable, torch.ones(channels[1], inputImsize, inputImsize)) -- input Image
   state = lstmState[i-1]
   for L = nlayers, 1, -1 do
      inTable[3*(L-1)+2] = state[L][1] -- Prev E
      inTable[3*(L-1)+3] = state[L][2] -- prev Cell
      inTable[3*(L-1)+4] = state[L][3] -- prev Hidden
   end

   inT[i] = inTable

   -- Update main down up
   print('inTable')
   print(inT[i])
   outTable = main:forward(inT[i])
   print('outTable')
   print(outTable)
end
