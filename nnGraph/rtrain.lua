-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
require 'models.m2Net'
require 'models.convLSTM'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

--make model cuda
function shipGPU(pn)
   pn.main:cuda()
   for _, m in pairs(pn.lstm)do
      m = m:cuda()
   end
end

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local channels = {1, 32, 64} -- layer maps sizes
local lstmCh = {96, 128} --  Out put size of lstm -- last chnel has no R_l+1 concat
local cellCh = {32, 64} --  Out put size of lstm -- This is same as output channels
local nlayers = 2

-- Option for lstm
local clOpt = {}
clOpt['nSeq'] = 1
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = 1
clOpt['pa'] = 1
clOpt['dropOut'] = 0
clOpt['lm'] = 1

-- create graph
print('Creating model:')
local pn = {}
pn.main  = mNet(nlayers,input_stride,poolsize,channels,clOpt)
pn.lstm = {}
for i = nlayers, 1, -1 do
   pn.lstm[i] = lstm(lstmCh[i],cellCh[i],clOpt)
end

shipGPU(pn)
-- test:
print('Testing model:')
local x,h,c,ht = {},{},{},{}
-- Init state for top LSTM
local initState = {}
for i = nlayers , 1, -1 do
   if i == nlayers then
      x[i] = torch.Tensor(lstmCh[i],32,32):zero():cuda()
   else
      -- Need to concat Err so leaving space
      x[i] = torch.Tensor(cellCh[i+1],32,32):zero():cuda()
   end
   c[i] = torch.Tensor(cellCh[i],32,32):zero()
   h[i] = c[i]:clone()
   ht[i] = {}
   --Here only lstm layer 1
   table.insert(ht[i],c[i]:cuda())
   table.insert(ht[i],h[i]:cuda())
end
for i = nlayers, 1, -1 do
   initState[i] = {x[i],unpack(ht[i])}
end
print('InitState of lstm')
print(initState)

--Make model cuda
local lstmState = {}
lstmState[1] = initState
function updateLSTM(pn,lstmState)
   --Update LSTM topDown lstmOut[1] : cell lstmOut[2]: hidden
   --Top down
   local lstmOut, E = {}, {}
   for i = nlayers, 1, -1 do
      if i == nlayers then
         lstmOut[i] = pn.lstm[i]:forward(lstmState[1][i])
      else
         upR = lstmOut[i+1][2]
         --Conv channels is 1 step forward since it starts from 1
         E[i] = torch.zeros(channels[i+1],32,32):cuda()
         --Fill up input of LSTM channels
         lstmState[1][i][1] = torch.cat(upR,E[i],1)
      end
   end
   for i, out in ipairs(lstmState) do
      print('OutPut of LSTM'..tostring(i))
      print(out)
   end
   return lstmState
end

-- local nT = 1 -- time sequence length
local inTable = {}
-- local outTable = {}
function updateR(L, lstmState)
   local rnnI = {}
   for i = 1 , L do
      --Iff lstm is layer 1
      rnnI[i] = lstmState[1][i][2]
   end
   return rnnI
end

--Top Down update convLSTM
local lstmState = updateLSTM(pn,lstmState)
--Extract hidden for input of main
rnnI = updateR(nlayers, lstmState)
print('rnnI')
print(rnnI)
--Creat in put for main
table.insert(inTable, torch.ones(channels[1], 32, 32):cuda()) -- input Image
for L = 1, nlayers do
   table.insert(inTable, rnnI[L]) -- prev E
end

-- Update main down up
print('inTable')
print(inTable)
outTable = pn.main:forward(inTable)
print('outTable')
print({outTable})
