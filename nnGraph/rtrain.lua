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
local mapss2 = {3, 32, 64, 128, 256} -- layer maps sizes
local mapss = {32, 64, 128, 256 } --  Out put size of lstm
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
local clOpt2 = {}
clOpt2['nSeq'] = 1
clOpt2['kw'] = 3
clOpt2['kh'] = 3
clOpt2['st'] = 1
clOpt2['pa'] = 1
clOpt2['dropOut'] = 0
clOpt2['lm'] = 1

-- create graph
print('Creating model:')
local pn = {}
pn.main  = mNet(nlayers,input_stride,poolsize,mapss2,clOpt)
pn.lstm = {}
for i = nlayers, 1, -1 do
   pn.lstm[i] = lstm(mapss[i+1],mapss[i],clOpt)
end

shipGPU(pn)
-- test:
print('Testing model:')
x = {}
x[nlayers] = torch.Tensor(mapss[nlayers+1],32,32):zero():cuda()
-- Init state for top LSTM
local initState = {}
for i = nlayers , 1, -1 do
   local c = torch.Tensor(mapss[i],32,32):zero():cuda()
   local h = torch.Tensor(mapss[i],32,32):zero():cuda()
   local ht = {}
   for i =1, clOpt.lm do
      table.insert(ht,c:cuda())
      table.insert(ht,h:cuda())
   end
   if i == nlayers then
      initState[i] = {x[i],unpack(ht)}
   else
      initState[i] = {}
      initState[i][2] = c
      initState[i][3] = h
   end
end
print('InitState of lstm')
print(initState)

--Make model cuda

function updateLSTM(pn,initState)
   --Update LSTM topDown lstmOut[1] : cell lstmOut[2]: hidden
   local lstmOut = {}
   for i = nlayers, 1, -1 do
      print(i)
      lstmOut[i] = pn.lstm[i]:forward(initState[i])
      print(lstmOut)
      if i ~= 1 then
         x[i-1] = lstmOut[i][2]
         initState[i-1][1] = x[i-1]
         print('initState: '..tostring(i))
         print(initState[i-1])
      end
   end

   for i, out in ipairs(lstmOut) do
      print('OutPut of LSTM'..tostring(i))
      print(out)
   end
   return lstmOut
end

-- local nT = 1 -- time sequence length
local inTable = {}
-- local outTable = {}
function updateR(L, lstmOut)
   local rnnI = {}
   for i = 1 , L do
      --Iff lstm is layer 1
      rnnI[i] = lstmOut[i][2]
   end
   return rnnI
end

--Top Down update convLSTM
local lstmOut = updateLSTM(pn,initState)
--Extract hidden for input of main
rnnI = updateR(nlayers, lstmOut)
print('rnnI')
print(rnnI)
--Creat in put for main
table.insert(inTable, torch.ones(mapss2[1], 32, 32):cuda()) -- input Image
for L = 1, nlayers do
   table.insert(inTable, rnnI[L]) -- prev E
end

-- Update main down up
print('inTable')
print(inTable)
outTable = pn.main:forward(inTable)
print('outTable')
print({outTable})
