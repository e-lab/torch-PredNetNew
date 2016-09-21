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
-- Option for lstm
local clOpt = {}
nSeq = 2
clOpt['nSeq'] = nSeq
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = 1
clOpt['pa'] = 1
clOpt['dropOut'] = 0
clOpt['lm'] = 1
clOpt['upSize'] = 2

-- one layer, not time dependency:
local insize = 64
local input_stride = 1
local poolsize = 2
local inputImsize = 32
local imSize   ={16,8}
local channels = {1, 32, 64} -- layer maps sizes
local cellCh = {channels[1], channels[3]} --  Out put size of lstm -- This is same as output channels
local lstmCh = {channels[2]*2+cellCh[2], channels[3]*2} --  Out put size of lstm -- last chnel has no R_l+1 concat
local nlayers = 2

-- create graph
print('Creating model:')
local pn = {}
pn.main  = mNet(nlayers,input_stride,poolsize,channels,clOpt)
pn.lstm = {}
for i = nlayers, 1, -1 do
   if i == 1 then
      pn.lstm[i] = lstm(lstmCh[i],cellCh[i],clOpt,false)
   else
      pn.lstm[i] = lstm(lstmCh[i],cellCh[i],clOpt,clOpt.upSize)
   end
end

shipGPU(pn)
-- test:
print('Testing model:')
local x,h,c,ht = {},{},{},{}
-- Init state for top LSTM
local initState = {}
for i = nlayers , 1, -1 do
   if i == nlayers then
      x[i] = torch.Tensor(lstmCh[i],imSize[i],imSize[i]):zero():cuda()
   else
      -- Need to concat Err so leaving space
      x[i] = torch.Tensor(cellCh[i+1],imSize[i],imSize[i]):zero():cuda()
   end
   c[i] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
   h[i] = c[i]:clone()
   ht[i] = {}
   --Here only lstm layer 1
   table.insert(ht[i],c[i]:cuda())
   table.insert(ht[i],h[i]:cuda())
end
for i = nlayers, 1, -1 do
   if i == nlayers then
      initState[i] = {x[i],unpack(ht[i])}
   else
      initState[i] = {[2] = c[i]:cuda() , [3] = h[i]:cuda()}
   end
end
print('InitState of lstm')
print(initState)

--Make model cuda
function updateLSTM(pn,states ,seq,E)
   --Update LSTM topDown lstmOut[1] : cell lstmOut[2]: hidden
   --Top down
   local lstmOut = {}
   local E = E or {}
   for i = nlayers, 1, -1 do
      if i == nlayers then
         lstmOut[i] = pn.lstm[i]:forward(states[i])
         print('Test lstmOut'..tostring(i))
         print(lstmOut[i])
      else
         if seq == 1 then
            E[i] = torch.zeros(channels[i+2],imSize[i],imSize[i]):cuda()
         end
         upR = lstmOut[i+1][2]
         --Conv channels is 1 step forward since it starts from 1
         --Fill up input of LSTM channels
         states[i][1] = torch.cat(upR,E[i],1)
         lstmOut[i] = pn.lstm[i]:forward(states[i])
         print('Test lstmOut'..tostring(i))
         print(lstmOut[i])

         --test = pn.lstm[i]:forward(states[i])
      end
   end
   --[[
   for i, out in ipairs(states) do
      print('OutPut of LSTM'..tostring(i))
      print(out)
   end
   --]]
   return states
end

-- local outTable = {}
function updateR(L, states)
   local rnnI = {}
   for i = 1 , L do
      --Iff lstm is layer 1
      rnnI[i] = states[i][2]
   end
   return rnnI
end

--Top Down update convLSTM
--lstmState 1 layer LSTM {input, cell, output}
local lstmState,rnnI,inT = {} , {} , {}
for i = 1, nSeq do
   if i == 1 then
      lstmState[0] = initState
      lstmState[i] = updateLSTM(pn,lstmState[i-1],i)
   else
      --LSTM sould be previous t-1
      lstmState[i] = updateLSTM(pn,lstmState[i-1],i, outTable)
   end
   print('LSTM sateSequences')
   print(lstmState)
   --Extract hidden for input of main
   rnnI[i] = updateR(nlayers, lstmState[i])
   print('rnnI')
   print(rnnI)
   --Creat in put for main
   -- local nT = 1 -- time sequence length
   local inTable = {}
   table.insert(inTable, torch.ones(channels[1], inputImsize, inputImsize):cuda()) -- input Image
   for L = 1, nlayers do
      table.insert(inTable, rnnI[i][L]) -- prev E
   end

   inT[i] = inTable

   -- Update main down up
   print('inTable')
   print(inT)
   outTable = pn.main:forward(inT[i])
   print('outTable')
   print(outTable)
end
