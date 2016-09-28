-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'models.m2NetV2'
require 'models.convLSTM'
require 'image'
local m = require 'opts'

-- option is always gloable
opt = m.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

local function main()
   --cutorch.setDevice(1)
   --Test Mnist data
   paths.dofile('data-mnist.lua')
   datasetSeq = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   print('==> training model')
   torch.manualSeed(opt.seed)
   inputTable = {}
   target  = torch.Tensor()--= torch.Tensor(opt.transf,opt.memorySizeH, opt.memorySizeW)
   sample = datasetSeq[t]
   data = sample[1]
   for i = 1,data:size(1)-1 do
     table.insert(inputTable, data[i])
   end

   target:resizeAs(data[1]):copy(data[data:size(1)])

   if false then
     _im1_ = image.display{image={ inputTable[#inputTable-4]:squeeze(),
                                   inputTable[#inputTable-3]:squeeze(),
                                   inputTable[#inputTable-2]:squeeze(),
                                   inputTable[#inputTable-1]:squeeze(),
                                   inputTable[#inputTable]:squeeze(),
                                   target:squeeze(),
                                   target:squeeze() },
                           win = _im1_, nrow = 7, legend = 't-4, -3, -2, -2, t, Target, Output'}
   end
   print(inputTable[1]:size())
   -- one layer, not time dependency:
   -- Option for main
   local input_stride = 1
   local poolsize = 2
   local inputImsize = 64
   local nlayers = opt.nlayers
   local nSeq    = 1
   -- Option for lstm
   local clOpt = {}
   clOpt['nSeq'] = nSeq
   clOpt['kw'] = 3
   clOpt['kh'] = 3
   clOpt['st'] = 1
   clOpt['pa'] = 1
   clOpt['dropOut'] = 0
   clOpt['lm'] = 1

   --A anc Ah size
   local imSize   ={64,32}
   local channels = {1, 32} -- layer maps sizes
   local prevE  = {channels[1]*2,channels[2]*2}
   local cellCh = {32,prevE[1]} --  Out put size of lstm -- This is same as output channels
   local lstmCh = {cellCh[2]+prevE[1],prevE[2]} --  Out put size of lstm -- last chnel has no R_l+1 concat
   clOpt['cellCh'] = cellCh
   clOpt['lstmCh'] = lstmCh

   -- create graph
   print('Creating model:')
   main  = mNet(nlayers,input_stride,poolsize,channels,clOpt)
   dofile('utils.lua')
   local clones = clone_many_times(main, opt.nSeq)


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


   --Top Down update convLSTM
   --lstmState 1 layer LSTM {input, cell, output}
   local lstmState,rnnI,inT = {} , {} , {}
   local outTable = {}
   for i = 1, nSeq do
      print(i,' step processing')
      if i == 1 then
         state = initState
         --Test initState
      end
      -- local nT = 1 -- time sequence length
      local inTable = {}
      image = inputTable[i]
      table.insert(inTable, image) -- input Image
      for L = nlayers, 1, -1 do
         if i == 1 then
            inTable[3*(L-1)+2] = state[L][1] -- Prev E
            inTable[3*(L-1)+3] = state[L][2] -- prev Cell
            inTable[3*(L-1)+4] = state[L][3] -- prev Hidden
         else
            inTable[3*(L-1)+2] = outTable[3*(L-1)+2] -- Prev E
            inTable[3*(L-1)+3] = outTable[3*(L-1)+3] -- prev Cell
            inTable[3*(L-1)+4] = outTable[3*(L-1)+4] -- prev Hidden
         end
      end

      inT[i] = inTable

      -- Update main down up
      print('inTable')
      print(inT[i])
      outTable[i] = main:forward(inT[i])

      print('outTable: ',i)
      print(outTable[i])
   end
end
main()
