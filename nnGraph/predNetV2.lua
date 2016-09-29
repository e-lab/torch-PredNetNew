-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'image'
require 'model-m2NetV2'
local m = require 'opts'
-- option is always gloable
opt = m.parse(arg)
nSeq = 3
nlayers = opt.nlayers
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
   criterion, main = createModel(opt, channels, clOpt)
   print(inputTable[1]:size())
   -- one layer, not time dependency:

   -- test:
   local e,h,c,ht = {} ,{} ,{} ,{}
   -- Init state for top LSTM
   local initState = {}
   for i = nlayers , 1, -1 do
      initState[3*(i-1)+1] = torch.Tensor(prevE[i],imSize[i],imSize[i]):zero()
      initState[3*(i-1)+2] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
      initState[3*(i-1)+3] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
   end
   print('Test initState')
   print(initState)

   print(inputTable)
   print('Test model module')
   input = {inputTable, unpack(initState)}
   out = model:forward(input)
   print(out:size())

end
main()
