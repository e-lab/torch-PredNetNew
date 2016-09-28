-- MatchNet training: predicting future frames in video
-- Eugenio Culurciello, August - September 2016
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
--

require 'nn'
require 'paths'
require 'torch'
require 'image'
require 'optim'
require 'env'
require 'pl'

-- lapp = require 'pl.lapp'
opt = lapp [[
  Command line options:
  --dir    (default results)  subdirectory to save experiments in
  --seed   (default 1250)     initial random seed
  --useGPU (default false)    use GPU in training

  Training parameters:
  -r,--learningRate       (default 1e-3)        learning rate
  -d,--learningRateDecay  (default 1e-7)        learning rate decay (in # samples)
  -w,--weightDecay        (default 5e-4)        L2 penalty on the weights
  -m,--momentum           (default 0.9)         momentum

  Model parameters:
  --nlayers       (default 2)     number of layers of MatchNet
  --inputSizeW    (default 64)    width of each input patch or image
  --inputSizeH    (default 64)    width of each input patch or image
  --maxIter       (default 30000) max number of updates
  
  --nSeq          (default 19)    input video sequence lenght
  --nFilters      (default {1,32,32,32})  number of filters in the encoding/decoding layers
  --stride        (default 1)     stride in convolutions
  --poolsize      (default 2)     maxpooling size

  --dataFile      (default 'data-small-train.t7')
  --dataFileTest  (default 'data-small-test.t7')
  --statInterval  (default 50)    interval for printing error
  -v              (default false) verbose output
  --display       (default true)  display stuff
  --displayInterval (default 50)
  -s,--save      (default true)   save models
  --saveInterval (default 10000)
]]

opt.nFilters  = {1,32,32,32} -- number of filters in the encoding/decoding layers

torch.setdefaulttensortype('torch.FloatTensor') 
torch.manualSeed(opt.seed)

opt.useGPU = false
print('Using GPU?', opt.useGPU)

if opt.useGPU then
  require 'cunn'
  require 'cutorch'
end

local function main()
  local w, dE_dw

  -- cutorch.setDevice(1)
  paths.dofile('data-mnist.lua')
  paths.dofile('model-matchnet.lua')
  -- print('This is the model:', {model})

  datasetSeq = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames

  print  ('Loaded ' .. datasetSeq:size() .. ' images')

  print('==> training model')

  w, dE_dw = model:getParameters()
  print('Number of parameters ' .. w:nElement())
  print('Number of grads ' .. dE_dw:nElement())

  local eta = opt.learningRate
  local err = 0
  local epoch = 0
 
  local optimState = {
    learningRate = opt.eta,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
  }
  
  model:training()

  -- train:
  for t = 1,opt.maxIter do

    -- define eval closure
    local eval_E = function(w)
      local f = 0
 
      model:zeroGradParameters()

      -- setup initial variables:
      local inTableG0 = {} -- global inputs reset
      for L=1, opt.nlayers do
         table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)))-- previous time E
         if L==1 then 
            table.insert( inTableG0, torch.zeros(opt.nFilters[L+1], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
         else
            table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
         end
      end
      inputTable = {}
      
      target  = torch.Tensor()
      sample = datasetSeq[t]
      data = sample[1]
      for i = 1,data:size(1)-1 do
        table.insert(inputTable, data[i])
      end

      target:resizeAs(data[1]):copy(data[data:size(1)])
      if opt.useGPU then target = target:cuda() end
      
      -- estimate f and gradients
      table.insert(inTableG0, inputTable)
      output = model:forward(inTableG0)
      f = f + criterion:forward(output, target)
      local dE_dy = criterion:backward(output, target)
      model:backward(inTableG0,dE_dy)
      dE_dw:add(opt.weightDecay, w)

      -- return f and df/dX
      return f, dE_dw
    end
   
    if math.fmod(t,20000) == 0 then
      epoch = epoch + 1
      eta = opt.learningRate*math.pow(0.5,epoch/50)  
      optimState.learningRate = eta  
    end
    
    _,fs = optim.adam(eval_E, w, optimState)

    err = err + fs[1]

    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.nSeq) == 1 then
      print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..eta )
      err = 0
      if opt.save and math.fmod(t, opt.nSeq*1000) == 1 and t>1 then       
        torch.save(opt.dir .. '/model_' .. t .. '.net', model)
        torch.save(opt.dir .. '/optimState_' .. t .. '.t7', optimState)
      end
      
      if opt.display then
        _im1_ = image.display{image={ inputTable[#inputTable-4]:squeeze(),
                                      inputTable[#inputTable-3]:squeeze(),
                                      inputTable[#inputTable-2]:squeeze(),
                                      inputTable[#inputTable-1]:squeeze(),
                                      inputTable[#inputTable]:squeeze(),
                                      target:squeeze(),
                                      output:squeeze() },
                              win = _im1_, nrow = 7, legend = 't-4, -3, -2, -2, t, Target, Output'}
      end  
    end
  end
  print ('Training completed!')
  collectgarbage()
end
main()
