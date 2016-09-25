-- MatchNet training: predicting future frames in video
-- Eugenio Culurciello, August - September 2016
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
--

require 'nn'
-- require 'cunn'
require 'paths'
require 'torch'
-- require 'cutorch'
require 'image'
require 'optim'
require 'env'
require 'pl'

-- lapp = require 'pl.lapp'
opt = lapp [[
  Command line options:
  --dir   (default outputs_mnist_line)  subdirectory to save experiments in
  --seed  (default 1250)          initial random seed

  Model parameters:
  --nlayers       (default 2)     number of layers of MatchNet
  --inputSizeW    (default 64)    width of each input patch or image
  --inputSizeH    (default 64)    width of each input patch or image
  --eta           (default 1e-4)  learning rate
  --etaDecay      (default 1e-5)  learning rate decay
  --momentum      (default 0.9)   gradient momentum
  --maxIter       (default 30000) max number of updates
  
  --nSeq          (default 19)    input video sequence lenght
  --nFilters      (default {1,32,32,32})  number of filters in the encoding/decoding layers
  --kernelSize    (default 7)     size of kernels in encoder/decoder layers
  --kernelSizeMemory (default 7)
  --padding       (default torch.floor(opt.kernelSize/2))  pad input before convolutions
  --gradClip      (default 50)
  --stride        (default 1)     stride in convolutions
  --poolsize      (default 2)     maxpooling size
  --constrWeight  (default {0,1,0.001})

  --dataFile      (default 'data-small-train.t7')
  --dataFileTest  (default 'data-small-test.t7')
  --modelFile     (default nil)
  --configFile    (default nil)
  --statInterval  (default 50)    interval for printing error
  -v              (default false) verbose output
  --display       (default true)  display stuff
  --displayInterval (default 50)
  -s,--save      (default true)   save models
  --saveInterval (default 10000)
]]
opt.nFilters  = {1,32,32,32} -- number of filters in the encoding/decoding layers
opt.constrWeight = {0,1,0.001}

torch.setdefaulttensortype('torch.FloatTensor') 
torch.manualSeed(opt.seed)


local function main()
  -- cutorch.setDevice(1)
  paths.dofile('data-mnist.lua')
  paths.dofile('model-matchnet.lua')
  -- print('This is the model:', {model})

  datasetSeq = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames

  print  ('Loaded ' .. datasetSeq:size() .. ' images')

  print('==> training model')

  parameters, grads = model:getParameters()
  print('Number of parameters ' .. parameters:nElement())
  print('Number of grads ' .. grads:nElement())

  local eta0 = 1e-6
  local eta = opt.eta
  local err = 0
  local iter = 0
  local epoch = 0
  rmspropconf = {learningRate = eta}
  
  model:training()

  -- train:
  for t = 1,opt.maxIter do

    -- define eval closure
    local feval = function()
      local f = 0
 
      model:zeroGradParameters()

      -- setup initial variables:
      local inTableG0 = {} -- global inputs reset
      for L=1, opt.nlayers do
         table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)))-- previous time E
         if L==1 then 
            table.insert( inTableG0, torch.zeros(opt.nFilters[L+1], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
         else
            table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
         end
      end
      inputTable = {}
      
      target  = torch.Tensor()--= torch.Tensor(opt.transf,opt.memorySizeH, opt.memorySizeW) 
      sample = datasetSeq[t]
      data = sample[1]
      for i = 1,data:size(1)-1 do
        table.insert(inputTable, data[i])--:cuda())
      end

      target:resizeAs(data[1]):copy(data[data:size(1)])
      target = target--:cuda()
      
      -- estimate f and gradients
      table.insert(inTableG0, inputTable)
      output = model:updateOutput( inTableG0 )
      gradtarget = gradloss:updateOutput(target):clone()
      gradoutput = gradloss:updateOutput(output)

      f = f + criterion:updateOutput(gradoutput,gradtarget)

      -- gradients
      local gradErrOutput = criterion:updateGradInput(gradoutput,gradtarget)
      local gradErrGrad = gradloss:updateGradInput(output,gradErrOutput)
           
      model:updateGradInput(inputTable,gradErrGrad)

      model:accGradParameters(inputTable, gradErrGrad)  

      grads:clamp(-opt.gradClip,opt.gradClip)
      return f, grads
    end
   
   
    if math.fmod(t,20000) == 0 then
      epoch = epoch + 1
      eta = opt.eta*math.pow(0.5,epoch/50)  
      rmspropconf.learningRate = eta  
    end
    
    _,fs = optim.rmsprop(feval, parameters, rmspropconf)

    err = err + fs[1]
    -- model:forget()
    --------------------------------------------------------------------
    -- compute statistics / report error
    if math.fmod(t , opt.nSeq) == 1 then
      print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..eta ) -- err/opt.statInterval)
      err = 0
      if opt.save and math.fmod(t , opt.nSeq*1000) == 1 and t>1 then
        -- clean model before saving to save space
        --  model:forget()
        -- cleanupModel(model)         
        torch.save(opt.dir .. '/model_' .. t .. '.bin', model)
        torch.save(opt.dir .. '/rmspropconf_' .. t .. '.bin', rmspropconf)
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
  print ('Training done')
  collectgarbage()
end
main()
