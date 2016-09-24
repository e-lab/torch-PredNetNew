unpack = unpack or table.unpack

require 'nn'
-- require 'cunn'
require 'paths'
require 'torch'
-- require 'cutorch'
require 'image'
require 'optim'
-- require 'ConvLSTM'
require 'env'

torch.setdefaulttensortype('torch.FloatTensor') 
-- does not work on float, maybe rnn package from Element Research?

local function main()
  -- cutorch.setDevice(1)
  paths.dofile('opts-mnist.lua')
  paths.dofile('data-mnist.lua')
  paths.dofile('model-matchnet.lua')
  -- print('This is the model:', {model})

  -- test model:
  -- local a = model:forward(torch.CudaTensor(1,1,64,64))
  -- print('Network test: output size:', a:size())
  -- local inputTable = {}
  -- for i = 1, 19 do
    -- table.insert(inputTable, torch.CudaTensor(1,64,64))
  -- end
  -- local a = model:updateOutput(inputTable)
  -- print('Network test: output size w table:', #a, a[1]:size())

  datasetSeq = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames

  print  ('Loaded ' .. datasetSeq:size() .. ' images')

  print('==> training model')

  torch.manualSeed(opt.seed)

  -- init LSTM parameters to small values, uniformly distributed
  -- local lstm_params, lstm_grads = model.modules[2].modules[2].modules[1].module:getParameters()
  -- local lstm_params, lstm_grads = model.modules[2].modules[1]:getParameters()
  -- lstm_params:uniform(-0.08,0.08)
  -- init LSTM biases to (forget_bias, other_bias)
  -- model.modules[2].modules[1]:initBias(0,0)
  -- call LSTM forget to reset the memory
  -- model.modules[2].modules[1]:forget()
  -- useful to display optical flow
  -- local optical_flow = model.modules[2].modules[2].modules[3].modules[7].output  

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
      
        print (' ==== Displaying weights ==== ')
        -- -- get weights
        -- eweight = model.modules[1].modules[1].modules[1].modules[1].weight
        -- dweight = model.modules[4].modules[2].weight
        -- dweight_cpu = dweight:view(opt.nFilters[2], opt.kernelSize, opt.kernelSize)
        -- eweight_cpu = eweight:view(opt.nFilters[2], opt.kernelSize, opt.kernelSize)
        -- -- render filters
        -- dd = image.toDisplayTensor{input=dweight_cpu,
        --                            padding=2,
        --                            nrow=math.floor(math.sqrt(opt.nFilters[2])),
        --                            symmetric=true}
        -- de = image.toDisplayTensor{input=eweight_cpu,
        --                            padding=2,
        --                            nrow=math.floor(math.sqrt(opt.nFilters[2])),
        --                            symmetric=true}

        -- -- live display
        -- if opt.display then
        --    _win1_ = image.display{image=dd, win=_win1_, legend='Decoder filters', zoom=8}
        --    _win2_ = image.display{image=de, win=_win2_, legend='Encoder filters', zoom=8}
        -- end
      end  
    end
  end
  print ('Training done')
  collectgarbage()
end
main()
