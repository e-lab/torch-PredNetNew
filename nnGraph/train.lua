require 'model-m2NetV2'
require 'optim'
require 'image'
require 'torch'
require 'cutorch'
require 'cudnn'
require 'cunn'
function train(opt)
   imSize = opt.imSize
   prevE  = opt.prevE
   cellCh = opt.cellCh

   --Get main model
   predNet = mNet(opt.nlayers,opt.input_stride,opt.poolsize,opt.channels,opt.clOpt)
   param, gradParam = predNet:getParameters()
   --Train model clones
   criterion, model = createModel(opt, predNet)
   opt.gpu = true
   if opt.gpu then
      model:cuda()
      criterion:cuda()
      predNet:cuda()
      param:cuda()
      gradParam:cuda()
   end
   print(model)
   --Get dataset
   paths.dofile('data-mnist.lua')
   datasetSeq = getdataSeq_mnist(opt.dataFile) -- we sample nSeq consecutive frames
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   print('==> training model')
   target  = torch.Tensor()--= torch.Tensor(opt.transf,opt.memorySizeH, opt.memorySizeW)
   local initState = {}
   for i = nlayers , 1, -1 do
      initState[3*(i-1)+1] = torch.Tensor(prevE[i],imSize[i],imSize[i]):zero()
      initState[3*(i-1)+2] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
      initState[3*(i-1)+3] = torch.Tensor(cellCh[i],imSize[i],imSize[i]):zero()
   end
   -- Init state for top LSTM
   print('Test initState')
   if opt.gpu then
      for i, p in ipairs(initState) do
         initState[i] = p:cuda()
      end
   end
   local eta0 = 1e-6
   local eta = 1e-3
--Training loop
   iterMax = datasetSeq:size(1) * 100
   local optimState = {
    learningRate = eta,
    momentum = 0.9,
    learningRateDecay = 0
  }
   model:training()
   for t =1 , iterMax do


      local err = 0
       -- define eval closure
      local feval = function(param)
        local f = 0

         model:zeroGradParameters()
         sample = datasetSeq[t]
         frames = sample[1]
         inputTable ={}
         for i = 1,frames:size(1)-1 do
           table.insert(inputTable, frames[i]:cuda())
         end

         --target:resizeAs(frames[1]):copy(frames[frames:size(1)])
         target = frames[20]
         if opt.gpu then target = target:cuda() end
         -- test:
         local e,h,c,ht = {} ,{} ,{} ,{}
         --print(inputTable)
         input = {inputTable, unpack(initState)}
         out = model:forward(input)
         f = f + criterion:forward(out,target)
         dfdo = criterion:backward(out,target)
         model:backward(input,dfdo)
         opt.clipSize = 0.1
         --gradParam:clamp(-opt.clipSize,opt.clipSize)
         --Display input
         if false and math.fmod(t,20) == 0 then
           _im1_ = image.display{image={ inputTable[#inputTable-4]:squeeze(),
                                         inputTable[#inputTable-3]:squeeze(),
                                         inputTable[#inputTable-2]:squeeze(),
                                         inputTable[#inputTable-1]:squeeze(),
                                         inputTable[#inputTable]:squeeze(),
                                         target:squeeze(),
                                         out:squeeze() },
                                 win = _im1_, nrow = 7, legend = 't-4, -3, -2, -2, t, Target, Output'}
         end
         return f, gradParam
      end

      _,fs = optim.adam(feval, param, optimState)

      err = err + fs[1]
      if math.fmod(t,1) == 0 then      print('Error : ',err, 'iter size: ', t) end
   end

end
