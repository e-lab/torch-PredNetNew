function train(opt,datasetSeq, epoch, trainLog)
   --Init optimState
   local optimState = {
     learningRate = opt.learningRate,
     momentum = opt.momentum,
     learningRateDecay = opt.learningRateDecay,
     weightDecay = opt.weightDecay
   }

      --Get model
   paths.dofile('models/model.lua')
   model = getModel()
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
      cutorch.setDevice(opt.GPUID)
      model:cuda()
   end
   print('==> training model')
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   model:training()
   local w, dE_dw = model:getParameters()
   print('Number of parameters ' .. w:nElement())
   print('Number of grads ' .. dE_dw:nElement())

   local cerr, ferr, loss= 0, 0, 0
   -- set training iterations and epochs according to dataset size:
  print('Training epoch #', epoch)

   -- train:
   local iteration = datasetSeq:size()
   iteration = 10
   for t = 1, iteration do
      xlua.progress(t, iteration)
      -- define eval closure
      local eval_E = function(w)

        model:zeroGradParameters()
        local sample = datasetSeq[t]
        local inTableG0, targetC, targetF = prepareData(opt,sample)
        --Get output
        -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
        output = model:forward(inTableG0)
        tcerr , tferr = computMatric(targetC, targetF, output)
        -- estimate f and gradients
        -- Criterion is embedded
        local dE_dy = {torch.zeros(output[1]:size()):cuda(),output[2]}
        -- Update Model
        model:backward(inTableG0,dE_dy)
        -- Calculate Error and sum
        cerr = cerr + tcerr
        ferr = ferr + tferr
        f = output[2]:sum()

        -- return f and df/dX
        return f, dE_dw
      end


      _,fs = optim.adam(eval_E, w, optimState)

      loss = loss + fs[1]

      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        -- Display
        if opt.display then
           display(seqTable,targetF,targetC)
        end
      end
   end
   -- Save file
   if math.fmod(epoch,opt.maxEpochs ) == 0  then
      save(target, output, model, optimState, opt)
   end
   cerr = cerr/iteration
   ferr = ferr/iteration
   loss = loss/iteration
   writLog(cerr,ferr,loss,trainLog)
   print('Learning Rate: ',optimState.learningRate)
   print ('Training completed!')
end
