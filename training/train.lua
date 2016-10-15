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

   local iteartion
   if opt.iteration == 0 then
      iteration = datasetSeq:size()
   else
      iteration = opt.iteration
   end
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
         -- Criterion is embedded
         -- estimate f and gradients
         local dE_dy = {torch.zeros(output[1]:size()):cuda(),output[2]}
         -- Update Grad input
         model:backward(inTableG0,dE_dy)

         -- Display and Save picts
         if math.fmod(t, opt.disFreq) == 0 then
           display(opt, seqTable, targetF, targetC, output[1])
         end
         if opt.savePic then
           if math.fmod(t, opt.picFreq) == 0 then
           savePics(opt,targetF,output[1],epoch,t)
           end
         end
         --Calculate Matric
         -- Calculate Error and sum
         tcerr , tferr = computMatric(targetC, targetF, output)
         cerr = cerr + tcerr
         ferr = ferr + tferr
         f = output[2]:sum()
         -- return f and df/dw
         return f, dE_dw
      end
      --Update model
      _,fs = optim.adam(eval_E, w, optimState)
      -- compute statistics / report error
      loss = loss + fs[1]
      --------------------------------------------------------------------
   end
   -- Save model
   if math.fmod(epoch, opt.saveEpoch) == 0  then
      save(model, optimState, opt, epoch)
   end
   --Average errors
   cerr = cerr/iteration
   ferr = ferr/iteration
   loss = loss/iteration
   writLog(cerr,ferr,loss,trainLog)
   print('Learning Rate: ',optimState.learningRate)
   print ('Training completed!')
end
