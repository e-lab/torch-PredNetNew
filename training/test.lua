function test(opt,datasetSeq,epoch,testLog)

   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   print('==> training model')
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   model:evaluate()

   local cerr, ferr, loss = 0, 0, 0

   -- set training iterations and epochs according to dataset size:
  print('Validation epoch #', epoch)

   -- train:
   local iteration = datasetSeq:size()
   for t = 1, iteration do
        xlua.progress(t, iteration)
      -- define eval closure
        local f = 0

        model:zeroGradParameters()
        local sample = datasetSeq[t]
        local inTableG0, targetC, targetF = prepareData(opt,sample)
        --Get output
        -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
        output = model:forward(inTableG0)
        tcerr , tferr = computMatric(targetC, targetF, output)
        -- estimate f and gradients
        -- Calculate Error and sum
        cerr = cerr + tcerr
        ferr = ferr + tferr
        loss = loss + output[2]:sum()

      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        -- Display
        if opt.display then
           display(seqTable,targetF,targetC)
        end
      end
   end
   cerr = cerr/iteration
   ferr = ferr/iteration
   loss = loss/iteration
   writLog(cerr,ferr,loss,testLog)
   print ('Validation completed!')
end
