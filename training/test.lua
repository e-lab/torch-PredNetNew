function test(opt,datasetSeq,epoch)

   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   print('==> training model')
   print  ('Loaded ' .. datasetSeq:size() .. ' images')
   model:evaluate()

   local err = 0

   -- set training iterations and epochs according to dataset size:
  print('Validation epoch #', epoch)

   -- train:
   local iteration = datasetSeq:size()
   for t = 1, iteration do

      -- define eval closure
        local f = 0

        model:zeroGradParameters()
        local sample = datasetSeq[t]
        local inTableG0, targetP, targetC = prepareData(opt,sample)
        --Get output
        -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
        output = model:forward(inTableG0)
        -- estimate f and gradients
        -- Calculate Error and sum
        err = err + output[2]:sum()

      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        print('==>Test iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq)  )
        err = 0
        -- Display
        if opt.display then
           display(seqTable,targetC,targetP)
        end
      end
   end
   print ('Validation completed!')
end
