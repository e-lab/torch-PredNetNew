function computMatric(targetC, targetF, output)
   criterion = nn.MSECriterion()
   cerr = criterion:forward(targetC,output[1])
   ferr = criterion:forward(targetF,output[1])
   return cerr, ferr
end
function writLog(cerr,ferr,loss,logger)
   print(string.format('cerr : %.4f ferr: %.4f loss: %.2f',cerr, ferr, loss))
   logger:add{
      ['cerr'] = cerr,
      ['ferr']  = ferr,
      ['loss'] = loss
   }
end
function prepareData(opt, sample)
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
   end
   -- reset initial network state:
   local inTableG0 = {}
   for L=1, opt.nlayers do
     if opt.useGPU then
       table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- E(t-1)
       table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- C(t-1)
       table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)):cuda() ) -- H(t-1)
     else
       table.insert( inTableG0, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)) ) -- E(t-1)
       table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- C(t-1)
       table.insert( inTableG0, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- H(t-1)
     end
   end
   -- get input video sequence data:
   seqTable = {} -- stores the input video sequence
   data = sample[1]
   for i = 1, data:size(1) do
     if opt.useGPU then
       table.insert(seqTable, data[i]:cuda())
     else
       table.insert(seqTable, data[i]) -- use CPU
     end
   end
   -- prepare table of states and input:
   table.insert(inTableG0, seqTable)
   -- Target
   targetC, targetF= torch.Tensor(), torch.Tensor()
   targetF:resizeAs(data[1]):copy(data[data:size(1)])
   targetC:resizeAs(data[1]):copy(data[data:size(1)-1])
   if opt.useGPU then
      targetF = targetF:cuda()
      targetC = targetC:cuda()
   end
   return inTableG0, targetC, targetF
end
function display(opt, seqTable,targetF,targetC,output)
   if opt.display then
      require 'env'
      local pic = { seqTable[#seqTable-3]:squeeze(),
                    seqTable[#seqTable-2]:squeeze(),
                    targetC:squeeze(),
                    targetF:squeeze(),
                    output:squeeze() }
      _im1_ = image.display{image=pic, min=0, max=1, win = _im1_, nrow = 7,
                         legend = 't-3, t-2, t-1, Target, Prediction'}
   end
end
function savePics(opt,target,output,epoch,t)
   --Save pics
   print('Save pics!')
   if math.fmod(t, opt.picFreq) == 0 then
      image.save(paths.concat(opt.savedir ,'pic_target_'..epoch..'_'..t..'.jpg'), target)
      image.save(paths.concat(opt.savedir ,'pic_output_'..epoch..'_'..t..'.jpg'), output)
   end
end
function save( model, optimState, opt, epoch)
   --Save models
   if opt.save  then
      print('Save models!')
      if opt.multySave then
         torch.save(paths.concat(opt.savedir ,'model_' .. epoch .. '.net'), model)
         torch.save(paths.concat(opt.savedir ,'optimState_' .. epoch .. '.t7'), optimState)
         torch.save(paths.concat(opt.savedir ,'opt' .. epoch .. '.t7'), opt)
      else
         torch.save(paths.concat(opt.savedir ,'model.net'), model)
         torch.save(paths.concat(opt.savedir ,'optimState.t7'), optimState)
         torch.save(paths.concat(opt.savedir ,'opt.t7'), opt)
      end
   end
end
