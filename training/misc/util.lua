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
   targetP, targetC= torch.Tensor(), torch.Tensor()
   targetC:resizeAs(data[1]):copy(data[data:size(1)])
   targetP:resizeAs(data[1]):copy(data[data:size(1)])
   if opt.useGPU then
      targetP = targetP:cuda()
      targetC = targetC:cuda()
   end
   return inTableG0, targetP, targetC
end
function display(seqTable,targetC,targetP,output)
         local pic = { seqTable[#seqTable-3]:squeeze(),
                       seqTable[#seqTable-2]:squeeze(),
                       targetP:squeeze(),
                       targetC:squeeze(),
                       output[1]:squeeze() }
         _im1_ = image.display{image=pic, min=0, max=1, win = _im1_, nrow = 7,
                            legend = 't-3, t-2, t-1, Target, Prediction'}
end
function save(target, output, model, optimState, opt)
    if opt.savePics and math.fmod(t, opt.dataEpoch) == 1 and t>1 then
      image.save(opt.savedir ..'/pic_target_'..t..'.jpg', target)
      image.save(opt.savedir ..'/pic_output_'..t..'.jpg', output)
    end

    if opt.save and math.fmod(t, opt.dataEpoch) == 1 and t>1 then
      torch.save(opt.savedir .. '/model_' .. t .. '.net', model)
      torch.save(opt.savedir .. '/optimState_' .. t .. '.t7', optimState)
    end
end
