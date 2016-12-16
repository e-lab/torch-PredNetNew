--------------------------------------------------------------------------------
-- Train any prednet or prednet like model
-- First call __init(opt) and then updateModel() to train model
--
-- opt is a table containing following indexes:
-- batch, channels, datapath, dev, display, layers, learningRate,
-- learningRateDecay, momentum, nGPU, weightDecay
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local train = {}

require 'optim'
require 'image'

function train:__init(opt)
   -- Model parameter
   self.layers = opt.layers

   self.dev     = opt.dev
   self.display = opt.display

   -- Optimizer parameter
   self.optimState = {learningRate      = opt.learningRate,
                      momentum          = opt.momentum,
                      learningRateDecay = opt.learningRateDecay,
                      weightDecay       = opt.weightDecay}

   -- Dataset parameters
   local datapath = opt.datapath .. '/trainData.t7'
   self.dataset = torch.load(datapath):float()
   self.dataset = self.dataset/self.dataset:max()
   print("Loaded " .. self.dataset:size(1) .. " training image sequences")

   self.batch      = opt.batch
   self.seq        = self.dataset:size(2)
   opt.channels[opt.channels[0] and 0 or 1] = self.dataset:size(3)
   self.height     = self.dataset:size(4)
   self.width      = self.dataset:size(5)
   self.shuffle    = opt.shuffle

   opt.seq    = self.seq
   opt.height = self.height
   opt.width  = self.width

   print("Image resolution: " .. self.height .. " x " .. self.width)

   self.channels = opt.channels

   -- Initialize model generator
   local model
   if     opt.model == 'pred' then model = require 'models.prednet'
   elseif opt.model == 'PCBC' then model = require 'models.PCBC'
   else error('Model not supported.') end

   model:__init(opt)
   -- Get the model unwrapped over time as well as the prototype
   self.model, self.prototype = model:getModel()
   self.saveProto = self.prototype:clone() -- float saving prototype
   self.criterion = nn.MSECriterion()       -- citerion to calculate loss

   if self.dev == 'cuda' then
      self.model:cuda()
      self.criterion:cuda()

      local cp = {}                                   -- Color Pallete
      cp.r     = '\27[31m'
      cp.g     = '\27[32m'
      cp.reset = '\27[0m'

      -- Use multiple GPUs
      local gpuList = {}
      for i = 1, opt.nGPU do gpuList[i] = i end
      self.model = nn.DataParallelTable(1, true, true):add(self.model:cuda(), gpuList)
      print(cp.g .. '\n' .. opt.nGPU .. " GPUs being used" .. cp.reset)
   end

   -- Put model parameters into contiguous memory
   self.w, self.dE_dw = self.model:getParameters()
   print("# of parameters " .. self.w:nElement())

   return self.saveProto, self.saveProto:getParameters()
end


function train:updateModel()
   local model = self.model
   local criterion = self.criterion
   local w = self.w
   local dE_dw = self.dE_dw

   model:training()                       -- Ensure model is in training mode

   local trainError = 0
   local interFrameError = 0
   local optimState = self.optimState
   local L          = self.layers
   local channels   = self.channels
   local height     = self.height
   local width      = self.width
   local seq        = self.seq
   local batch      = self.batch

   local dataSize = self.dataset:size(1)
   local s = self.shuffle
   local shuffle = s and torch.randperm(dataSize) or torch.range(1, dataSize)

   local time = sys.clock()

   -- Initial state/input of the network
   -- {imageSequence, RL+1, R1, E1, R2, E2, ..., RL, EL}
   local c = channels[0] -- false for PredNet model
   local m = c and 1 or 2 -- multiplier for double E maps
   local H0 = {}
   H0[3] = c and torch.Tensor() or torch.zeros(batch, channels[1], height, width)
   height = c and height/2 or height                                   -- C1[0]
   width  = c and width/2 or width
   H0[4] = torch.zeros(batch, channels[1], height, width)              -- H1[0]
   H0[5] = torch.zeros(batch, m*channels[1], height, width)            -- E1[0]

   for l = 2, L do
      height = height/2
      width  = width/2                                                 -- C1[0]
      H0[3*l]   = c and torch.Tensor() or torch.zeros(batch, channels[l], height, width)
      H0[3*l+1] = torch.zeros(batch, channels[l], height, width)       -- Hl[0]
      H0[3*l+2] = torch.zeros(batch, m*channels[l], height, width)     -- El[0]
   end
   height = height/2
   width  = width/2
   H0[2] = torch.zeros(batch, channels[L+1], height, width)            -- RL+1

   if self.dev == 'cuda' then
      for l = 2, 3*L+2 do
         H0[l] = H0[l]:cuda()
      end
   end

   -- Dimension seq x channels x height x width
   c = c or channels[1] -- input channels
   H0[1] = torch.Tensor(batch, seq, c, self.height, self.width)
   if self.dev == 'cuda' then H0[1] = H0[1]:cuda() end
   local prediction = H0[1]:clone()
   local H = {}; for i = 1, #H0 do H[i] = H0[i] end

   for itr = 1, dataSize, batch do
      if itr + batch > dataSize then
         break
      end

      xlua.progress(itr, dataSize)

      for i = itr, itr + batch - 1 do
         local tseq = self.dataset[shuffle[i]]  -- 1 -> 20 input image
         H0[1][i-itr+1]:copy(tseq:resize(1, seq, c, self.height, self.width))
      end

      local h

      local eval_E = function()
--------------------------------------------------------------------------------
         -- Forward pass
--------------------------------------------------------------------------------
         -- Output is table of all predictions
         -- T = seq
         -- {Y1, ..., YT, C1, H1, E1, ..., CL, HL, EL}, # = T + 3*L
         h = model:forward(H)
         -- Merge all the predictions into a batch from 2 -> LAST sequence
         --       Table of 2         Batch of 2
         -- {(64, 64), (64, 64)} -> (2, 64, 64)
         for i = 1, seq do prediction:select(2, i):copy(h[i]) end
         if self.shuffle or iter == 1 then -- Ignore 1st pred if shuffle or 1st
            prediction:select(2, 1):copy(H0[1]:select(2, 1))
         end

         local err = criterion:forward(prediction, H0[1])

         -- Reset gradParameters
         model:zeroGradParameters()

--------------------------------------------------------------------------------
         -- Backward pass
--------------------------------------------------------------------------------
         local dE_dh = criterion:backward(prediction, H0[1])

         -- model:backward() expects dE_dh to be a table of sequence length
         -- Since 1st frame was ignored while calculating error (prediction[1] = H0[1][1]),
         -- 1st tensor in dE_dhTable is just a zero tensor
         local dE_dhTable = {}
         for i = 1, seq do dE_dhTable[i] = dE_dh:select(2, i) end
         for i = 1, 3*L do dE_dhTable[seq+i] = H0[2+i] end

         model:backward(H, dE_dhTable)

         if not self.shuffle then for i = 1, 3*L do H[2+i] = h[seq+i] end end

         -- Display last prediction of every sequence
         if self.display then
            self.dispWin = image.display{
               image=torch.cat(H0[1]:select(2, seq), prediction:select(2, seq), 4),
               legend='Train - Real | Pred',
               win = self.dispWin,
               nrow = 1,
            }
         end

         return err, dE_dw
      end

      local err
      w, err = optim.adam(eval_E, w, optimState)

      trainError = trainError + err[1]
      interFrameError = interFrameError +
         criterion:forward(prediction:select(2, seq), H0[1]:select(2, seq-1))
   end

   -- Calculate time taken by 1 epoch
   time = sys.clock() - time
   trainError = trainError/dataSize
   interFrameError = interFrameError/dataSize
   print("\nTraining Error: " .. trainError)
   print("Time taken to learn 1 sample: " .. (time*1000/dataSize) .. "ms")

   --collectgarbage()
   return trainError, interFrameError
end

return train
