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
   local shuffle = torch.randperm(dataSize)  -- Get shuffled index of dataset

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

   for itr = 1, dataSize, batch do
      if itr + batch > dataSize then
         break
      end

      xlua.progress(itr, dataSize)

      -- Dimension seq x channels x height x width
      local xSeq = torch.Tensor()
      c = c or channels[1] -- input channels
      xSeq:resize(batch, seq, c, self.height, self.width)
      for i = itr, itr + batch - 1 do
         local tseq = self.dataset[shuffle[i]]  -- 1 -> 20 input image
         xSeq[i-itr+1] = tseq:resize(1, seq, c, self.height, self.width)
      end

      H0[1] = xSeq:clone()

      local h = {}
      local prediction = xSeq:clone()

      if self.dev == 'cuda' then
         prediction = prediction:cuda()
         xSeq = xSeq:cuda()
         H0[1] = H0[1]:cuda()
      end

      local eval_E = function()
--------------------------------------------------------------------------------
         -- Forward pass
--------------------------------------------------------------------------------
         -- Output is table of all predictions
         h = model:forward(H0)
         -- Merge all the predictions into a batch from 2 -> LAST sequence
         --       Table of 2         Batch of 2
         -- {(64, 64), (64, 64)} -> (2, 64, 64)
         for i = 2, #h do
            prediction:select(2, i):copy(h[i])
         end

         local err = criterion:forward(prediction, xSeq)

         -- Reset gradParameters
         model:zeroGradParameters()

--------------------------------------------------------------------------------
         -- Backward pass
--------------------------------------------------------------------------------
         local dE_dh = criterion:backward(prediction, xSeq)

         -- model:backward() expects dE_dh to be a table of sequence length
         -- Since 1st frame was ignored while calculating error (prediction[1] = xSeq[1]),
         -- 1st tensor in dE_dhTable is just a zero tensor
         local dE_dhTable = {}
         dE_dhTable[1] = dE_dh:select(2, 1):clone():zero()
         for i = 2, seq do
            dE_dhTable[i] = dE_dh:select(2, i)
         end

         model:backward(H0, dE_dhTable)

         -- Display last prediction of every sequence
         if self.display then
            self.dispWin = image.display{
               image=torch.cat(xSeq:select(2, seq), prediction:select(2, seq), 4),
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
         criterion:forward(prediction:select(2, seq), xSeq:select(2, seq-1))
   end

   -- Calculate time taken by 1 epoch
   time = sys.clock() - time
   trainError = trainError/dataSize
   interFrameError = interFrameError/dataSize
   print("\nTraining Error: " .. trainError)
   print("Time taken to learn 1 sample: " .. (time*1000/dataSize) .. "ms")

   collectgarbage()
   return trainError, interFrameError
end

return train
