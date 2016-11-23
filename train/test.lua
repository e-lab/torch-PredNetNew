local test = {}

require 'optim'
require 'image'

-- local packages

function test:__init(opt)
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
   self.channels = opt.channels

   local datapath = opt.testData
   self.dataset = torch.load(datapath):float()
   self.dataset = self.dataset/self.dataset:max()
   print("Loaded " .. self.dataset:size(1) .. " testing image sequences")

   self.batch  = opt.batch
   self.seq    = self.dataset:size(2)
   self.height = self.dataset:size(4)
   self.width  = self.dataset:size(5)

   self.criterion = nn.MSECriterion()       -- citerion to calculate loss

   if self.dev == 'cuda' then
      self.criterion:cuda()
   end

end


function test:updateModel(model)
   local criterion = self.criterion
   local w = self.w
   local dE_dw = self.dE_dw

   model:evaluate()                       -- Ensure model is in evaluate mode

   local testError, err, interFrameError = 0, 0, 0
   local optimState = self.optimState
   local L = self.layers
   local channels = self.channels
   local height = self.height
   local width  = self.width
   local seq = self.seq
   local batch = self.batch

   local dataSize = self.dataset:size(1)
   local shuffle = torch.randperm(dataSize)  -- Get shuffled index of dataset

   local time = sys.clock()

   -- Initial state/input of the network
   -- {imageSequence, RL+1, R1, E1, R2, E2, ..., RL, EL}
   local H0 = {}
   H0[3] = torch.zeros(batch, channels[1], height, width)              -- C1[0]
   H0[4] = torch.zeros(batch, channels[1], height, width)              -- H1[0]
   H0[5] = torch.zeros(batch, 2*channels[1], height, width)            -- E1[0]

   for l = 2, L do
      height = height/2
      width  = width/2
      H0[3*l]   = torch.zeros(batch, channels[l], height, width)       -- C1[0]
      H0[3*l+1] = torch.zeros(batch, channels[l], height, width)       -- Hl[0]
      H0[3*l+2] = torch.zeros(batch, 2*channels[l], height, width)     -- El[0]
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
      xSeq:resize(batch, seq, channels[1], self.height, self.width)
      for i = itr, itr + batch - 1 do
         local tseq = self.dataset[shuffle[i]]  -- 1 -> 20 input image
         xSeq[i-itr+1] = tseq:resize(1, seq, channels[1], self.height, self.width)
      end

      H0[1] = xSeq:clone()

      local h = {}
      local prediction = xSeq:clone()

      if self.dev == 'cuda' then
         prediction = prediction:cuda()
         xSeq = xSeq:cuda()
         H0[1] = H0[1]:cuda()
      end

-----------------------------------------------------------------------------
      -- Forward pass
-----------------------------------------------------------------------------
      -- Output is table of all predictions
      h = model:forward(H0)
      -- Merge all the predictions into a batch from 2 -> LAST sequence
      --       Table of 2         Batch of 2
      -- {(64, 64), (64, 64)} -> (2, 64, 64)
      for i = 2, #h do
         prediction:select(2, i):copy(h[i])
      end

      err = criterion:forward(prediction, xSeq)

      -- Display last prediction of every sequence
      if self.display then
            self.dispWin = image.display{
               image=torch.cat(xSeq:select(2, seq), prediction:select(2, seq), 4),
               legend='Test - Real | Pred',
               win = self.dispWin,
               nrow = 1,
            }
      end

      testError = testError + err
      interFrameError = interFrameError +
         criterion:forward(prediction:select(2, seq), xSeq:select(2, seq-1))
   end

   -- Calculate time taken by 1 epoch
   time = sys.clock() - time
   testError = testError/dataSize
   interFrameError = interFrameError/dataSize
   print("\nTest Error: " .. testError)
   print("Time taken to learn 1 sample: " .. (time*1000/dataSize) .. "ms")

   collectgarbage()
   return testError, interFrameError
end

return test
