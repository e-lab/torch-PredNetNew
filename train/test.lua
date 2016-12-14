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

   local datapath = opt.datapath .. 'testData.t7'
   self.dataset = torch.load(datapath):float()
   self.dataset = self.dataset/self.dataset:max()
   print("Loaded " .. self.dataset:size(1) .. " testing image sequences")

   self.batch  = opt.batch
   self.seq    = self.dataset:size(2)
   self.height = self.dataset:size(4)
   self.width  = self.dataset:size(5)
   self.shuffle    = opt.shuffle

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
   local shuffle
   if self.shuffle then
      shuffle = torch.randperm(dataSize)  -- Get shuffled index of dataset
   else
      shuffle = torch.range(1, dataSize)
   end

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

-----------------------------------------------------------------------------
      -- Forward pass
-----------------------------------------------------------------------------
      -- Output is table of all predictions
      h = model:forward(H0)
      -- Merge all the predictions into a batch from 2 -> LAST sequence
      --       Table of 2         Batch of 2
      -- {(64, 64), (64, 64)} -> (2, 64, 64)
      prediction:select(2, 1):copy(H0[1]:select(2, 1))
      for i = 2, #h do
         prediction:select(2, i):copy(h[i])
      end

      err = criterion:forward(prediction, H0[1])

      -- Display last prediction of every sequence
      if self.display then
            self.dispWin = image.display{
               image=torch.cat(H0[1]:select(2, seq), prediction:select(2, seq), 4),
               legend='Test - Real | Pred',
               win = self.dispWin,
               nrow = 1,
            }
      end

      testError = testError + err
      interFrameError = interFrameError +
         criterion:forward(prediction:select(2, seq), H0[1]:select(2, seq-1))
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
