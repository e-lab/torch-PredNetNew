-- This can be used for visualizing and testing trained PredNet model
--
-- Abhishek Chaurasia & Alfredo Canziani
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'
require 'image'
require 'qtwidget'

torch.setdefaulttensortype('torch.FloatTensor')

-- Gather all the arguments
local opts = require 'opts'
local opt = opts.parse(arg)

if opt.dev == 'cuda' then
   require 'cunn'
   require 'cudnn'
end

torch.manualSeed(opt.seed)

local L = opt.layers

--------------------------------------------------------------------------------
-- Network
local modelPath
modelPath = opt.dmodel .. '/model-' .. opt.net .. '.net'
assert(paths.filep(modelPath), 'Model not present at ' .. modelPath)
print("Loading model from: " .. modelPath)

local model = {}
model[1] = torch.load(modelPath)
model[2] = torch.load(modelPath)

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cuda' then
   model[1]:cuda()
   model[2]:cuda()
end



-- local img = frame.forward(img)
local dataset = torch.load(opt.input):float()
local batches = opt.nrow
local height = dataset:size(4)
local width = dataset:size(5)

local img, imgGPU, predImgGPU

-- Width and height of each window to be displayed
-- Multiple images are shown in grid format
-- Display windows: Input/Predicted/Fake image
local winImg

-- Input/Output channels for A of every layer
local channels = {}
-- XXX Do not forget to change 1st channel number based on if it is RGB or gray-scale image
channels[0] = 3
for l = 1, opt.layers + 1 do
   channels[l] = 2^(l + 4)
end
-- {[0]=1|3, 32, 64, 128, 256, 512} -> PCBC

-- Initial states
local init = require('init_'..opt.model)
local H0 = init(channels, height, width, L, opt.dev)
local check = 0
for itr = 1, dataset:size(1) do

   local frames = {}
   local predImg

   local seqLength = dataset:size(2)
   for t = 1, seqLength do
      -- Input frame should be 3 dimensional
      -- channel x height x width
      img = dataset[itr][t]
      frames[t] = img:clone()
      predImg = predImg or img

      if opt.dev == 'cuda' then
         imgGPU = imgGPU or torch.CudaTensor(img:size())
         imgGPU:copy(img)
         img = imgGPU
         H0[1][1] = img:cuda()

         predImgGPU = predImgGPU or torch.CudaTensor(predImg:size())
         predImgGPU:copy(predImg)
         predImg = predImgGPU
         H0[2][1] = predImg:cuda()
      else
         H0[1][1] = img
         H0[2][1] = predImg
      end

      local h = {}
      h[1] = model[1]:forward(H0[1])
      h[2] = model[2]:forward(H0[2])
      for l = 2, #h[1] do
         H0[1][l+1] = h[1][l]
         H0[2][l+1] = h[2][l]
      end

      -- Store all the input frames and predictions of one sequence
      frames[t+1*seqLength] = h[1][1]:clone()
      frames[t+2*seqLength] = h[2][1]:clone()

      -- Copy output of previous prediction onto input
      predImg:copy(h[2][1])

   end
   winImg = image.display{
      image = frames,
      legend = 'Original frames / Predicted frames / Imagined frames',
      nrow = batches, win = winImg, min = 0, max = 1,
   }
   io.write("i: init, e: exit, <Return>: keep predicting: "); io.flush()
   local c = io.read()
   if c == 'i' then H0 = init(channels, height, width, L, opt.dev)
   elseif c == 'e' then break end
end

winImg.window:close()
