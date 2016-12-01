--------------------------------------------------------------------------------
-- Prepare CamVid data, which can later be used to train decoder/classifier
--
-- Input is video frames used for training MatchNet
-- These frames are forwarded through MatchNet
-- Output for which labels exist are saved in a tensor,
-- which can be later used as input for the decoder for training.
--
-- Outputs are two tensors: Input and corresponding label
--
-- Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'
require 'image'
require 'qtwidget'
require 'xlua'

-- Gather all the arguments
local opts = require 'opts'
local opt = opts.parse(arg)

if opt.dev == 'cuda' then
   require 'cunn'
   require 'cudnn'
end

torch.manualSeed(opt.seed)

local L = opt.layers
local dirRoot  = opt.datapath
local imHeight = opt.imHeight
local imWidth  = opt.imWidth

local red      = '\27[31m'
local green    = '\27[32m'
local resetCol = '\27[0m'
--------------------------------------------------------------------------------
-- Network
local modelPath
modelPath = opt.dmodel .. '/model-' .. opt.net .. '.net'
assert(paths.filep(modelPath), 'Model not present at ' .. modelPath)
print("Loading model from: " .. modelPath)

local model = torch.load(modelPath)

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cuda' then
   model:cuda()
end
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- Input/Output channels for A of every layer
channels = torch.ones(L + 1)
-- XXX Do not forget to change 1st channel number based on if it is RGB or gray-scale image
channels[1] = 3
for l = 2, L + 1 do
   channels[l] = 2^(l+3)
end
-- {1, 32, 64, 128, 256, 512}

local function defaultH(height, width, channels, dev)
   local H0 = {}
   H0[3] = torch.zeros(channels[1], height, width)                  -- C1[0]
   H0[4] = torch.zeros(channels[1], height, width)                  -- H1[0]
   H0[5] = torch.zeros(2*channels[1], height, width)                -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      H0[3*l]   = torch.zeros(channels[l], height, width)           -- C1[0]
      H0[3*l+1] = torch.zeros(channels[l], height, width)           -- Hl[0]
      H0[3*l+2] = torch.zeros(2*channels[l], height, width)         -- El[0]
   end
   height = height/2
   width = width/2
   H0[2] = torch.zeros(channels[L+1], height, width)                -- RL+1

   -- Convert states into CudaTensors if device is cuda
   if dev == 'cuda' then
      for l = 2, 3*L+2 do
         H0[l] = H0[l]:cuda()
      end
   end
   return H0
end

--------------------------------------------------------------------------------
-- Function to check if the given file is a valid video
local function validVideo(filename)
   local ext = string.lower(paths.extname(filename))

   local videoExt = {'avi', 'mp4', 'mxf'}
   for i = 1, #videoExt do
      if ext == videoExt[i] then
         return true
      end
   end
   print(red .. ext .. " extension is NOT supported!!!" .. resetCol)
   return false
end

--------------------------------------------------------------------------------
-- Initialize class Frame which can be used to read videos/camera
local frame = assert(dofile('../framevideo.lua'))

-- TODO Get rid of this part
-- To do so you will have to modify framevideo.lua
local camRes = 'QHD'
local fps = 30

local source = {}
-- switch input sources
source.res = {QHD   = {w =  640, h =  360}}
source.w = source.res[camRes].w
source.h = source.res[camRes].h
source.fps = fps

local labelPrefixTable = {'0001TP_0', 'Seq05VD_f', '0006R0_f', '0016E5_'}
local labelStart      = {6690,   0, 930, 390}
local labelOffset     = {  30,   1, 931, 391}
local maxSampleFrames = { 124, 171, 101, 305}
local frameSeqTrain, frameSeqTest

-----------------------------------------------------------------------------------
-- forward img and gather label and representation for that label
-- Input : directory path containing videos, directory number
-- Output: tensors storing labels and their representation
-----------------------------------------------------------------------------------
local function forwardSeq(inputData, inputLabel, input, dirN)
   -- source height and width gets updated by __init based on the input video
   frame:init(input, source)
   local nFrames = frame.nFrames() or 2000          -- # of total frames in the video

   local currentFrame = torch.FloatTensor(3, imHeight, imWidth):zero()
   local frameToDisp  = torch.FloatTensor(3, 3, imHeight, imWidth):zero()

   local img = frame.forward(img)
   local n = - labelOffset[dirN]          -- Counter for progress bar
   local count = 1                        -- Counter for how many frames have been added to one sequence
   local switchFlag  = 'train'
   local switchCount = 1
   local labelPath
   local labelPrefix = labelPrefixTable[dirN]

   -- Initial states
   local H0 = defaultH(imHeight, imWidth, channels, opt.dev)

   -- Input data and labels
   local representation = torch.zeros(1, 3, imHeight, imWidth)
   local label = torch.zeros(1, imHeight, imWidth)

   local tempImg, imgGpu
   while count <= maxSampleFrames[dirN] do
      xlua.progress(count, maxSampleFrames[dirN])
      tempImg = image.scale(img[1], imWidth, imHeight)
      frameToDisp[1] = tempImg:clone()

      --------------------------------------------------------------------------
      -- Forward image to model
      -- Input frame should be 3 dimensional
      -- channel x height x width
      --------------------------------------------------------------------------
      if opt.dev == 'cuda' then
         imgGPU = imgGPU or torch.CudaTensor(tempImg:size())
         imgGPU:copy(tempImg)
         tempImg = imgGPU
         H0[1] = tempImg:cuda()
      else
         H0[1] = tempImg
      end

      local h = model:forward(H0)
      for l = 2, #h do
         H0[l+1] = h[l]
      end

      --------------------------------------------------------------------------
      -- Save representation alongwith corresponding label, only if label exists
      --------------------------------------------------------------------------
      frameToDisp[2]:zero()
      local labelIdx = n + labelStart[dirN]
      labelPath = dirRoot .. dirN .. '/label/' .. labelPrefix .. string.format('%05d_L.png', labelIdx)
      if paths.filep(labelPath) then
         count = count + 1

         -- Load current label
         local currentLabel = image.load(labelPath, 3, 'byte'):float()
         -- Rescale label
         frameToDisp[2] = image.scale(currentLabel, imWidth, imHeight, 'simple')
         -- Convert RGB into grayscale
         local labelRGB = frameToDisp[2]/64
         label[1] = labelRGB[1] * 16 + labelRGB[2] * 4 + labelRGB[3]

         -- Store representation and label in tensors
         representation[1] = h[3]:clone():float()

         if inputData then
            inputData  = inputData:cat(representation, 1)
            inputLabel = inputLabel:cat(label, 1)
         else
            inputData  = representation:clone()
            inputLabel = label:clone()
         end
      end
      frameToDisp[1] = 255*frameToDisp[1]
      frameToDisp[3] = frameToDisp[1] + frameToDisp[2]

      if paths.filep(labelPath) and n > 0 then
         win = image.display{image = frameToDisp, legend = 'Image / Label / Image+Label', win = win}
         -- io.read()
      end
      img = frame.forward(img)
      n = n + 1
   end
   collectgarbage()
   return inputData, inputLabel
end

--------------------------------------------------------------------------------
-- Section to convert images into tensor
local inputData, inputLabel
assert(paths.dirp(dirRoot), 'No folder found at: ' .. dirRoot)
for dirN = 1, 4 do
   local dirPath = dirRoot .. dirN .. '/input/'
   for file in paths.iterfiles(dirPath) do
      print(red .. "\nGetting representations for: " .. resetCol .. file)
      -- process each image
      if validVideo(file) then
         local vidPath = dirPath .. file
         inputData, inputLabel = forwardSeq(inputData, inputLabel, vidPath, dirN)
         print(green .. "Loaded representation and labels!!!" .. resetCol)
      end
   end
end

print(green .. "Representation resolution: " .. inputData:size(2) .. 'x' .. inputData:size(3)  .. 'x' .. inputData:size(4) .. resetCol)
print(green .. "Label resolution: " .. 3 .. 'x' .. inputLabel:size(2) .. 'x' .. inputLabel:size(3) .. resetCol)
print("Loading of representation and corresponding labels completed.")
print("\n# of representations and labels: " .. inputData:size(1))
print("\nSaving tensor to location: " .. opt.save)
local saveRep = opt.save .. '/input.t7'
local saveLabel = opt.save .. '/label.t7'
torch.save(saveRep,   inputData)
torch.save(saveLabel, inputLabel)
print(green .. "Tensor saved successfully!!!" .. resetCol)
