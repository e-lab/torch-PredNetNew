-- This can be used for visualizing and testing trained PredNet model
--
-- Abhishek Chaurasia
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

local model = torch.load(modelPath)

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cuda' then
   model:cuda()
end

-- Input/Output channels for A of every layer
channels = torch.ones(L + 1)
-- XXX Do not forget to change 1st channel number based on if it is RGB or gray-scale image
channels[1] = 3
for l = 2, L + 1 do
   channels[l] = 2^(l+3)
end
-- {1, 32, 64, 128, 256, 512}

-- -- Initialize class Frame which can be used to read videos/camera
-- local frame
-- if string.sub(opt.input, 1, 3)  == 'cam' and tonumber(string.sub(opt.input,4,-1)) ~= nil then
--    frame = assert(require('frame.framecamera'))
-- elseif opt.input:lower():match('%.jpe?g$') or opt.input:lower():match('%.png$') then
--    frame = assert(require('frame.frameimage'))
-- elseif paths.dirp(opt.input) then
--    frame = assert(require('frame.frameimage'))
-- else
--    frame = assert(require('frame.framevideo'))
-- end
--
-- local source = {}
-- -- switch input sources
-- source.res = {
--    HVGA  = {w =  320, h =  240},
--    QHD   = {w =  640, h =  360},
--    VGA   = {w =  640, h =  480},
--    FWVGA = {w =  854, h =  480},
--    HD    = {w = 1280, h =  720},
--    FHD   = {w = 1920, h = 1080},
-- }
-- source.w = source.res[opt.camRes].w
-- source.h = source.res[opt.camRes].h
-- source.fps = opt.fps
--
-- -- opt.input is mandatory
-- -- source height and width gets updated by __init based on the input video
-- frame:init(opt, source)

-- local img = frame.forward(img)
local dataset = torch.load(opt.input):float()
local batches = opt.nrow
local height = dataset:size(4)
local width = dataset:size(5)

local img, imgGPU

-- Width and height of each window to be displayed
-- Multiple images are shown in grid format
local mapE = {{        width,                                  height},
              {batches*width, math.ceil(2*channels[2]/batches)*height},
              {batches*width, math.ceil(2*channels[3]/batches)*height}}

local mapR = {{      2*width,                                height},
              {batches*width, math.ceil(channels[2]/batches)*height},
              {batches*width, math.ceil(channels[3]/batches)*height}}

-- Display windows: Input/Predicted image, Error, Representation respectively
local winImg, winE, winR

-- Create qtwidget windows for displaying all states
-- mapsX controls the size of the window
winE = {}
winR = {}
winC = {}
for l = 1, L do
   winE[l] = qtwidget.newwindow(mapE[l][1], mapE[l][2], 'Error: Layer ' .. l)
   winR[l] = qtwidget.newwindow(mapR[l][1], mapR[l][2], 'Hidden State: Layer ' .. l)
   winC[l] = qtwidget.newwindow(mapR[l][1], mapR[l][2], 'Cell State: Layer ' .. l)
end

local check = 0
for itr = 1, dataset:size(1) do
   height = dataset:size(4)
   width = dataset:size(5)
   -- Initial states
   local H0 = {}
   H0[3] = torch.zeros(channels[1], height, width)                  -- C1[0]
   H0[4] = torch.zeros(channels[1], height, width)                  -- H1[0]
   H0[5] = torch.zeros(2*channels[1], height, width)                -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      H0[3*l]   = torch.zeros(channels[l], height, width)           -- Cl[0]
      H0[3*l+1] = torch.zeros(channels[l], height, width)           -- Hl[0]
      H0[3*l+2] = torch.zeros(2*channels[l], height, width)         -- El[0]
   end
   height = height/2
   width = width/2
   H0[2] = torch.zeros(channels[L+1], height, width)                -- RL+1

   -- Convert states into CudaTensors if device is cuda
   if opt.dev == 'cuda' then
      for l = 2, 3*L+2 do
         H0[l] = H0[l]:cuda()
      end
   end

   local seqOfFrames = {}
   local errorImg = {}
   local RImg = {}
   local CImg = {}

   for l = 1, L do
      errorImg[l] = {}
      RImg[l] = {}
      CImg[l] = {}
   end

   for seq = 1, dataset:size(2) do
      -- Input frame should be 3 dimensional
      -- channel x height x width
      img = dataset[itr][seq]

      if opt.dev == 'cuda' then
         imgGPU = imgGPU or torch.CudaTensor(img:size())
         imgGPU:copy(img)
         img = imgGPU
         H0[1] = img:cuda()
      else
         H0[1] = img
      end

      local h = model:forward(H0)
      for l = 2, #h do
         H0[l+1] = h[l]
      end

      -- Store all the input frames and predictions of one sequence
      seqOfFrames[seq] = img:clone()
      seqOfFrames[seq+dataset:size(2)] = h[1]:clone()

      -- Store states for displaying
      for l = 1, L do
         -- Gather error maps
         for maps = 1, h[3*l+1]:size(1) do
            errorImg[l][maps] = h[3*l+1][maps]:clone()
         end
         -- Gather representation maps
         for maps = 1, h[3*l]:size(1) do
            RImg[l][maps] = h[3*l][maps]:clone()
            CImg[l][maps] = h[3*l-1][maps]:clone()
         end
      end
      if seq > 1 then
         -- Display different states
         -- Comment out lines whichever you do not want see
         for l = 1, L do
            image.display{image = errorImg[l], nrow = batches, zoom = 2^(l-1), win = winE[l]}
            image.display{image = RImg[l], nrow = batches, zoom = 2^(l-1), win = winR[l]}
            image.display{image = CImg[l], nrow = batches, zoom = 2^(l-1), win = winC[l]}
         end
         -- This pause gives time to position windows manualy.
         -- This pause happens only after the very first output is displayed
         if check == 0 then
            print("Press any key to continue ...")
            io.read()
            check = 1
         end
         -- This is to slow-down display
         os.execute("sleep " .. 0.1)
      end
   end
   winImg = image.display{image = seqOfFrames,
                          legend = 'Original Frames / Predicted Frames',
                          nrow = batches, win = winImg}
   print("Press any key to visualize next sequence ...")
   io.read()
end
