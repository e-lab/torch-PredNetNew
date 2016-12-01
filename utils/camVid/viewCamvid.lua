require 'image'
require 'xlua'

-- Input video to be converted into a tensor
local dirRoot = '/media/HDD1/Datasets2/originalDatasets/CamVid/large/'

local red      = '\27[31m'
local green    = '\27[32m'
local resetCol = '\27[0m'

-- Specify desired height and width of the dataset as well as the sequence length
local imHeight = 360
local imWidth  = 480

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
local maxSampleFrames = { 123, 169, 100, 304}
local frameSeqTrain, frameSeqTest
local function videoToTensor(input, dirN)
   -- source height and width gets updated by __init based on the input video
   frame:init(input, source)
   local nFrames = frame.nFrames()            -- # of total frames in the video

   local currentFrame = torch.FloatTensor(3, imHeight, imWidth):zero()
   local frameToDisp  = torch.FloatTensor(3, 3, imHeight, imWidth):zero()

   local img = frame.forward(img)

   --------------------------------------------------------------------------------
   -- Section to convert image into tensor
   local n = - labelOffset[dirN]          -- Counter for progress bar
   local count = 0                        -- Counter for how many frames have been added to one sequence
   local switchFlag  = 'train'
   local switchCount = 1
   local labelPath
   local labelPrefix = labelPrefixTable[dirN]
   while count < maxSampleFrames[dirN] do
      xlua.progress(count, maxSampleFrames[dirN])
      frameToDisp[1] = image.scale(img[1], imWidth, imHeight)

      frameToDisp[2]:zero()
      local labelIdx = n + labelStart[dirN]
      labelPath = dirRoot .. tostring(dirN) .. '/label/' .. labelPrefix .. string.format('%05d_L.png', labelIdx)
      if paths.filep(labelPath) then
         local currentLabel = image.load(labelPath, 3, 'byte'):float()/255
         frameToDisp[2] = image.scale(currentLabel, imWidth, imHeight)
         count = count + 1
      end
      frameToDisp[3] = frameToDisp[1] + frameToDisp[2]

      if (labelStart[1] - n) % 30 == 0 and n > 0 then
         win = image.display{image = frameToDisp, legend = 'Image / Label / Image+Label', win = win}
         -- io.read()
      end
      img = frame.forward(img)
      n = n + 1
   end
end

print(green .. "Frame resolution is " .. 3 .. 'x' .. imHeight .. 'x' .. imWidth .. resetCol)

--------------------------------------------------------------------------------
assert(paths.dirp(dirRoot), 'No folder found at: ' .. dirRoot)
-- load train/test images:
for dirN = 1, 4 do
   local dirPath = dirRoot .. dirN .. '/input/'
   for file in paths.iterfiles(dirPath) do
      print(red .. "\nLoading traning & testing data using file: " .. resetCol .. file)
      -- process each image
      if validVideo(file) then
         local imgPath = dirPath .. file
         -- print(imgPath)
         videoToTensor(imgPath, dirN)
         print(green .. "\nLoaded traning & testing data!!!" .. resetCol)
      end
   end
end
print(green .. "No more labeled CamVid data!!!" .. resetCol)
