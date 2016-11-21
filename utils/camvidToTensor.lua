--------------------------------------------------------------------------------
-- Load Camvid dataset and save it as:
-- trainData.t7 and testData.t7
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'image'
require 'xlua'

-- Dataset path
local datapath = '/media/HDD1/Datasets2/originalDatasets/CamVid/SegNet/CamVid'
-- Destination path where tensor will be saved
local cachepath = '/media/HDD1/Models/MatchNet/'

local imHeight  = 128
local imWidth   = 192
local seqLength = 5

local trainFile = datapath .. '/train.txt'
local testFile  = datapath .. '/test.txt'

local cacheDir  = paths.concat(cachepath, 'camVid')
local trainPath = paths.concat(cachepath, 'camVid', 'trainData.t7')
local testPath  = paths.concat(cachepath, 'camVid', 'testData.t7')

local red      = '\27[41m'
local green    = '\27[32m'
local resetCol = '\27[0m'

if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

-- Function to read txt file and return image and ground truth path
local function getPath(filepath)
   print("Extracting file names from: " .. filepath)
   local file = io.open(filepath, 'r')
   local imgPath = {}
   local gtPath = {}
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      col1 = datapath .. col1
      col2 = datapath .. col2
      table.insert(imgPath, col1)
      table.insert(gtPath, col2)
      fline = file:read()
   end
   return imgPath, gtPath
end

local function imgsToTensor(imgList, seqLength, imHeight, imWidth, destPath)
   local count = 1               -- Counter for how many frames have been added to one sequence
   local imgPath, gtPath = getPath(imgList)

   local prevImgSet = string.sub(imgPath[1], 1, -12)
   local rawImg, frameSeq
   local currentFrame = torch.FloatTensor(1, seqLength, 3, imHeight, imWidth):zero()

   for i = 1, #imgPath do
      -- load original image
      rawImg = image.load(imgPath[i])
      local currentImgSet = string.sub(imgPath[i], 1, -12)

      -- Ensure that images from same set of scene are in one sequence
      if currentImgSet ~= prevImgSet then
         count = 1
      end

      currentFrame[1][count] = image.scale(rawImg, imWidth, imHeight)

      if count == seqLength then
         count = 1
         if frameSeq then        -- Concat current seq to the output tensor
            frameSeq = frameSeq:cat(currentFrame, 1)
         else                    -- When it is first seq then just put it in the output tensor
            frameSeq = currentFrame:clone()
         end
      else
         count = count + 1
      end
      prevImgSet = currentImgSet

      -- -- load corresponding ground truth
      -- rawImg = image.load(gtPath[i], 1, 'byte'):squeeze():float() + 2
      -- local mask = rawImg:eq(13):float()
      -- rawImg = rawImg - mask * 12

      xlua.progress(i, #imgPath)
      collectgarbage()
   end
   print("Conversion from video to tensor completed.")
   print("# of chunks created: " .. frameSeq:size(1))
   print("Saving tensor to location: " .. destPath)
   torch.save(destPath, frameSeq)
   print(green .. "Tensor saved successfully!!!" .. resetCol)

   return frameSeq
end

print(green .. "Frame resolution is " .. 3 .. 'x' .. imHeight .. 'x' .. imWidth .. resetCol)
local trainData, testData
if paths.filep(testPath) then
   print('Loading cache data')
   trainData = torch.load(trainPath)
   testData  = torch.load(testPath)
   assert(trainData ~= nil, 'No trainData')
   assert(testData ~= nil, 'No testData')

   collectgarbage()
else
   --------------------------------------------------------------------------------
   -- Section to convert images into tensor
   print(red .. "\n==> Loading traning data" .. resetCol)
   trainData = imgsToTensor(trainFile, seqLength, imHeight, imWidth, trainPath)

   print(red .. "\n==> Loading testing data" .. resetCol)
   testData  = imgsToTensor(testFile, seqLength, imHeight, imWidth, testPath)
end
