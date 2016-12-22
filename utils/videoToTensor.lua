require 'image'
require 'xlua'

-- Input video to be converted into a tensor
local dirRoot = '/Users/abhi/Documents/Workspace/Dataset/'
-- Location to save the tensor
local cachepath = './'

local cacheDir  = paths.concat(cachepath, 'vid2Ten')
local saveTrain = paths.concat(cachepath, 'vid2Ten', 'trainData.t7')
local saveTest  = paths.concat(cachepath, 'vid2Ten', 'testData.t7')

local red      = '\27[31m'
local green    = '\27[32m'
local resetCol = '\27[0m'

if not paths.dirp(cacheDir) then paths.mkdir(cacheDir) end

-- Specify desired height and width of the dataset as well as the sequence length
local imHeight = 128
local imWidth = 192
local seqLength = 5
local trainRatio = 0.8                  -- train/total data

-- Function to check if the given file is a valid video
local function validVideo(filename)
   local ext = string.lower(path.extension(filename))

   local videoExt = {'.avi', '.mp4', '.mxf'}
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
local frame = assert(require('framevideo'))

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

local frameSeqTrain, frameSeqTest
local function videoToTensor(input)
   -- source height and width gets updated by __init based on the input video
   frame:init(input, source)
   local nFrames = frame.nFrames() or 2000         -- # of total frames in the video

   local currentFrame = torch.FloatTensor(1, seqLength, 3, imHeight, imWidth):zero()

   local img = frame.forward(img)
   local trc = trainRatio * nFrames                -- # of training frames
   --------------------------------------------------------------------------------
   -- Section to convert image into tensor
   local n = 1                 -- Counter for progress bar
   local count = 1             -- Counter for how many frames have been added to one sequence

   while img and n <= nFrames do
      xlua.progress(n, nFrames)

      currentFrame[1][count] = image.scale(img[1], imWidth, imHeight)

      if count == seqLength then
         count = 1
         if n <= trc then
            if frameSeqTrain then   -- Concat current seq to the output tensor
               frameSeqTrain = frameSeqTrain:cat(currentFrame, 1)
            else                    -- When it is first seq then just put it in the output tensor
               frameSeqTrain = currentFrame:clone()
            end
         else
            if frameSeqTest then   -- Concat current seq to the output tensor
               frameSeqTest = frameSeqTest:cat(currentFrame, 1)
            else                    -- When it is first seq then just put it in the output tensor
               frameSeqTest = currentFrame:clone()
            end
         end
      else
         count = count + 1
      end

      img = frame.forward(img)
      n = n + 1
   end
end

print(green .. "Frame resolution is " .. 3 .. 'x' .. imHeight .. 'x' .. imWidth .. resetCol)
local trainData, testData
if paths.filep(saveTrain) then
   print('Loading cache data')
   trainData = torch.load(saveTrain)
   testData  = torch.load(saveTest)
   assert(trainData ~= nil, 'No trainData')
   assert(testData ~= nil, 'No testData')

   collectgarbage()
else
   --------------------------------------------------------------------------------
   -- Section to convert images into tensor
   assert(paths.dirp(dirRoot), 'No folder found at: ' .. dirRoot)
   -- load train/test images:
   for file in paths.iterfiles(dirRoot) do
      print(red .. "\nLoading traning & testing data using file: " .. resetCol .. file)
      -- process each image
      if validVideo(file) then
         local imgPath = path.join(dirRoot, file)
         -- print(imgPath)
         videoToTensor(imgPath)
         print(green .. "\nLoaded traning & testing data!!!" .. resetCol)
      end
   end
end

print("Conversion from video to tensor completed.")
print("\n# of training chunks created: " .. frameSeqTrain:size(1))
print("\n# of testing chunks created: " .. frameSeqTest:size(1))
print("\nSaved train data at: " .. saveTrain)
print("\nSaved test  data at: " .. saveTest)
torch.save(saveTrain, frameSeqTrain)
torch.save(saveTest,  frameSeqTest)
print(green .. "Tensor saved successfully!!!" .. resetCol)
