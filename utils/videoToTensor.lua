require 'image'
require 'xlua'

-- Input video to be converted into a tensor
local input = '/media/HDD1/Datasets2/testVideos/highway/highway08.mp4'
-- Location to save the tensor
local saveTrain = './highway08Train.t7'
local saveTest  = './highway08Test.t7'
local trainTestRatio = 4                  -- train/test

-- Specify desired height and width of the dataset as well as the sequence length
local height = 128
local width = 256
local seqLength = 5

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

-- source height and width gets updated by __init based on the input video
frame:init(input, source)
local nFrames = frame.nFrames()          -- # of total frames in the video

local frameSeqTrain, frameSeqTest
local currentFrame = torch.FloatTensor(1, seqLength, 3, height, width):zero()

local img = frame.forward(img)
print("Maximum pixel value: " .. torch.max(img))
print("If needed then use this value to normalize your dataset after loading.")

--------------------------------------------------------------------------------
-- Section to convert image into tensor
local n = 1                 -- Counter for progress bar
local count = 1             -- Counter for how many frames have been added to one sequence
local checkTrain  = 0       -- Check if it is the very first seq for train data
local checkTest   = 0       -- Check if it is the very first seq for test data
local switchFlag  = 'train'
local switchCount = 1

while img do
   xlua.progress(n, nFrames)

   currentFrame[1][count] = image.scale(img[1], width, height)

   if count == seqLength then
      count = 1
      if switchFlag == 'train' then
         if checkTrain == 0 then                       -- When it is first seq then just put it in the output tensor
            checkTrain = 1
            frameSeqTrain = currentFrame:clone()
         else
            frameSeqTrain = frameSeqTrain:cat(currentFrame, 1)   -- Concat current seq to the output tensor
         end

         switchCount = switchCount + 1
         if switchCount > trainTestRatio then
            switchFlag = 'test'
         end
      else
         if checkTest == 0 then                       -- When it is first seq then just put it in the output tensor
            checkTest = 1
            frameSeqTest = currentFrame:clone()
         else
            frameSeqTest = frameSeqTest:cat(currentFrame, 1)   -- Concat current seq to the output tensor
         end

         switchCount = 1
         switchFlag = 'train'
      end
   else
      count = count + 1
   end

   img = frame.forward(img)
   n = n + 1
end

print("Conversion from video to tensor completed.")
print("\n# of training chunks created: " .. frameSeqTrain:size(1))
print("\n# of testing chunks created: " .. frameSeqTest:size(1))
print("Frame resolution is " .. height .. ' x ' .. width)
print("\nSaving tensor to location: " .. saveTrain)
torch.save(saveTrain, frameSeqTrain)
torch.save(saveTest,  frameSeqTest)
print("Tensor saved successfully!!!")
