require 'image'
require 'xlua'

-- Input video to be converted into a tensor
local input = '/media/HDD1/Datasets2/testVideos/highway/highway08.mp4'
-- Location to save the tensor
local save = './highway08.t7'

-- Specify desired height and width of the dataset as well as the sequence length
local height = 128
local width = 256
local seqLength = 5

local camRes = 'QHD'
local fps = 30

--------------------------------------------------------------------------------
-- Initialize class Frame which can be used to read videos/camera
local frame = assert(require('framevideo'))

-- TODO Get rid of this part
-- To do so you will have to modify framevideo.lua
local source = {}
-- switch input sources
source.res = {
   HVGA  = {w =  320, h =  240},
   QHD   = {w =  640, h =  360},
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}
source.w = source.res[camRes].w
source.h = source.res[camRes].h
source.fps = fps

-- source height and width gets updated by __init based on the input video
frame:init(input, source)
local nFrames = frame.nFrames()          -- # of total frames in the video

local frameSeq
local currentFrame = torch.FloatTensor(1, seqLength, 3, height, width):zero()

local img = frame.forward(img)
print("Maximum pixel value: " .. torch.max(img))
print("If needed then use this value to normalize your dataset after loading.")

--------------------------------------------------------------------------------
-- Section to convert image into tensor
local n = 1                -- Counter for progress bar
local count = 1            -- Counter for how many frames have been added to one sequence
local check = 0            -- Check if it is the very first seq

while img do
   xlua.progress(n, nFrames)
   n = n + 1

   currentFrame[1][count] = image.scale(img[1], width, height)

   if count == seqLength then
      count = 1
      if check == 0 then                        -- When it is first seq then just put it in the output tensor
         check = 1
         frameSeq = currentFrame:clone()
      else
         frameSeq = frameSeq:cat(currentFrame, 1)   -- Concat current seq to the output tensor
      end
   else
      count = count + 1
   end

   img = frame.forward(img)
end

print("Conversion from video to tensor completed.")
print("\n# of frames in videos are: " .. frameSeq:size(1))
print("Frame resolution is " .. height .. ' x ' .. width)
print("\nSaving tensor to location: " .. save)
torch.save(save, frameSeq)
print("Tensor saved successfully!!!")
