--------------------------------------------------------------------------------
-- Contains options required by main.lua
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   ## Dataset
   --input        (default /media/video.mp4)    Input folder/file location or cam0 for camera

   ## Model
   --dmodel       (default predNet)             model directory name
   --net          (default 1)                   model number
   --layers       (default 3)                   # of layers in the model

   ## Device
   --dev          (default cuda)                Device to be used: cpu/cuda
   --devID        (default 1)                   GPU number

   ## Miscellaneous parameters
   --seed         (default 8)
   -v                                           Verbose mode
   ]]

   return opt
end

return opts

