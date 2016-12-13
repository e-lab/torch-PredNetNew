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
   --input        (default /media/tensor.t7)    Input .t7 file location

   ## Model
   --dmodel       (default predNet)             model directory name
   --net          (default 1)                   model number
   --layers       (default 3)                   # of layers in the model
   --model        (default pred)                choose among [pred|PCBC]

   ## Device
   --dev          (default cuda)                Device to be used: cpu/cuda
   --devID        (default 1)                   GPU number

   ## Miscellaneous parameters
   --seed         (default 8)
   --nrow         (default 5)                   Images in one row
   -v                                           Verbose mode
   --save                                       Save views to file
   ]]

   return opt
end

return opts

