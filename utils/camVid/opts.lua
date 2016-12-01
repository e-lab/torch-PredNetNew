--------------------------------------------------------------------------------
-- Contains options required by run.lua
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   Device Related:
   -i,--devid              (default 1)           device ID (if using CUDA)
   --dev                   (default cuda)        cpu/cuda
   --seed                  (default 9)

   Dataset Related:
   --save                  (default /media/)     location to output tensors
   --datapath              (default /media/)     dataset location
   --imHeight              (default 360)         image height  (360 cv/256 cs/256 su)
   --imWidth               (default 480)         image width   (480 cv/512 cs/328 su)
   --seqLength             (default 5)           sequence length
   --trteRatio             (default 4)           train/test

   Model Related:
   --dmodel                (default /Models/)    path of a model
   --net                   (default 1)           model number
   --classifier            (default /Models/)    path of classifier
   --layers                (default 3)           number of layers in the model
 ]]

   return opt
end

return opts
