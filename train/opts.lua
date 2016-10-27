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
   ## Hyperpatameters
   --lr           (default 0.001)
   --nEpochs      (default 100)

   ## Dataset
   --datapath     (default /media/tensor.t7)

   ## Model
   --layers       (default 3)          number of layers in the model
   --seq          (default 5)          number of sequence to look at
   --save         (default /media/)    save model at this location

   ## Graph
   --saveGraph                         save the graphs

   ## Device
   --dev          (default cuda)       Device to be used: cpu/cuda
   --devID        (default 1)          GPU number

   ## Miscellaneous parameters
   --seed         (default 8)
   --display                           Display predictions while training/plots
   ]]

   return opt
end

return opts

