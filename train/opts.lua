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
   --lstmLayer    (default 1)

   ## Dataset
   --datapath     (default /media/) path for train and test data
   --batch        (default 5)          number of batch size
   --shuffle                           shuffle training data set

   ## Model
   --layers       (default 3)          number of layers in the model
   --seq          (default 5)          number of sequence to look at
   --save         (default ./media/)   save model at this location
   --model        (default pred)       choose among [pred|PCBC]
   --recursion    (string)             specify recursion [rnn|lstm]

   ## Graph
   --saveGraph                         save the graphs

   ## Device
   --dev          (default cuda)       Device to be used: cpu/cuda
   --devID        (default 1)          GPU number
   --nGPU         (default 2)          Number GPUs to be used

   ## Miscellaneous parameters
   --seed         (default 8)
   --display                           Display predictions while training/plots
   ]]

   return opt
end

return opts
