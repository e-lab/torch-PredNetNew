-- Main function
--
-- Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

-- Gather all the arguments
local opts = require 'opts'
local opt = opts.parse(arg)

if opt.dev == 'cuda' then
   require 'cunn'
   require 'cudnn'
end

torch.manualSeed(opt.seed)

local prednet = require 'prednet'
local train = require 'train'

-- Input/Output channels for A of every layer
-- XXX Change value of channels[1] to # of input image channels
opt.channels = torch.ones(opt.layers + 1)
opt.channels[1] = 3
for l = 2, opt.layers + 1 do
   opt.channels[l] = 2^(l+3)
end
-- {1, 32, 64, 128, 256, 512}

-- Sequence and resolution information of data
-- is added to 'opt' during this initialization.
-- It also call the model generator and returns
-- the model prototype
local prototype = train:__init(opt)

-- Logger
logger = optim.Logger('error.log')
logger:setNames{'Training Error', 'Interframe Training Error'}

print("\nTRAINING\n")
local prevTrainError = 10000

for epoch = 1, opt.nEpochs do
   print("Epoch: ", epoch)
   local trainError, interFrameError = train:updateModel()
   logger:add{trainError, interFrameError}
   logger:style{'+-', '+-'}
   logger:plot()

   -- Save the trained model
   if prevTrainError > trainError then
      local saveLocation = opt.save .. 'model-' .. epoch .. '.net'
      prototype:evaluate()
      torch.save(saveLocation, prototype)
      prevTrainError = trainError
   end
end
