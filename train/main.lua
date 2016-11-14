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
   require 'cutorch'
   cutorch.setDevice(opt.devID)
end

torch.manualSeed(opt.seed)

local train = require 'train'
local test  = require 'test'

-- Input/Output channels for A of every layer
opt.channels = torch.ones(opt.layers + 1)
for l = 2, opt.layers + 1 do
   opt.channels[l] = 2^(l+3)
end
-- {1, 32, 64, 128, 256, 512}

-- Sequence and resolution information of data
-- is added to 'opt' during this initialization.
-- It also call the model generator and returns
-- the model prototype
local prototype = train:__init(opt)
test:__init(opt)

-- Logger
if not paths.dirp(opt.save) then paths.mkdir(opt.save) end
logger = optim.Logger(paths.concat(opt.save,'error.log'))
logger:setNames{'Prediction Error', 'Replica Error'}
logger:display(opt.display)
testlogger = optim.Logger(paths.concat(opt.save,'testerror.log'))
testlogger:setNames{'Test Prediction Error', 'Test Replica Error'}
testlogger:display(opt.display)

print("\nTRAINING\n")
local prevTrainError = 10000

for epoch = 1, opt.nEpochs do
   print("Epoch: ", epoch)
   local predError, replicaError = train:updateModel()
   local tpredError, treplicaError

   tpredError, treplicaError = test:updateModel(train.model)
   logger:add{predError, replicaError}
   logger:style{'+-', '+-'}
   logger:plot()

   testlogger:add{tpredError, treplicaError}
   testlogger:style{'+-', '+-'}
   testlogger:plot()


   -- Save the trained model
   if replicaError > predError then
      print('Save !')
      local saveLocation = opt.save .. 'model-' .. epoch .. '.net'
      prototype:float():clearState():evaluate()
      torch.save(saveLocation, prototype)
      if opt.dev == 'cuda' then  print('Back') train.model:cuda() end
      prevTrainError = predError
   end
end
