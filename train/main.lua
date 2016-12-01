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

local cp = {}                                   -- Color Pallete
cp.r     = '\27[31m'
cp.g     = '\27[32m'
cp.reset = '\27[0m'

local train = require 'train'
local test  = require 'test'

-- Input/Output channels for A of every layer
opt.channels = {}
opt.channels[0] = opt.model == 'PCBC'
for l = 1, opt.layers + 1 do
   opt.channels[l] = 2^(l + (opt.channels[0] and 4 or 3))
end
-- {[1]=1|3, 32, 64, 128, 256, 512} -> PredNet
-- {[0]=1|3, 32, 64, 128, 256, 512} -> PCBC

-- Sequence and resolution information of data
-- is added to 'opt' during this initialization.
-- It also call the model generator and returns
-- the model prototype
local prototype, w = train:__init(opt)
test:__init(opt)

-- Logger
if not paths.dirp(opt.save) then paths.mkdir(opt.save) end

local logger, testlogger
logger = optim.Logger(paths.concat(opt.save,'error.log'))
logger:setNames{'Train prd. error', 'Train rpl. error'} -- training prediction/replica
logger:style{'+-', '+-'}
logger:display(opt.display)

testlogger = optim.Logger(paths.concat(opt.save,'testerror.log'))
testlogger:setNames{'Test prd. error', 'Test rpl. error'} -- testing prd/rpl
testlogger:style{'+-', '+-'}
testlogger:display(opt.display)

print("\nTRAINING...")
local prevTrainError = 10000

for epoch = 1, opt.nEpochs do
   print(cp.r .. "\nEpoch: ", epoch .. cp.reset)
   local predError, replicaError = train:updateModel()
   local tpredError, treplicaError = test:updateModel(train.model)

   logger:add{predError, replicaError}
   logger:plot()

   testlogger:add{tpredError, treplicaError}
   testlogger:plot()

   -- Save the trained model
   if treplicaError > tpredError then
      local saveLocation = paths.concat(opt.save, 'model-' .. epoch .. '.net')
      w:copy(train.w)
      torch.save(saveLocation, prototype)
      print(cp.g .. 'Model saved!!!' .. cp.reset)

      prevTrainError = tpredError
   end
end
