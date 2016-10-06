-- Main function
--
-- Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(9)

model = require 'model'

local opt = {}
opt.layers = 3
opt.seq = 1
opt.res = 64

model:__init(opt)
model:getModel()
