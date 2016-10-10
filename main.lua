-- Main function
--
-- Abhishek Chaurasia
--------------------------------------------------------------------------------

require 'nn'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(9)

local opts = require 'opts'
local prednet = require 'prednet'

-- Gather all the arguments
local opt = opts.parse(arg)
opt.res = 64

prednet:__init(opt)

prednet:getModel()
