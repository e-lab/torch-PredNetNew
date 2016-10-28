--Sangpil Kim, Eugenio Culurciello
-- with help from Alfredo Canziani and Abhishek Chaurasia
-- August - September 2016
-- PredNet in Torch7 - from: https://arxiv.org/abs/1605.08104
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
-- download data from: https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa?dl=1
--
-------------------------------------------------------------------------------

require 'nn'
require 'paths'
require 'torch'
require 'image'
require 'optim'
require 'xlua'
require 'pl'
local of = require 'opt'
local opt = of.parse(arg)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
torch.save(paths.concat(opt.savedir,'opt.t7'),opt)
print(opt)
print('Using GPU?', opt.useGPU)
print('GPU id?', opt.gpuId)
print('Batch size?', opt.batch)
print('How many layers?' ,opt.nlayers)
print('Keep mode?' ,opt.modelKeep)

--Call files
local U = require 'misc/util'
local loader = require 'misc/data'
local atari  = require 'misc/atari'
local M = require 'models/model'
local Tr= require 'train'
local Te= require 'test'

local util  = U(opt)
local initM = M(opt)
local tr    = Tr(opt)
local te
if not opt.trainOnly then te = Te(opt) end
local models = initM:getModel()

local function main()
   if not opt.atari then
   end
   --Main loop
   for epoch = 1 , opt.maxEpochs do
      tr:train(util, epoch, models)
      if not opt.trainOnly then te:test(util, epoch, models[1]) end
      collectgarbage()
   end
end

main()
