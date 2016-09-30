-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- Chainer implementation conversion based on: https://github.com/quadjr/PredNet/blob/master/net.py

require 'nn'
require 'nngraph'
require 'image'
require 'model-m2NetV2'
require 'train'
local m = require 'opts'
-- option is always gloable
opt = m.parse(arg)
nSeq = 3
nlayers = opt.nlayers
-- Option for lstm
local clOpt = {}
clOpt['nSeq'] = nSeq
clOpt['kw'] = 3
clOpt['kh'] = 3
clOpt['st'] = 1
clOpt['pa'] = 1
clOpt['dropOut'] = 0
clOpt['lm'] = 1
local input_stride = 1
local poolsize = 2
local inputImsize = 64

local imSize   ={64,32}
local channels = {1, 32} -- layer maps sizes
local prevE  = {channels[1]*2,channels[2]*2}
local cellCh = {32,prevE[1]} --  Out put size of lstm -- This is same as output channels
local lstmCh = {cellCh[2]+prevE[1],prevE[2]} --  Out put size of lstm -- last chnel has no R_l+1 concat
clOpt['cellCh'] = cellCh
clOpt['lstmCh'] = lstmCh
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
nngraph.setDebug(true)
opt.clOpt = clOpt
opt.poolsize = poolsize
opt.channels = channels
opt.prevE = prevE
opt.cellCh = cellCh
opt.lstmCh = lstmCh
opt.imSize = imSize

local function main()
   --cutorch.setDevice(1)
   --Create model cloned inside
   train(opt)


end
main()
