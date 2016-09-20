require 'nn'
require 'rnn'
require 'UntiedConvLSTM'

model = nn.Sequential()


-- Encoder
local encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(opt.nFilters[1], opt.nFilters[2], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))
encoder:add(nn.Tanh())
encoder:add(nn.SpatialMaxPooling(2,2,2,2))

-- Decoder, mirror of the encoder, but without non-linearity 
local decoder = nn.Sequential()
decoder:add(nn.SpatialUpSamplingNearest(2)) 
decoder:add(nn.SpatialConvolution(opt.nFilters[2], opt.nFilters[1], opt.kernelSize, opt.kernelSize, 1, 1, opt.padding, opt.padding))


local seqe = nn.Sequencer(encoder)
seqe:remember('both')
seqe:training()
model:add(seqe)

-- memory branch
-- local memory_branch = nn.Sequential()
local seq = nn.Sequencer(nn.ConvLSTM(opt.nFiltersMemory[1],opt.nFiltersMemory[2], opt.nSeq, opt.kernelSize, opt.kernelSizeMemory, opt.stride))
seq:remember('both')
seq:training()
model:add(seq)
model:add(nn.SelectTable(opt.nSeq))

-- add spatial decoder
model:add(decoder)


-- loss module: penalize difference of gradients
local gx = torch.Tensor(3,3):zero()
gx[2][1] = -1
gx[2][2] =  0
gx[2][3] =  1
gx = gx:cuda()
local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
gradx.weight:copy(gx)
gradx.bias:fill(0)

local gy = torch.Tensor(3,3):zero()
gy[1][2] = -1
gy[2][2] =  0
gy[3][2] =  1
gy = gy:cuda()
local grady = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
grady.weight:copy(gy)
grady.bias:fill(0)

local gradconcat = nn.ConcatTable()
gradconcat:add(gradx):add(grady)

gradloss = nn.Sequential()
gradloss:add(gradconcat)
gradloss:add(nn.JoinTable(1))

criterion = nn.MSECriterion()
--criterion.sizeAverage = false

-- send everything to GPU
model:cuda()
gradloss:cuda()
criterion:cuda()
