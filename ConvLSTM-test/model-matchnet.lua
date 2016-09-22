require 'nn'
-- require 'rnn'
require 'MatchNet'

-- nngraph.setDebug(true)

local nlayers = 1
local input_stride = 1
local poolsize = 2

-- instantiate MatchNet:
local unit = mNet(nlayers, input_stride, poolsize, opt.nFilters, {opt.nSeq, opt.stride}, false) -- false testing mode
-- nngraph.annotateNodes()
-- graph.dot(unit.fg, 'MatchNet-unit','Model-unit') -- graph the model!

-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
end

-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E0 = nn.Identity()()
local R0 = nn.Identity()()
local xi = nn.Identity()()
local tUnit, yo, xii
local E = E0
local R = R0
for i = 1, opt.nSeq-1 do
   xii = {xi} - nn.SelectTable(i)
   tUnit = clones[i]({ xii, E, R })
   E = { tUnit } - nn.SelectTable(1) -- connect output E to prev E of next clone
   R = { tUnit } - nn.SelectTable(2) -- connect output R to same layer E of next clone
end
yo = { clones[opt.nSeq]({ {xi} - nn.SelectTable(opt.nSeq), E, R }) } - nn.SelectTable(3) -- select Ah output of first layer as output of network
model = nn.gModule( {E0, R0, xi}, {yo} )
-- nngraph.annotateNodes()
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


-- test overall model
local inTable = {}
local inSeqTable = {}
for i = 1, opt.nSeq do table.insert( inSeqTable,  torch.ones( opt.nFilters[1], opt.inputSizeW, opt.inputSizeW) ) end -- input sequence
table.insert( inTable, torch.zeros( opt.nFilters[1], opt.inputSizeW, opt.inputSizeW) ) -- same layer E
table.insert( inTable, torch.zeros( opt.nFilters[2], opt.inputSizeW, opt.inputSizeW) ) -- same layer R
table.insert( inTable,  inSeqTable ) -- input sequence
local outTable = model:updateOutput(inTable)
print('Model output is: ', outTable:size())
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


-- loss module: penalize difference of gradients
local gx = torch.Tensor(3,3):zero()
gx[2][1] = -1
gx[2][2] =  0
gx[2][3] =  1
gx = gx--:cuda()
local gradx = nn.SpatialConvolution(1,1,3,3,1,1,1,1)
gradx.weight:copy(gx)
gradx.bias:fill(0)

local gy = torch.Tensor(3,3):zero()
gy[1][2] = -1
gy[2][2] =  0
gy[3][2] =  1
gy = gy--:cuda()
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
-- model:cuda()
-- gradloss:cuda()
-- criterion:cuda()
