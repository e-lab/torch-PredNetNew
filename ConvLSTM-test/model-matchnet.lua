require 'nn'
-- require 'rnn'
require 'MatchNet'

local nlayers = 1
local input_stride = 1
local poolsize = 2
local mapss = {1, 32, 64, 128, 256}

-- instantiate MatchNet:
local unit = mNet(nlayers, input_stride, poolsize, mapss, {opt.nSeq, opt.stride}, false) -- false testing mode
nngraph.annotateNodes()
-- nngraph.setDebug(true)


-- tests:
local inTable = {}
table.insert( inTable, torch.Tensor(mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- prev E
table.insert( inTable, torch.zeros( mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- this E
-- output is model[1].output[3]
local outTable = unit:forward(inTable)
print('Output test of one unit is: ')
print(outTable)
-- print('output', unit.outnode.children[3])
-- print('input', unit.innode)


-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
end

-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E0 = nn.Identity()()
local xi = nn.Identity()()
local tUnit, yo
-- xs = {xi} - nn.SplitTable(2)
E = E0
opt.nSeq = 1
for i = 1, opt.nSeq do
   tUnit = clones[i] ({ {xi} - nn.SelectTable(i), E })
   E = {tUnit} - nn.SelectTable(1) -- connect output E to prev E of next clone
end
yo = {tUnit} - nn.SelectTable(3) -- select Ah output of first layer as output of network
model = nn.gModule( {E0,xi}, {yo} )
nngraph.annotateNodes()
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
-- print(model)

local inTable = {}
local inSeqTable = {}
for i=1,1 do table.insert( inSeqTable,  torch.ones( mapss[1], opt.inputSizeW, opt.inputSizeW) ) end -- input sequence
table.insert( inTable,  inSeqTable ) -- input sequence
table.insert( inTable, torch.zeros( mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- h0 (this E coming in)
local outTable = model:updateOutput(inTable)
print('Output is: ')
print(outTable)
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!



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
-- model:cuda()
-- gradloss:cuda()
-- criterion:cuda()
