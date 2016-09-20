require 'nn'
-- require 'rnn'
require 'MatchNet'

local nlayers = 1
local input_stride = 1
local poolsize = 2
local mapss = {3, 32, 64, 128, 256}

-- instantiate MatchNet:
local unit = mNet(nlayers, input_stride, poolsize, mapss, {opt.nSeq, opt.stride}, false) -- false testing mode

-- tests:
-- local inTable = {}
-- table.insert( inTable, torch.Tensor(mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- prev E
-- table.insert( inTable, torch.zeros( mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- this E
-- -- output is model[1].output[3]
-- local outTable = unit:forward(inTable)
-- print('Output is: ')
-- print(outTable)
-- print('output', unit.outnode.children[3])
-- print('input', unit.innode)


-- clone model through time-steps:
dofile('utils.lua')
local clones = clone_many_times(unit, opt.nSeq)


-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
h0 = nn.Identity()()
x = nn.Identity()()
xs = nn.SplitTable(2)(x)
h = h0
for i = 1, opt.nSeq do
  h = clones[i] ( { h, nn.SelectTable(i) (xs) } )
end
y =  nn.SelectTable(1)(h)
model = nn.gModule( {h0,x}, {y} )
print(model)

local inTable = {}
table.insert( inTable, torch.Tensor(mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- prev E
table.insert( inTable, torch.zeros( mapss[1], opt.inputSizeW, opt.inputSizeW) ) -- this E
local outTable = unit:forward(inTable)
print('Output is: ')
print(outTable)



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
