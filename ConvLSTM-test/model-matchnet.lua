require 'nn'
-- require 'rnn'
require 'MatchNet'

nngraph.setDebug(true)

local nlayers = 1
local input_stride = 1
local poolsize = 2

-- instantiate MatchNet:
local unit = mNet(nlayers, input_stride, poolsize, opt.nFilters, {opt.nSeq, opt.stride}, false) -- false testing mode
nngraph.annotateNodes()
graph.dot(unit.fg, 'MatchNet-unit','Model-unit') -- graph the model!

-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
end

-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E, R, E0, R0, tUnit, yo, xii, uInputs
E={} R={} E0 ={} R0={} yo={}
for L=1, nlayers do
   E0[L] = nn.Identity()()
   R0[L] = nn.Identity()()
end
local xi = nn.Identity()()
for L=1, nlayers do
   E[L] = E0[L]
   R[L] = R0[L]
   for i = 1, opt.nSeq do
      xii = {xi} - nn.SelectTable(i)
      -- table.insert(uInputs, xii)
      -- clones input = {in, E, R, nR, E, R, nR ....} nR only if there is another layer after it
      tUnit = clones[i]({ xii, E[L], R[L] })
      if i < opt.nSeq then 
         E[L] = { tUnit } - nn.SelectTable(3*L-2) -- connect output E to prev E of next clone
         R[L] = { tUnit } - nn.SelectTable(3*L-1) -- connect output R to same layer E of next clone
      else
         yo[L] = { tUnit } - nn.SelectTable(3*L) -- select Ah output of first layer as output of network
      end
   end
end
model = nn.gModule( {table.unpack(E0), table.unpack(R0), xi}, {table.unpack(yo)} ) -- only care about layer 1 output here
nngraph.annotateNodes()
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!
-- local E, R, E0, R0, tUnit, yo, xii, uInputs
-- E={} R={} E0 ={} R0={} --yo={}
-- for L=1, nlayers do
--    E0[L] = nn.Identity()()
--    R0[L] = nn.Identity()()
-- end
-- local xi = nn.Identity()()
-- for i = 1, opt.nSeq do
--    uInputs={}
--    xii = {xi} - nn.SelectTable(i)
--    table.insert(uInputs, xii)
--    for L=1, nlayers do
--       E[L] = E0[L]
--       R[L] = R0[L]
--       table.insert(uInputs, E[L])
--       table.insert(uInputs, R[L])
--    end
--    tUnit = clones[i]({table.unpack(uInputs)})
--    for L=1, nlayers do
--       -- clones input = {in, E, R, nR, E, R, nR ....} nR only if there is another layer after it
--       E[L] = { tUnit } - nn.SelectTable(3*L-2) -- connect output E to prev E of next clone
--       R[L] = { tUnit } - nn.SelectTable(3*L-1) -- connect output R to same layer E of next clone
--    end
-- end
-- uInputs={}
-- xii = {xi} - nn.SelectTable(opt.nSeq)
-- table.insert(uInputs, xii)
-- for L=1, nlayers do
--    table.insert(uInputs, E[L])
--    table.insert(uInputs, R[L])
-- end
-- yo = { clones[opt.nSeq]({table.unpack(uInputs)}) } - nn.SelectTable() -- select Ah output of first layer as output of network
-- model = nn.gModule( {table.unpack(E0), table.unpack(R0), xi}, {table.unpack(yo)} ) -- only care about layer 1 output here
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
graph.dot(model.fg, 'MatchNet','Model') -- graph the model!



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
