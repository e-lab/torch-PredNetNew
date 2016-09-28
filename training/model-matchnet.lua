-- MatchNet training: predicting future frames in video
-- Eugenio Culurciello, August - September 2016
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
--

require 'nn'
require 'MatchNet'

-- nngraph.setDebug(true)

-- instantiate MatchNet:
local unit = mNet(opt.nlayers, opt.stride, opt.poolsize, opt.nFilters, {opt.nSeq, opt.stride}, false) -- false testing mode
-- nngraph.annotateNodes()
-- graph.dot(unit.fg, 'MatchNet-unit','Model-unit') -- graph the model!


-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = unit:clone('weight','bias','gradWeight','gradBias')
end


-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E, R, E0, R0, tUnit, P, xii, uInputs
E={} R={} E0={} R0={} P={}
-- initialize inputs:
local xi = nn.Identity()()
for L=1, opt.nlayers do
   E0[L] = nn.Identity()()
   R0[L] = nn.Identity()()
   E[L] = E0[L]
   R[L] = R0[L]
end
-- create model as combination of units:
for i=1, opt.nSeq do
   -- set inputs to clones:
   uInputs={}
   xii = {xi} - nn.SelectTable(i) -- select i-th input from sequence
   table.insert(uInputs, xii)
   for L=1, opt.nlayers do
      table.insert(uInputs, E[L])
      table.insert(uInputs, R[L])
   end
   -- clones inputs = {input_sequence, E_layer_1, R_layer_1, E_layer_2, R_layer_2, ...}
   tUnit = clones[i] ({ table.unpack(uInputs) }) -- inputs applied to clones
   -- connect clones:
   for L=1, opt.nlayers do
      if i < opt.nSeq then
         E[L] = { tUnit } - nn.SelectTable(3*L-2) -- connect output E to prev E of next clone
         R[L] = { tUnit } - nn.SelectTable(3*L-1) -- connect output R to same layer E of next clone
      else
         P[L] = { tUnit } - nn.SelectTable(3*L) -- select Ah output as output of network
      end
   end
end
local inputs = {}
local outputs = {}
for L=1, opt.nlayers do
   table.insert(inputs, E0[L])
   table.insert(inputs, R0[L])
   table.insert(outputs, P[L])
end
table.insert(inputs, xi)
if opt.nlayers > 1 then
   outputs = {outputs-nn.SelectTable(1)}
end
model = nn.gModule(inputs, outputs ) -- output is P_layer_1 (prediction / Ah)
-- nngraph.annotateNodes()
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!



-- test overall model
print('Testing model')
local inTable = {}
local inSeqTable = {}
for i = 1, opt.nSeq do table.insert( inSeqTable,  torch.ones( opt.nFilters[1], opt.inputSizeW, opt.inputSizeW) ) end -- input sequence
for L=1, opt.nlayers do
   table.insert( inTable, torch.zeros(2*opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1)))-- previous time E
   if L==1 then 
      table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
   else
      table.insert( inTable, torch.zeros(opt.nFilters[L], opt.inputSizeW/2^(L-1), opt.inputSizeW/2^(L-1))) -- previous time R
   end
end
table.insert( inTable,  inSeqTable ) -- input sequence
local outTable = model:updateOutput(inTable)
print('Model output is: ', outTable:size())
-- graph.dot(model.fg, 'MatchNet','Model') -- graph the model!


criterion = nn.MSECriterion()

-- send everything to GPU
if opt.useGPU then
   model:cuda()
   criterion:cuda()
end