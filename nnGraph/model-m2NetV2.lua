
require 'models.m2NetV2'

function createModel(opt, predNet)
-- Option for main
-- create graph
print('Creating connection and clonning:')

-- clone model through time-steps:
local clones = {}
for i = 1, opt.nSeq do
   clones[i] = predNet:clone('weight','bias','gradWeight','gradBias')
end
-- create model by connecting clones outputs and setting up global input:
-- inspired by: http://kbullaughey.github.io/lstm-play/rnn/
local E, C, R, E0, C0, R0, tUnit, P, xii, uInputs
E={} C={} R={} E0={} C0={} R0={} P={}
-- initialize inputs:
local xi = nn.Identity()()
for L=1, opt.nlayers do
   E0[L] = nn.Identity()()
   C0[L] = nn.Identity()()
   R0[L] = nn.Identity()()
   E[L] = E0[L]
   C[L] = C0[L]
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
      table.insert(uInputs, C[L])
      table.insert(uInputs, R[L])
   end
   -- clones inputs = {input_sequence, E_layer_1, R_layer_1, E_layer_2, R_layer_2, ...}
   tUnit = clones[i] ({ table.unpack(uInputs) }) -- inputs applied to clones
   -- connect clones:
   for L=1, opt.nlayers do
      if i < opt.nSeq then
         E[L] = { tUnit } - nn.SelectTable(4*L-3) -- connect output E to prev E of next clone
         C[L] = { tUnit } - nn.SelectTable(4*L-2) -- connect output E to prev E of next clone
         R[L] = { tUnit } - nn.SelectTable(4*L-1) -- connect output R to same layer E of next clone
      else
         P[L] = { tUnit } - nn.SelectTable(4*L) -- select Ah output as output of network
      end
   end
end
local inputs = {}
local outputs = {}
for L=1, opt.nlayers do
   table.insert(inputs, E0[L])
   table.insert(inputs, C0[L])
   table.insert(inputs, R0[L])
   table.insert(outputs, P[L])
end
table.insert(inputs, xi)
if opt.nlayers > 1 then
   outputs = {outputs-nn.SelectTable(1)}
end
model = nn.gModule(inputs, outputs ) -- output is P_layer_1 (prediction / Ah)
criterion = nn.MSECriterion()
   return criterion,  model
end
