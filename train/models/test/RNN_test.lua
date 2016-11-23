--------------------------------------------------------------------------------
-- Testing script for RNN.lua
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 16
--------------------------------------------------------------------------------

local rnn = require '../RNN'
paths.mkdir('graphs')

-- RNN.getModel(cRup, cR, cE, vis), c: channels
local net = rnn.getModel(32, 16, 6, true)

local Rup, R, E, O, b

-- No batch
Rup = torch.rand(32, 4, 4)
R   = torch.rand(16, 8, 8)
E   = torch.rand( 6, 8, 8)

O = net:forward{Rup, R, E}

graph.dot(net.fg, 'RNN-tensor', 'graphs/RNN-tensor')
print('Simple test: ' .. sys.COLORS.green .. 'pass')

-- Batch of 4
b   = 4
Rup = torch.rand(b, 32, 4, 4)
R   = torch.rand(b, 16, 8, 8)
E   = torch.rand(b,  6, 8, 8)

O = net:forward{Rup, R, E}

graph.dot(net.fg, 'RNN-batch', 'graphs/RNN-batch')
print('Batch test: ' .. sys.COLORS.green .. 'pass')
