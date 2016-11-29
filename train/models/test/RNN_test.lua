--------------------------------------------------------------------------------
-- Testing script for RNN.lua
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 16
--------------------------------------------------------------------------------

local rnn = require '../RNN'
paths.mkdir('graphs')

-- RNN.getModel(channels, vis), c: channels
local net = rnn.getModel({upR = 32, R = 16}, true)

local upR, R, E, O, b

-- No batch
upR = torch.rand(32, 4, 4)
R   = torch.rand(16, 8, 8)
E   = torch.rand(16, 8, 8)

O = net:forward{upR, R, E}

graph.dot(net.fg, 'RNN-tensor', 'graphs/RNN-tensor')
print('Simple test: ' .. sys.COLORS.green .. 'pass')

-- Batch of 4
b   = 4
upR = torch.rand(b, 32, 4, 4)
R   = torch.rand(b, 16, 8, 8)
E   = torch.rand(b, 16, 8, 8)

O = net:forward{upR, R, E}

graph.dot(net.fg, 'RNN-batch', 'graphs/RNN-batch')
print('Batch test: ' .. sys.COLORS.green .. 'pass')
