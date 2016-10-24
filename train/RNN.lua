-- Generates RNN block
-- Inputs:  {Rl+1, Rl[t-1], El[t-1]}
-- Outputs: Rl
-- function: getModel
--
-- September, 2016

require 'nn'
require 'nngraph'

local RNN = {}

-- l: layer for which RNN is being built
-- L: total # of layers
--
--[[
                          |
      ###############################################################
      #                   |                                         #
      #     +-----+       |       +-----+             +------+      #
      #     |     |       v       |     |---------+   |      |      #
      #     | E_l |----->(+)----->| R_l |         +-->| Ah_l |      #
      #     |     |       ^       |     |------+      |      |      #
      #     +-----+       |       +-----+      |      +------+      #
      #                   |           |        |                    #
      #                   |           |        |                    #
      #                   |           |        |                    #
      #                   +-----------+        |                    #
      #                                        |                    #
      ###############################################################
                                               |
--]]

function RNN.getModel(channelRl, channelRl_1)
   -- Rl+1 has 2x # of channels as compared to Rl
   -- Rl[t-1] and El[t-1] have same # of channels
   local input = {}
   input[1] = nn.Identity()()       -- Rl+1
   input[2] = nn.Identity()()       -- Rl[t-1]
   input[3] = nn.Identity()()       -- El[t-1]

   local SC = nn.SpatialConvolution

   local n = nn.Sequential()
   local m = nn.ParallelTable()
   -- Conv(Rl+1)
   m:add(nn.SpatialFullConvolution
        (channelRl_1, channelRl, 3, 3, 2, 2, 1, 1, 1, 1))

   -- Conv(Rl[t-1])
   m:add(SC(channelRl, channelRl, 3, 3, 1, 1, 1, 1))

   -- Conv(El[t-1])
   m:add(SC(2*channelRl, channelRl, 3, 3, 1, 1, 1, 1))
   n:add(m)
   n:add(nn.CAddTable())
   n:add(nn.ReLU())

   local Rl = input - n
   local g = nn.gModule(input, {Rl})
   graph.dot(g.fg, 'RNN', 'graphs/RNN')
   return g
end

return RNN
