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

function RNN.getModel(channels, vis)
   -- Channels count
   -- upR: channels.upR
   -- R:   channels.R
   -- E:   channels.E

   local input = {}
   input[1] = nn.Identity()():annotate{name = 'upR'}
   input[2] = nn.Identity()():annotate{name = 'R'}
   input[3] = nn.Identity()():annotate{name = 'prjE'}

   local SC = nn.SpatialConvolution
   local SFC = nn.SpatialFullConvolution
   local c = channels

   local m = nn.ParallelTable()
   -- UpConv(upR[t])
   m:add(SFC(c.upR, c.R, 3, 3, 2, 2, 1, 1, 1, 1))
   -- Conv(R[t-1])
   m:add(SC(c.R, c.R, 3, 3, 1, 1, 1, 1))
   -- Iden(prjE[t])
   m:add(nn.Identity())

   local n = nn.Sequential()
   n:add(m)
   n:add(nn.CAddTable(1, 1))
   n:add(nn.ReLU())

   local R = input - n
   local g = nn.gModule(input, {R})

   if vis then
      graph.dot(g.fg, 'RNN', 'graphs/RNN')
   end

   return g
end

return RNN
