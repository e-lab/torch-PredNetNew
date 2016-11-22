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

function RNN.getModel(cRup, cR, cE, vis)
   -- Channels count
   -- Rup: cRup
   -- R:   cR
   -- E:   cE

   local input = {}
   input[1] = nn.Identity()():annotate{name = 'Rup'}
   input[2] = nn.Identity()():annotate{name = 'R'}
   input[3] = nn.Identity()():annotate{name = 'E'}

   local SC = nn.SpatialConvolution
   local SFC = nn.SpatialFullConvolution

   local m = nn.ParallelTable()
   -- UpConv(Rup[t])
   m:add(SFC(cRup, cR, 3, 3, 2, 2, 1, 1, 1, 1))
   -- Conv(R[t-1])
   m:add(SC(cR, cR, 3, 3, 1, 1, 1, 1))
   -- Conv(E[t-1])
   m:add(SC(cE, cR, 3, 3, 1, 1, 1, 1))

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
