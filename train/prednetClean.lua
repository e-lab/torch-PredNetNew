local prednet = {}

require 'nn'
require 'nngraph'

-- included local packages
local RNN = require 'RNN'

function prednet:__init(opt)
   -- Input/Output channels for A of every layer
   self.channels = torch.Tensor({{  1,   1},
                                 {  2,  32},
                                 { 64,  64},
                                 {128, 128},
                                 {256, 256}})
   self.layers = opt.layers
   self.seq = opt.seq
   self.res = opt.res
   self.vis = opt.vis
end

-- Macros
local SC = nn.SpatialConvolution

--[[

                 Rl+1[t]
                   |
                   |
                   V
               +-------+
               |       |
    Rl[t-1]--->| Rl[t] |<---El[t-1]
               |       |
               +-------+     +-------------------------------+
                   |         |        Block (Layer: l)       |
                   |         |        ================       |
                   |         |    +------+                   |
                   |         |    | ^    |                   |
                   +------------->| A[t] |--+                |
              Rl[t]          |    |      |  |                |
                             |    +------+  |    +------+    |
                             |              +--->|      |    |
                             |                   | E[t] |---------->El[t]
                             |              +--->|      |    |
                             |    +------+  |    +------+    |
                             |    |      |  |                |
        Img/El-1[t]-------------->| A[t] |--+                |
                             |    |      |                   |
                             |    +------+                   |
                             |                               |
                             +-------------------------------+

--]]

local function block(l, L, iChannel, oChannel)
   local inputs = {}

   --[[
       Inputs and outputs of ONE BLOCK
       Inputs: Image   -> A1 / El-1    -> Al
               Rl      -> Rl
               Total = 2

      Outputs: El
               Ah for layer 1
               Total = 1/2
   --]]

   -- Create input and output containers for nngraph gModule for ONE block
   table.insert(inputs, nn.Identity()())         -- A1 / El-1
   table.insert(inputs, nn.Identity()())         -- Rl

   local A
   local layer = tostring(l)
   if l == 1 then
      A = inputs[1]
   else
      local nodeA = nn.Sequential()
      A = (inputs[1]
           - nodeA:add(SC(iChannel, oChannel, 3, 3, 1, 1, 1, 1))
                  :add(nn.ReLU())
                  :add(nn.SpatialMaxPooling(2, 2, 2, 2)))
   end

   -- Get Rl
   local R = inputs[2]

   -- Predicted A
   local nodeAh = nn.Sequential()
   local Ah = R - nodeAh:add(SC(2*oChannel, oChannel, 3, 3, 1, 1, 1, 1))
                        :add(nn.ReLU())

   -- Error between A and A hat
   local E = {{A, Ah} - nn.CSubTable(1) - nn.ReLU(),
              {Ah, A} - nn.CSubTable(1) - nn.ReLU()}
             - nn.JoinTable(1)

   local g
   if l == 1 then
      -- For first layer return Ah for viewing
      g = nn.gModule(inputs , {E, Ah})
   else
      g = nn.gModule(inputs, {E})
   end

   return g
end

local function stackBlocks(L, channels)
   --[[
       L -> Total number of layers
       Input and outputs in time series
       Inputs: Image   -> A1
               RL+1    -> RL        This is always set to ZERO
               Rl[t-1] -> Rl
               El[t-1] -> Rl
               Total = 2L+2

      Outputs: Ah1, Rl, El
               Total = 2L+1
   --]]

   local inputs = {}
   local outputs = {}

   -- Create input and output containers for nngraph gModule for TIME SERIES
   for i = 1, (2*L+1) do
      table.insert(inputs, nn.Identity()())
      table.insert(outputs, nn.Identity()())
   end
   table.insert(inputs, nn.Identity()())

--------------------------------------------------------------------------------
-- Get Rl
--------------------------------------------------------------------------------
   local Rl_1Channel = 2*channels[L+1][2]
   local iChannel    = channels[L][1]
   local oChannel    = channels[L][2]
   -- Calculate RL
   outputs[2*L] = {inputs[2], inputs[2*L+1], inputs[2*L+2]}
                  - RNN.getModel(2*oChannel, Rl_1Channel)

   -- Calculate RL-1 -> RL-2 -> ... -> R1
   for l = L-1, 1, -1 do
      Rl_1Channel = 2*channels[l+1][2]
      iChannel    = channels[l][1]
      oChannel    = channels[l][2]
      -- Input for R:             Rl+1,     Rl[t-1],       El[t-1]
      outputs[2*l] = {outputs[2*(l+1)], inputs[2*l+1], inputs[2*l+2]}
                     - RNN.getModel(2*oChannel, Rl_1Channel)
   end

--------------------------------------------------------------------------------
-- Stack blocks to form the model for time t
--------------------------------------------------------------------------------
   for l = 1, L do
      iChannel = channels[l][1]
      oChannel = channels[l][2]

      if l == 1 then          -- First layer block has E and Ah as output
         --                img,         Rl
         local E_Ah = {inputs[1], outputs[2*l]}
                      - block(l, L, iChannel, oChannel)
         local E, Ah = E_Ah:split(2)
         outputs[2*l+1] = E
         outputs[1] = Ah
      else                    -- Rest of the blocks have only E as output
                              -- El-1,           Rl
         outputs[2*l+1] = {outputs[2*l-1], outputs[2*l]}
                          - block(l, L, iChannel, oChannel)
      end
   end

   return nn.gModule(inputs, outputs)
end

function prednet:getModel()
   local seq = self.seq
   local res = self.res
   local L = self.layers
   local channels = self.channels
   local vis = self.vis

   local prototype = stackBlocks(L, self.channels)

   local clones = {}
   for i = 1, seq do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()     -- Get input image as batch
   local RL_1 = nn.Identity()()              -- RL + 1 state which is always zero

   local H0 = {}                             -- Default Rl - El pairs
   local H = {}                              -- States linking seq 1 -> 2 -> ...
   local outputs = {}                        -- Ah1

   for l = 1, 2*L do
      -- Link being created between input states to hidden states
      H0[l] = nn.Identity()()
      H[l] = H0[l]
   end

   -- Input sequence needs to be sent as batch
   -- eg for 3 grayscale image your input will be of dimension 3xhxw
   local splitInput = nn.SplitTable(1)(inputSequence)

   for i = 1, seq do
      local inputFrame = nn.SelectTable(i)(splitInput)

      -- Get Ah1 and all the El-Rl pairs as output from all the stacked layers
      local tempStates = {inputFrame, RL_1, table.unpack(H)}
                          - clones[i]

      -- Only Ah1 is sent as output
      outputs[i] = nn.SelectTable(1)(tempStates)         -- Send Ah to output

      -- Rest of the {Rl, El} pairs are reused as {Rl[t-1], El[t-1]}
      if i < seq then
         for l = 1, 2*L do
            H[l] = nn.SelectTable(l+1)(tempStates)     -- Pass state values to next sequence
         end
      end
   end

   local g = nn.gModule({inputSequence, RL_1, table.unpack(H0)}, outputs)

   if vis then
      -- If you want to view tensor dimensions then uncomment this line
      graph.dot(g.fg, 'PredNet for whole sequence', 'graphs/wholeModel')
      graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet')
   end
end

return prednet
