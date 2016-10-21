local prednet = {}

require 'nn'
require 'nngraph'

-- nngraph.setDebug(true)

-- included local packages
local RNN = require 'RNN'

function prednet:__init(opt)
   -- Input/Output channels for A of every layer
   self.channels = opt.channels
   self.layers = opt.layers
   self.seq = opt.seq
   self.res = opt.res
   self.vis = opt.vis
end

-- Macros
local SC = nn.SpatialConvolution
local gaA  = {color = 'blue', fontcolor = 'blue'}
local gaAh = {style = 'filled', fillcolor = 'skyblue'}
local gaE  = {style = 'filled', fillcolor = 'lightpink'}
local gaR  = {style = 'filled', fillcolor = 'springgreen'}

-- This function is used if you want to save the graphs
local function getInput(seq, res, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(channels[1], res, res)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(seq, channels[1], res, res)        -- Image
   end
   x[3] = torch.zeros(channels[1], res, res)                -- R1[0]
   x[4] = torch.zeros(2*channels[1], res, res)              -- E1[0]

   for l = 2, L do
      res = res / 2
      x[2*l+1] = torch.zeros(channels[l], res, res)         -- Rl[0]
      x[2*l+2] = torch.zeros(2*channels[l], res, res)       -- El[0]
   end
   res = res / 2
   x[2] = torch.zeros(channels[L+1], res, res)              -- RL+1

   return x
end

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

local function block(l, L, iChannel, oChannel, vis)
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
      A = inputs[1]:annotate{name = 'A' .. layer,
                    graphAttributes = gaA}
   else
      local nodeA = nn.Sequential()
      A = (inputs[1]
           - nodeA:add(SC(iChannel, oChannel, 3, 3, 1, 1, 1, 1))
                  :add(nn.ReLU())
                  :add(nn.SpatialMaxPooling(2, 2, 2, 2)))
                  :annotate{name = 'A' .. layer,
                   graphAttributes = gaA}
   end

   -- Get Rl
   local R = inputs[2]

   -- Predicted A
   local nodeAh = nn.Sequential()
   local Ah = (R:annotate{name = 'R' .. layer, graphAttributes = {
                          style = 'filled',
                          fillcolor = 'springgreen'}}
               - nodeAh:add(SC(oChannel, oChannel, 3, 3, 1, 1, 1, 1))
                       :add(nn.ReLU()))
                       :annotate{name = 'Ah' .. layer,
                        graphAttributes = gaAh}

   -- Error between A and A hat
   local E = ({{A, Ah} - nn.CSubTable(1) - nn.ReLU(),
              {Ah, A} - nn.CSubTable(1) - nn.ReLU()}
             - nn.JoinTable(1))
               :annotate{name = 'E' .. layer,
                graphAttributes = gaE}

   local g
   if l == 1 then
      -- For first layer return Ah for viewing
      g = nn.gModule(inputs , {E, Ah})
   else
      g = nn.gModule(inputs, {E})
   end

   if vis then
      if l == 1 then
         graph.dot(g.fg, 'Block_1', 'graphs/Block_1')
      elseif l == L then
         graph.dot(g.fg, 'Block_L', 'graphs/Block_L')
      end
   end
   return g
end

local function stackBlocks(L, channels, vis)
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
   for i = 1, (2*L+2) do
      table.insert(inputs, nn.Identity()())
   end

--------------------------------------------------------------------------------
-- Get Rl
--------------------------------------------------------------------------------
   local Rl_1Channel = channels[L+1]
   local RlChannel    = channels[L]
   local oChannel    = channels[L]
   -- Calculate RL
   outputs[2*L] = ({inputs[2], inputs[2*L+1], inputs[2*L+2]}
                  - RNN.getModel(RlChannel, Rl_1Channel))
                   :annotate{name = 'R' .. tostring(L),
                    graphAttributes = gaR}

   -- Calculate RL-1 -> RL-2 -> ... -> R1
   for l = L-1, 1, -1 do
      Rl_1Channel = channels[l+1]
      RlChannel   = channels[l]
      oChannel    = channels[l]
      -- Input for R:              Rl+1,       Rl[t-1],       El[t-1]
      outputs[2*l] = ({outputs[2*(l+1)], inputs[2*l+1], inputs[2*l+2]}
                     - RNN.getModel(RlChannel, Rl_1Channel))
                      :annotate{name = 'R' .. tostring(l),
                       graphAttributes = gaR}
   end

--------------------------------------------------------------------------------
-- Stack blocks to form the model for time t
--------------------------------------------------------------------------------
   for l = 1, L do
      local oChannel = channels[l]

      if l == 1 then          -- First layer block has E and Ah as output
         --                img,         Rl
         local E_Ah = ({inputs[1], outputs[2*l]}
                      - block(l, L, oChannel, oChannel, vis))
                       :annotate{name = '{E / Ah}: ' .. tostring(l),
                        graphAttributes = gaE}
         local E, Ah = E_Ah:split(2)
         outputs[2*l+1] = E:annotate{name = 'E: ' .. tostring(l),
                            graphAttributes = {
                            style = 'filled',
                            fillcolor = 'hotpink'}}
         outputs[1] = Ah:annotate{name = 'Prediction',
                         graphAttributes = gaAh}
      else                    -- Rest of the blocks have only E as output
                              -- El-1,           Rl
         local iChannel = 2 * channels[l-1]
         outputs[2*l+1] = ({outputs[2*l-1], outputs[2*l]}
                          - block(l, L, iChannel, oChannel, vis))
                           :annotate{name = 'E: ' .. tostring(l),
                            graphAttributes = gaE}
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

   local prototype = stackBlocks(L, channels, vis)

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
      -- for annotation
      local styleColor = 'lightpink'
      local nodeName = 'E Sequence(1), Layer(' .. tostring(math.ceil(l/2)) .. ')'
      if l % 2 == 0 then
         styleColor = 'springgreen'
         nodeName = 'R Sequence(1), Layer(' .. tostring(l/2) .. ')'
      end

      -- Link being created between input states to hidden states
      H0[l] = nn.Identity()()
      H[l] = H0[l]:annotate{name = nodeName,
                   graphAttributes = {
                   style = 'filled',
                   fillcolor = styleColor}}
   end

   -- Input sequence needs to be sent as batch
   -- eg for 5 grayscale images your input will be of dimension 5xhxw
   local splitInput = nn.SplitTable(1)(inputSequence)

   for i = 1, seq do
      local inputFrame = nn.SelectTable(i)(splitInput)
                         :annotate{name = 'Input Frame #' .. tostring(i),
                          graphAttributes = {
                          style = 'filled',
                          fillcolor = 'gold1'}}

      -- Get Ah1 and all the El-Rl pairs as output from all the stacked layers
      local tempStates = ({inputFrame, RL_1, table.unpack(H)}
                           - clones[i])
                            :annotate{name = 'Model Clone #' .. tostring(i),
                             graphAttributes = {
                             style = 'filled',
                             fillcolor = 'moccasin'}}

      -- Only Ah1 is sent as output
      outputs[i] = nn.SelectTable(1)(tempStates)       -- Send Ah to output
                   :annotate{name = 'Prediction',
                    graphAttributes = gaAh}

      -- Rest of the {Rl, El} pairs are reused as {Rl[t-1], El[t-1]}
      if i < seq then
         for l = 1, 2*L do
            local styleColor = 'lightpink'
            local nodeName = 'E Sequence(' .. tostring(seq) .. '), Layer(' .. tostring(math.ceil(l/2)) .. ')'
            if l % 2 == 0 then
               styleColor = 'springgreen'
               nodeName = 'R Sequence(' .. tostring(seq) .. '), Layer(' .. tostring(l/2) .. ')'
            end

            H[l] = nn.SelectTable(l+1)(tempStates)     -- Pass state values to next sequence
                   :annotate{name = nodeName,
                    graphAttributes = {
                    style = 'filled',
                    fillcolor = styleColor}}
         end
      end
   end

   local g = nn.gModule({inputSequence, RL_1, table.unpack(H0)}, outputs)

   if vis then
      -- If you want to view tensor dimensions then uncomment these lines
      -- local dummyX = getInput(seq, res, L, channels, 2)
      -- local o = g:forward(dummyX)
      graph.dot(g.fg, 'PredNet for whole sequence', 'graphs/wholeModel')

      -- dummyX = getInput(seq, res, L, channels, 1)
      -- o = prototype:forward(dummyX)
      graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet')
   end

   return g, clones[1]
end

return prednet