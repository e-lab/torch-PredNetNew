local prednet = {}

require 'nngraph'

--nngraph.setDebug(true)

-- included local packages
local RNN = paths.dofile('PCBC_RNN.lua')

function prednet:__init(opt)
   -- Input/Output channels for A of every layer
   self.channels = opt.channels
   self.layers = opt.layers
   self.seq = opt.seq
   self.height = opt.height
   self.width  = opt.width
   self.saveGraph = opt.saveGraph or false
   self.dev = opt.dev or 'cpu'
   self.lstmLayer = opt.lstmLayer or 1

   if self.saveGraph then paths.mkdir('graphs') end
end

-- Macros
local SC = nn.SpatialConvolution
local SFC = nn.SpatialFullConvolution
local gaA  = {color = 'blue', fontcolor = 'blue'}
local gaAh = {style = 'filled', fillcolor = 'skyblue'}
local gaE  = {style = 'filled', fillcolor = 'lightpink'}
local gaR  = {style = 'filled', fillcolor = 'springgreen'}

--[[
    E: Error
    C: Cell State
    R: Hidden State (H)

                 Rl+1[t]            Rl[t-1]   El[t-1]
                   |                   |         |
                   |                   |         |
                   V                   V         V
            +--------------+      +-------------------+
            |   Upsample   |      | Rl[t-1] | El[t-1] |
            +--------------+      +-------------------+
                   |                        |
                   |                        |
                   V                        |
               +-------+                    |
               |       |                    |
    Cl[t-1]--->| Rl[t] |<-------------------+
               |       |
               +-------+     +-------------------------------+
                  | |        |        Block (Layer: l)       |
                  | |        |        ================       |
                  | |        |    +------+                   |
                  | |        |    | ^    |                   |
           <------+ +------------>| A[t] |--+                |
      Cl[t]          Rl[t]   |    |      |  |                |
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

local function block(l, L, iC, oC, vis)
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

   local layer = tostring(l)
   local X = inputs[1]
   if l == 1 then X:annotate{name = 'X', graphAttributes = gaA}
   else           X:annotate{name = 'prjE' .. l-1, graphAttributes = gaA} end

   -- Get Rl
   local R = inputs[2]:annotate{name = 'R' .. layer, graphAttributes = {
                          style = 'filled', fillcolor = 'springgreen'}}

   -- Predicted A
   local nodeAh = nn.Sequential()
                  :add(SFC(oC, iC, 3, 3, 2, 2, 1, 1, 1, 1))
                  :add(nn.ReLU())

   local Ah = (R - nodeAh)
              :annotate{name = 'Ah' .. layer, graphAttributes = gaAh}

   -- Error between A and A hat
   local E = ({{X, Ah} - nn.CSubTable() - nn.ReLU(),
              {Ah, X} - nn.CSubTable() - nn.ReLU()}
             - nn.JoinTable(1, 3))
             :annotate{name = 'E' .. layer, graphAttributes = gaE}

   local nodeA = nn.Sequential()
                 :add(SC(2*iC, oC, 3, 3, 1, 1, 1, 1))
                 :add(nn.ReLU())
                 :add(nn.SpatialMaxPooling(2, 2, 2, 2))

   A = (E - nodeA)
       :annotate{name = 'A' .. layer, graphAttributes = gaA}

   local g
   if l == 1 then
      -- For first layer return Ah for viewing
      g = nn.gModule(inputs , {A, Ah})
   else
      g = nn.gModule(inputs, {A})
   end

   if vis then
      graph.dot(g.fg, 'Block_' .. l, 'graphs/Block_' .. l)
   end
   return g
end

local function stackBlocks(L, channels, vis, lstmLayer)
   --[[
       L -> Total number of layers
       Input and outputs in time series
       Inputs: Image   -> X      (1)
               RL+1    -> LSTM   (2)      This is always set to ZERO

               Cl[t-1] -> LSTM   (3l)     Cell state
               Hl[t-1] -> LSTM   (3l+1)   Hidden state
               El[t-1] -> LSTM   (3l+2)   Error
               Total = 3L+2

               {1, 3l-1, 3l, 3l+1}
      Outputs: Ah1, Cl, Hl, El
               Total = 3L+1
   --]]

   local inputs = {}
   local outputs = {}

   -- Create input and output containers for nngraph gModule for TIME SERIES
   for i = 1, (3*L+2) do
      table.insert(inputs, nn.Identity()():annotate{
         graphAttributes = {fontcolor = 'blue'}
      })
      local l = math.floor(i/3)
      if     i == 1 then   inputs[#inputs]:annotate{name = 'X[t]'}
      elseif i == 2 then   inputs[#inputs]:annotate{name = 'R'..(L+1)..'[t]'}
      elseif i%3 == 0 then inputs[#inputs]:annotate{name = 'C'..l..'[t-1]'}
      elseif i%3 == 1 then inputs[#inputs]:annotate{name = 'R'..l..'[t-1]'}
      elseif i%3 == 2 then inputs[#inputs]:annotate{name = 'prjE'..l..'[t-1]'} end
   end

--------------------------------------------------------------------------------
-- Get Rl
--------------------------------------------------------------------------------

   local upR, R, E, C, rnn
   local c = {}

   -- Calculate RL-1 -> RL-2 -> ... -> R1
   for l = L, 1, -1 do
      if l == L then upR = inputs[2] else upR = outputs[3*(l+1)] end
      c.upR = channels[l+1]
      E = inputs[3*l + 2]
      c.E = channels[l]
      R = inputs[3*l + 1]
      c.R = channels[l]

      rnn = ({upR, R, E} - RNN.getModel(c, vis))
         :annotate{name = 'RNN ' .. l, graphAttributes = gaR}

      outputs[3*l] = rnn

      -- Bypass C when using RNN
      C = inputs[3*l]
      outputs[3*l-1] = C
   end

--------------------------------------------------------------------------------
-- Stack blocks to form the model for time t
--------------------------------------------------------------------------------
   for l = 1, L do
      local iC = channels[l-1]
      local oC = channels[l]


      if l == 1 then          -- First layer block has E and Ah as output
         --                img,        Rl/Hl
         local E_Ah = ({inputs[1], outputs[3*l]}
                      - block(l, L, iC, oC, vis))
                      :annotate{name = '{prjE,Ah}'..l..'[t]',
                      graphAttributes = gaE}

         local E, Ah = E_Ah:split(2)

         outputs[3*l+1] = E:annotate{name = 'prjE'..l..'[t]',
                                     graphAttributes = {
                                     style = 'filled',
                                     fillcolor = 'hotpink'}}

         outputs[1] = Ah:annotate{name = 'Prediction',
                                  graphAttributes = gaAh}
      else                    -- Rest of the blocks have only E as output
                              -- El-1,           Rl/Hl
         outputs[3*l+1] = ({outputs[3*(l-1)+1], outputs[3*l]}
                          - block(l, L, iC, oC, vis))
                          :annotate{name = 'prjE'..l..'[t]',
                          graphAttributes = gaE}
      end
   end

   return nn.gModule(inputs, outputs)
end

function prednet:getModel()
   local seq = self.seq
   local height = self.height
   local width  = self.width
   local L = self.layers
   local channels = self.channels
   local vis = self.saveGraph

   local prototype = stackBlocks(L, channels, vis, self.lstmLayer)

   if self.dev == 'cuda' then
      prototype:cuda()
   end

   local clones = {}
   for i = 1, seq do
      clones[i] = prototype:clone('weight', 'bias', 'gradWeight', 'gradBias')
   end

   local inputSequence = nn.Identity()()     -- Get input image as batch
   local RL_1 = nn.Identity()()              -- RL + 1 state which is always zero

   local H0 = {}                             -- Default Rl - El pairs
   local H = {}                              -- States linking seq 1 -> 2 -> ...
   local outputs = {}                        -- Ah1

   for l = 1, 3*L do
      -- for annotation
      local styleColor = 'lightpink'
      local nodeName = 'E Sequence(' .. seq .. '), Layer(' .. math.ceil(l/3) .. ')'
      if l % 3 == 2 then
         styleColor = 'springgreen'
         nodeName = 'C Sequence(' .. seq .. '), Layer(' .. ((l+1)/3) .. ')'
      elseif l % 3 == 0 then
         styleColor = 'burlywood'
         nodeName = 'H Sequence(' .. seq .. '), Layer(' .. (l/3) .. ')'
      end

      -- Link being created between input states to hidden states
      H0[l] = nn.Identity()()
      H[l] = H0[l]:annotate{name = nodeName,
                            graphAttributes = {
                            style = 'filled',
                            fillcolor = styleColor}}
   end

   -- eg for 5 grayscale images your input will be of dimension 5xhxw
   local splitInput = nn.SplitTable(1, 4)(inputSequence)

   for i = 1, seq do
      local inputFrame = nn.SelectTable(i)(splitInput):annotate{name = 'Input Frame #' .. i,
                                                                graphAttributes = {
                                                                style = 'filled',
                                                                fillcolor = 'gold1'}}

      -- Get Ah1 and all the El-Rl pairs as output from all the stacked layers
      local tempStates = ({inputFrame, RL_1, table.unpack(H)}
                           - clones[i]):annotate{name = 'Model Clone #' .. i,
                                                 graphAttributes = {
                                                 style = 'filled',
                                                 fillcolor = 'moccasin'}}

      -- Only Ah1 is sent as output
      outputs[i] = nn.SelectTable(1)(tempStates)       -- Send Ah to output
                                                :annotate{name = 'Prediction',
                                                          graphAttributes = gaAh}

      -- Rest of the {Cl, Hl, El} pairs are reused as {Cl[t-1], Hl[t-1], El[t-1]}
      if i < seq then
         for l = 1, 3*L do
            local styleColor = 'lightpink'
            local nodeName = 'E Sequence(' .. seq .. '), Layer(' .. math.ceil(l/3) .. ')'
            if l % 3 == 2 then
               styleColor = 'springgreen'
               nodeName = 'C Sequence(' .. seq .. '), Layer(' .. ((l+1)/3) .. ')'
            elseif l % 3 == 0 then
               styleColor = 'burlywood'
               nodeName = 'H Sequence(' .. seq .. '), Layer(' .. (l/3) .. ')'
            end

            -- Pass state values to next sequence
            H[l] = nn.SelectTable(l+1)(tempStates):annotate{
               name = nodeName, graphAttributes = {
                  style = 'filled', fillcolor = styleColor}}
         end
      end
   end

   local g = nn.gModule({inputSequence, RL_1, table.unpack(H0)}, outputs)

   if vis then
      graph.dot(prototype.fg, 'PCBC', 'graphs/PCBC')
      graph.dot(g.fg, 'PredNet for whole sequence', 'graphs/wholeModel')
   end

   return g, clones[1]
end

return prednet
