local prednet = {}

require 'nn'
require 'nngraph'

--[[
                     Rl+1
                       |
                    Rl V
                +-------------+
                |             |El
  El[t-1]------>|             |------> Al+1 / Rl[t+1]
              Rl|             |
                |    BLOCK    |
                |             |Al
  Rl[t-1]------>|             |<------ A1 / El-1
              Rl|             |
                +------+------+
                       | Rl
                       V
                     Rl-1
--]]

local function block(l, L, inputChannel, outputChannel)
   local input = {}
   local output = {}

   --[[
       Input and outputs of ONE BLOCK
       Inputs: A1      -> A1 / El-1    -> Al
               Rl+1    -> Rl
               Rl[t-1] -> Rl
               El[t-1] -> Rl
               Total = 4

      Outputs: Rl, El
               Total = 2
   --]]

   -- Create input and output containers for nngraph gModule for ONE block
   for i = 1 to 4 do
      table.insert(input, Identity)
   end
   for i = 1 to 2 do
      table.insert(output, Identity)
   end

   local A, nodeA
   local nodeName = 'A -> ' .. tostring(l)
   if l == 1 then
      A = input[1]
   else
      nodeA = nn.Sequential()
      A = (input[1] - nodeA:add(nn.SpatialConvolution(inputChannel, outputChannel, 3, 3, 1, 1))
                          :add(nn.ReLU(true))
                          :add(nn.SpatialMaxPooling(2,2,2,2)))
                          :annotate{
                           name = nodeName, style = 'filled', fillcolor = 'cyan'}
   end

   -- Input for R:   Rl+1,  Rl[t-1],  El[t-1]
   local R = RNN(input[2], input[3], input[4])

   nodeName = 'Ah -> ' .. tostring(l)
   local NodeAh = nn.Sequential()
   local Ah = (R - NodeAh:add(nn.SpatialConvolution(outputChannel, inputChannel, 3, 3, 1, 1)))
                         :add(nn.ReLU(true))
                         :annotate{
                          name = nodeName, style = 'filled', fillcolor = 'blue'}

   nodeName = 'E -> ' .. tostring(l)
   local E = ({{A, Ah} - nn.CSubTable(1) - nn.ReLU(),
               {Ah, A} - nn.CSubTable(1) - nn.ReLU()}
               - nn.JoinTable(1))
               :annotate{
                name = nodeName, style = 'filled', fillcolor = 'yellow'}

   if l == 1 then
      -- For first layer return Ah for viewing
      output[1] = Ah
   else
      -- For rest, output is Rl which will be later sent to Rl-1
      output[1] = Rl
   end

   -- This El will be used by Al+1
   output[2] = El

   return nn.gModule(input, output)
end

function prednet.getModel(L)
   local input = {}
   local output = {}

   --[[
       L -> Total number of layers
       Input and outputs in time series
       Inputs: A1 -> A1
               Rl -> Rl
               El -> Rl
               Total = 2L+1

      Outputs: Ah1, Rl, El
               Total = 2L+1
   --]]

   -- Create input and output containers for nngraph gModule for TIME SERIES
   for i = 1 to (2*L+1) do
      table.insert(input, Identity)
      table.insert(output, Identity)
   end


end

return prednet
