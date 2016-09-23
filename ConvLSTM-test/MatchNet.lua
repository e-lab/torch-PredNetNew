-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'nn'
require 'nngraph'
-- require 'UntiedConvLSTM'
local c = require 'trepl.colorize'


function mNet(nlayers, input_stride, poolsize, mapss, clOpt, testing)

   local pE, A, upR, R, RR, RE, Ah, E, cR, cRR, cRE, cA, mA, cAh, up, op
   E={} -- output from layers are saved to connect to next layer input

   -- Ah = prediction branch, A_hat in paper
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {prevE, thisE, nextR}
   -- outputs = {E , R}, E == discriminator output, R == generator output

   -- creating input / output list:
   local inputs = {}
   local outputs = {}

   -- initializing inputs:
   inputs[1] = nn.Identity()() -- global input
   for L = 1, nlayers do
      inputs[1+3*L-2] = nn.Identity()() -- same layer E (from previous time)
      inputs[1+3*L-1] = nn.Identity()() -- same layer R (from previous time)
      if L < nlayers then inputs[1+3*L] = nn.Identity()() end -- next R
   end
   
   -- generating network layers:
   for L = 1, nlayers do
      if testing then print('MatchNet model: creating layer:', L) end 

      -- A branch:
      if L == 1 then
         A = inputs[1] -- global input
      else
         pE = E[L-1] -- previous layer E
         cA = nn.SpatialConvolution(mapss[L-1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
         mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
         A = pE - cA - mA - nn.ReLU()
      end
      -- if testing then A:annotate{graphAttributes = {color = 'green', fontcolor = 'black'}} end
      
      -- R / recurrent branch:
      up = nn.SpatialUpSamplingNearest(poolsize)
      if L == nlayers then
         cR = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- same layer E
         RE = {inputs[1+3*L-2]} - cR -- same layer E
         R = {RE, inputs[1+3*L-1]} - nn.CAddTable(1) -- same layer E processed +  same layer R (from previous time)
      elseif L == 1 then
         cRE = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- same layer E (same dims)
         cRR = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- prev layer R (higher dims)
         RR = {inputs[1+3*L]} - up - cRR -- upsampling of next layer R + conv cRR
         RE = {inputs[1+3*L-2]} - cRE -- same layer E + conv cRE 
         R = {RR, RE, inputs[1+3*L-1]} - nn.CAddTable(1) -- add all R and previous time R
      else
         cRE = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- same layer E (same dims)
         cRR = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- prev layer R (higher dims)
         RR = {inputs[1+3*L]} - up - cRR -- upsampling of next layer R + conv cRR
         RE = {inputs[1+3*L-2]} - cRE -- same layer E + conv cRE 
         R = {RR, RE, inputs[1+3*L-1]} - nn.CAddTable(1) -- add all R and previous time R
      end
      -- if testing then R:annotate{graphAttributes = {color = 'red', fontcolor = 'black'}} end

      -- A-hat branch:
      if L == 1 then
         cAh = nn.SpatialConvolution(mapss[L+1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      else
         cAh = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      end
      Ah = {R} - cAh - nn.ReLU()
      op = nn.PReLU(mapss[L])
      E[L] = {A, Ah} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      -- if testing then E:annotate{graphAttributes = {color = 'blue', fontcolor = 'black'}} end

      -- output list:
      outputs[3*L-2] = E[L] -- this layer E
      outputs[3*L-1] = R -- this layer R
      outputs[3*L] = Ah -- prediction output
   
   end

   local g = nn.gModule(inputs, outputs)
   if testing then nngraph.annotateNodes() end
   return g

end
