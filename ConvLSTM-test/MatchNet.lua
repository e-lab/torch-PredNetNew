-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'nn'
require 'nngraph'
-- require 'UntiedConvLSTM'
local c = require 'trepl.colorize'

function mNet(nlayers, input_stride, poolsize, mapss, clOpt, testing)

   local pE, A, upR, R, Ah, E, cR, cRR, cRE, cA, mA, cAh, up, op

   -- Ah = prediction branch, A_hat in paper
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {prevE, thisE, nextR}
   -- outputs = {E , R}, E == discriminator output, R == generator output

   -- creating input / output list:
   local inputs = {}
   local outputs = {}

   -- initializing inputs:
   for L = 1, nlayers do
      inputs[4*L-3] = nn.Identity()() -- previous layer E / input
      inputs[4*L-2] = nn.Identity()() -- same layer E (from previous time)
      inputs[4*L-1] = nn.Identity()() -- same layer R (from previous time)
      if L < nlayers then inputs[4*L] = nn.Identity()() end -- next R
   end
   
   -- generating network layers:
   for L = 1, nlayers do
      if testing then print('MatchNet model: creating layer:', L) end 

      pE = inputs[4*L-3] -- previous layer E
      if testing then pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}} end
      
      -- A branch:
      if L == 1 then
         A = pE
      else
         cA = nn.SpatialConvolution(mapss[L-1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
         mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
         A = pE - cA - mA - nn.ReLU()
      end

      -- R / recurrent branch:
      if L == nlayers then
         cR = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- same layer E
         RE = {inputs[4*L-2]} - cR
         R = {RE, inputs[4*L-1]} - nn.CAddTable(1) -- same layer E, previous time R
      else
         cRE = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- same layer E (same dims)
         cRR = nn.SpatialConvolution(mapss[L+1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- prev layer R (higher dims)
         up = nn.SpatialUpSamplingNearest(poolsize)
         RR = {inputs[4*L]} - up - cRR -- upsampling of next layer R + conv cRR
         RE = {inputs[4*L-2]} - cRE -- same layer E + conv cRE 
         R = {RR, RE, inputs[4*L-1]} - nn.CAddTable(1) -- add all R and previous time R
      end
      if testing then R:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}} end

      -- A-hat branch:
      cAh = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      Ah = {R} - cAh - nn.ReLU()
      op = nn.PReLU(mapss[L])
      E = {A, Ah} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      if testing then E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}} end

      -- output list:
      outputs[3*L-2] = E -- this layer E
      outputs[3*L-1] = R -- this layer R
      outputs[3*L] = Ah -- prediction output
   
   end

   return nn.gModule(inputs, outputs)

end
