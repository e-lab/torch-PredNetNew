-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104
-- LSTM version from SangPil Kim

require 'nn'
require 'nngraph'
require 'ReLUX' -- rectified linear unit with saturation to 1
local c = require 'trepl.colorize'


function mNet(nlayers, input_stride, poolsize, mapss, clOpt, testing)
   local pE, A, upR, R, RR, RE, Ah, E, cR, cRR, cRE, cA, mA, cAh, up, op
   E={} -- output from layers are saved to connect to next layer input
   R={} -- next layer R also connect directly to this layer R

   -- Ah = prediction / generator branch, A_hat in paper, E = error output
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {same_layer_E, same_layer_R}
   -- outputs = {E, R, Ah}

   -- creating input / output list:
   local inputs = {}
   local outputs = {}

   -- initializing inputs: (1+2*nlayers) in total
   inputs[1] = nn.Identity()() -- global input
   for L = 1, nlayers do
      inputs[2*L] = nn.Identity()() -- same_layer_E (from previous time)
      inputs[2*L+1] = nn.Identity()() -- same_layer_R (from previous time)
   end
   
   -- generating network layers (2 for loops):

   -- first recurrent branch needs to be updated from top:
   for L = nlayers,1,-1 do

      -- R / recurrent branch:
      up = nn.SpatialUpSamplingNearest(poolsize)
      if L == nlayers then
         cR = nn.SpatialConvolution(2*mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- same_layer_E (2x because E is 2xL)
         RE = {inputs[1+2*L-1]} - cR -- same layer E
         R[L] = {RE, inputs[1+2*L]} - nn.CAddTable(1) -- same_layer_E processed +  same_layer_R (from previous time)
      else
         cRE = nn.SpatialConvolution(2*mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- same_layer_E (same dims)
         cRR = nn.SpatialConvolution(mapss[L+1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- next_layer_R (higher dims)
         RR = {R[L+1]} - up - cRR -- upsampling of next_layer_R + conv cRR
         RE = {inputs[1+2*L-1]} - cRE -- same_layer_E + conv cRE 
         R[L] = {RR, RE, inputs[1+2*L]} - nn.CAddTable(1) -- add all R and same_layer_R
      end
      if testing then R[L]:annotate{graphAttributes = {color = 'red', fontcolor = 'black'}} end
   end

   -- the we update bottom-up discriminator and generator network:
   for L = 1, nlayers do
      if testing then print('MatchNet model: creating layer:', L) end 

      -- A branch:
      if L == 1 then
         A = inputs[1] -- global input
      else
         pE = E[L-1] -- previous layer E
         cA = nn.SpatialConvolution(2*mapss[L-1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
         mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
         A = pE - cA - mA - nn.ReLU()
      end
      if testing then A:annotate{graphAttributes = {color = 'green', fontcolor = 'black'}} end

      -- A-hat branch:
      cAh = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      Ah = {R[L]} - cAh - nn.ReLU()
      op = nn.PReLU(mapss[L])

      -- E[L] = {A, Ah} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      E[L] = { {A, Ah} - nn.CSubTable(1) - nn.ReLU(), {Ah, A} - nn.CSubTable(1) - nn.ReLU() } - nn.JoinTable(1) -- same and PredNet model
      if testing then E[L]:annotate{graphAttributes = {color = 'blue', fontcolor = 'black'}} end

      -- output list ( 3 for each layer, 3*nlayers in total):
      outputs[3*L-2] = E[L] -- this layer E
      outputs[3*L-1] = R[L] -- this layer R
      outputs[3*L] = Ah -- prediction output
   end

   local g = nn.gModule(inputs, outputs)
   if testing then nngraph.annotateNodes() end
   return g

end
