-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'nn'
require 'nngraph'
-- require 'UntiedConvLSTM'

function mNet(nlayers, input_stride, poolsize, mapss, clOpt, testing)

   local pE, A, upR, R, Ah, E, cR, cA, mA, cAh, up, op

   -- Ah = prediction branch, A_hat in paper
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {prevE, thisE, nextR}
   -- outputs = {E , R}, E == discriminator output, R == generator output

   -- creating input and output lists:
   local inputs = {}
   local outputs = {}
   for L = 1, nlayers do
      inputs[3*L-2] = nn.Identity()() -- previous layer E / input
      inputs[3*L-1] = nn.Identity()() -- prevous time E (from same layer)
      if L < nlayers then inputs[3*L] = nn.Identity()() end -- next R
   end
   
   for L = 1, nlayers do
      if testing then print('MatchNet model: creating layer:', L) end

      pE = inputs[3*L-2] -- previous layer E
      if testing then pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}} end
      
      -- A branch:
      if L > 1 then
         cA = nn.SpatialConvolution(mapss[L-1], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
         mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
         A = pE - cA - mA - nn.ReLU()
      else
         A = pE
      end

      -- R branch:
      cR = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- recurrent / convLSTM temp model
      if L == nlayers then
         R = inputs[3*L-1] - cR -- this E = inputs[3*L-1] in this layer!
      else
         up = nn.SpatialUpSamplingNearest(poolsize)
         upR = inputs[3*L] - up -- upsampling of next layer R
         R = {inputs[3*L-1], upR} - nn.CAddTable(1) - cR -- this E = inputs[3*L-1] in this layer!
      end
      if testing then R:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}} end

      -- A-hat branch:
      cAh = nn.SpatialConvolution(mapss[L], mapss[L], 3, 3, input_stride, input_stride, 1, 1) -- Ah convolution
      Ah = {R} - cAh - nn.ReLU()
      op = nn.PReLU(mapss[L])
      E = {A, Ah} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
      -- set outputs:
      outputs[3*L-2] = E -- this layer E
      outputs[3*L-1] = R -- this layer R
      outputs[3*L] = Ah -- prediction output
   
   end

   return nn.gModule(inputs, outputs)

end
