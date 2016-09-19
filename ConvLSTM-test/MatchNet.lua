-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'nn'
require 'nngraph'
require 'UntiedConvLSTM'

nngraph.setDebug(true)

function mNet(nlayers, input_stride, poolsize, mapss, clOpt, testing)

   local pE, A, upR, R, P, E, cR

   -- P = prediction branch, A_hat in paper
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {prevE, thisE, nextR}
   -- outputs = {E , R}, E == discriminator output, R == generator output

   -- creating input and output lists:
   local inputs = {}
   local outputs = {}
   for L = 1, nlayers do
      inputs[3*L-2] = nn.Identity()() -- previous E
      inputs[3*L-1] = nn.Identity()() -- this E
      if L < nlayers then inputs[3*L] = nn.Identity()() end -- next R
   end
   
   for L = 1, nlayers do
      print('MatchNet model: creating layer:', L)

      -- define layer functions:
      -- forward branch:
      local cA = nn.SpatialConvolution(mapss[L], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- A convolution, maxpooling
      local cP = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- P convolution
      local mA = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
      local up = nn.SpatialUpSamplingNearest(poolsize)
      local op = nn.PReLU(mapss[L+1])
      -- recurrent branch:
      if testing then
         cR = nn.SpatialConvolution(mapss[L+1], mapss[L+1], 3, 3, input_stride, input_stride, 1, 1) -- recurrent / convLSTM temp model
      else
         cR = nn.ConvLSTM(mapss[L],mapss[L+1], clOpt.nSeq, 3, 3, clOpt.stride)
         -- cR = nn.Sequencer(nn.ConvLSTM(mapss[L],mapss[L+1], clOpt.nSeq, 3, 3, clOpt.stride))
      end

      pE = inputs[3*L-2] -- previous layer E
      if testing then pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}} end
      A = pE - cA - mA - nn.ReLU()

      if L == nlayers then
         print(inputs[3*L-1], cR)
         R = inputs[3*L-1] - cR -- this E = inputs[3*L-1] in this layer!
      else
         upR = inputs[3*L] - up -- upsampling of next layer R
         R = {inputs[3*L-1], upR} - nn.CAddTable(1) - cR -- this E = inputs[3*L-1] in this layer!
      end
      if testing then R:annotate{graphAttributes = {color = 'red', fontcolor = 'red'}} end

      P = {R} - cP - nn.ReLU()
      E = {A, P} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
      -- set outputs:
      outputs[2*L-1] = E -- this layer E
      outputs[2*L] = R -- this layer R
   end

   return nn.gModule(inputs, outputs)

end
