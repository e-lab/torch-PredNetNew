-- Eugenio Culurciello
-- August 2016
-- MatchNet: a model of PredNet from: https://arxiv.org/abs/1605.08104

require 'nn'
require 'nngraph'
require 'UntiedConvLSTM'

nngraph.setDebug(true)

function mNet(nlayers, input_stride, poolsize, mapss, clOpt)
   
   -- P = prediction branch, A_hat in paper
   -- This module creates the MatchNet network model, defined as:
   -- inputs = {prevE, thisE, nextR}
   -- outputs = {E , R}, E == discriminator output, R == generator output

   -- creating input and output lists:
   local inputs = {}
   local outputs = {}
   inputs[1] = nn.Identity()() -- previous R
   for L = 1, nlayers do
      inputs[L+1] = nn.Identity()() -- previous R
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
      local cR = nn.ConvLSTM(mapss[L],mapss[L+1], clOpt.nSeq, 3, 3, clOpt.stride)
      -- local cR = nn.Sequencer(nn.ConvLSTM(mapss[L],mapss[L+1], clOpt.nSeq, 3, 3, clOpt.stride))

      local pE, A, upR, P, E

      if L == 1 then
         pE = inputs[1] -- previous layer E
      else
         pE = outputs[L-1] 
      end
      pE:annotate{graphAttributes = {color = 'green', fontcolor = 'green'}}
      A = pE - cA - mA - nn.ReLU() - up
      --iR is already updated so we do second forloop
      iR = inputs[L+1] 
      -- iR:annotate{graphAttributes = {color = 'blue', fontcolor = 'green'}}
      P = iR - cP - nn.ReLU()
      E = {A, P} - nn.CSubTable(1) - op -- PReLU instead of +/-ReLU
      -- E:annotate{graphAttributes = {color = 'blue', fontcolor = 'blue'}}
      -- set outputs:
      outputs[1] = E -- this layer E
   end

   return nn.gModule(inputs, outputs)

end
