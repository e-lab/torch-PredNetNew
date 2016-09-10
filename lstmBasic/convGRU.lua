-- First wrhgten by Sangpil Kim 
-- gruLSTM with nngraph
-- August 2016
-- Modified https://github.com/karpathy/char-rnn/blob/master/model/GRU.lua

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

-- Set up backend
local backend = cudnn
local sc = backend.SpatialConvolution
local scNB = backend.SpatialConvolution:noBias()
local sg = backend.Szgmoid

function lstm(inDim, outDim, kw, kh, st, pa, layerNum, dropout)
  local dropout = dropout or 0 
  local stw, sth = st, st
  local paw, pah = pa, pa
  local n = layerNum
  -- Input  is 1+ 2*#Layer
  -- Output is 1+ 2*#Layer
 dropout = dropout or 0 
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prevH[L]
  end


  local x
  local outputs = {}
  for L = 1,n do

      local prevH = inputs[L+1]
      -- the input to this layer
      if L == 1 then 
        x = inputs[1]
      else 
        x = outputs[(L-1)] 
        if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      end
      -- GRU tick
      local i2u, i2r, p1
      if L == 1 then
         i2u = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
         i2r = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
         p1 = scNB(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='h2Ig_'..L} 
      else
         i2u = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
         i2r = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
         p1 = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='h2Ig_'..L} 
      end
      local h2u = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2Ig_'..L}
      local h2r = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2Ig_'..L}
      local ug = nn.CAddTable()({i2u, h2u})
      local rg = nn.CAddTable()({i2r, h2r})
      -- forward the update and reset gates
      local update_gate = nn.Sigmoid()(ug)
      local reset_gate = nn.Sigmoid()(rg)
      -- compute candidate hidden state
      local gated_hidden = nn.CMulTable()({reset_gate, prevH})
      local p2 = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(gated_hidden):annotate{name='i2Ig_'..L}  
      local hidden_candidate = nn.Tanh()(nn.CAddTable()({p1,p2}))
      -- compute new interpolated hidden state, based on the update gate
      local zh = nn.CMulTable()({update_gate, hidden_candidate})
      local zhm1 = nn.CMulTable()({nn.AddConstant(1,false)(nn.MulConstant(-1,false)(update_gate)), prevH})
      local next_h = nn.CAddTable()({zh, zhm1})
  
      table.insert(outputs, next_h)
   end
 -- set up the decoder
   local top_h = outputs[#outputs]
   if dropout > 0 then top_hD = nn.Dropout(dropout)(top_h) end
   table.insert(outputs,top_hD)
   return nn.gModule(inputs, outputs)
end


