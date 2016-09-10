-- First written by Sangpil Kim 
-- ConvLSTM with nngraph
-- August 2016

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'

-- Set up backend
local backend = cudnn
local sc = backend.SpatialConvolution
local scNB = backend.SpatialConvolution:noBias()
local sg = backend.Sigmoid

function lstm(inDim, outDim, kw, kh, st, pa, layerNum, dropout)
  local dropout = dropout or 0 
  local stw, sth = st, st
  local paw, pah = pa, pa
  local n = layerNum
  -- Input  is 1+ 2*#Layer
  -- Output is 1+ 2*#Layer
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prevC[L]
    table.insert(inputs, nn.Identity()()) -- prevH[L]
  end

  local x 
  local outputs = {}
  for L = 1,n do
     -- Container for previous C and H
    local prevH = inputs[L*2+1]
    local prevC = inputs[L*2]
    -- Setup input
    if L == 1 then 
      x = inputs[1] --This form is from neuraltalk2
    else 
    -- Prev hidden output
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end
    -- In put convolution
    local i2Ig, i2Fg, i2Og, i2It
    if L == 1 then
       i2Ig  = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
       i2Fg = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Fg_'..L}
       i2Og = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Og_'..L}
       i2It = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2It_'..L}
    else
       i2Ig  = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Ig_'..L}
       i2Fg = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Fg_'..L}
       i2Og = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2Og_'..L}
       i2It = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2It_'..L}
    end

    local h2Ig  = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2Ig_'..L}
    local h2Fg = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2Fg_'..L}
    local h2Og = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevC):annotate{name='h2Og_'..L}
    local h2It = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevC):annotate{name='h2It_'..L}

    local ig = nn.CAddTable()({i2Ig, h2Ig})
    local fg = nn.CAddTable()({i2Fg, h2Fg})
    local og = nn.CAddTable()({i2Og, h2Og})
    local it = nn.CAddTable()({i2It, h2It})

    -- Gates calculation
    local inGate = sg()(ig)
    local fgGate = sg()(fg)
    local ouGate = sg()(og)
    local inTanh = cudnn.Tanh()(it)
    -- perform the LSTM update
    local nextC           = nn.CAddTable()({
        nn.CMulTable()({fgGate, prevC}),
        nn.CMulTable()({inGate,     inTanh})
      })
    -- gated cells form the output
    local nextH = nn.CMulTable()({ouGate, nn.Tanh()(nextC)})
    
    table.insert(outputs, nextC)
    table.insert(outputs, nextH)
  end

  -- Get last output
  local lastH = outputs[#outputs]
  --Apply dropout
  if dropout > 0 then lastHD = nn.Dropout(dropout)(lastH):annotate{name='drop_final'} end
  table.insert(outputs, lastHD)

  return nn.gModule(inputs, outputs)
end


