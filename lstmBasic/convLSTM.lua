require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
local backend = cudnn
local sc = backend.SpatialConvolution
local scNB = backend.SpatialConvolution:noBias()
local sg = backend.Sigmoid
--n is the number of layers
function lstm(inDim, outDim, kw, kh, st, pa, n, dropout)
  local dropout = dropout or 0 
  local stw, sth = st, st
  local paw, pah = pa, pa
  -- there will be 2*n+1 inputs
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
    local i2h, i2h2, i2h3, i2h4
    if L == 1 then
       i2h  = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h_'..L}
       i2h2 = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h2_'..L}
       i2h3 = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h3_'..L}
       i2h4 = sc(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h4_'..L}
    else
       i2h  = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h_'..L}
       i2h2 = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h2_'..L}
       i2h3 = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h3_'..L}
       i2h4 = sc(outDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h4_'..L}
    end

    local h2h  = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2h_'..L}
    local h2h2 = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevH):annotate{name='h2h2_'..L}
    local h2h3 = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevC):annotate{name='h2h3_'..L}
    local h2h4 = scNB(outDim, outDim, kw, kh, stw, sth,paw,pah)(prevC):annotate{name='h2h4_'..L}

    local n1 = nn.CAddTable()({i2h, h2h})
    local n2 = nn.CAddTable()({i2h2, h2h2})
    local n3 = nn.CAddTable()({i2h3, h2h3})
    local n4 = nn.CAddTable()({i2h4, h2h4})

    -- Gates calculation
    local inGate = sg()(n1)
    local fgGate = sg()(n2)
    local ouGate = sg()(n3)
    local inTanh = cudnn.Tanh()(n4)
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
  if dropout > 0 then lastH = nn.Dropout(dropout)(lastH):annotate{name='drop_final'} end
  table.insert(outputs, lastH)

  return nn.gModule(inputs, outputs)
end


