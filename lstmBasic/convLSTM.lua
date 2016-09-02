require 'nn'
require 'nngraph'

--n is the number of layers
function lstm(inDim, outDim, kw, kh, st, pa, n, dropout)
  dropout = dropout or 0 
  stw, sth = st, st
  paw, pah = pa, pa
  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x 
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = inputs[1] --This form is from neuraltalk2
    else 
      --previous h
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end
    -- evaluate the input sums at once for efficiency
    local i2h  = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h_'..L}
    local i2h2 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h2_'..L}
    local i2h3 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h3_'..L}
    local i2h4 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(x):annotate{name='i2h4_'..L}

    local h2h  = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(prev_h):annotate{name='h2h_'..L}
    local h2h2 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(prev_h):annotate{name='h2h2_'..L}
    local h2h3 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(prev_h):annotate{name='h2h3_'..L}
    local h2h4 = nn.SpatialConvolution(inDim, outDim, kw, kh, stw, sth,paw,pah)(prev_h):annotate{name='h2h4_'..L}

    local n1 = nn.CAddTable()({i2h, h2h})
    local n2 = nn.CAddTable()({i2h2, h2h2})
    local n3 = nn.CAddTable()({i2h3, h2h3})
    local n4 = nn.CAddTable()({i2h4, h2h4})

    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  table.insert(outputs, top_h)

  return nn.gModule(inputs, outputs)
end


