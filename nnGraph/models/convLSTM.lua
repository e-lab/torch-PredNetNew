-- First written by Sangpil Kim
<<<<<<< HEAD:nnGraph/models/convLSTM.lua
=======
-- Notation is from https://github.com/oxford-cs-ml-2015/practical6/blob/master/LSTM.lua
>>>>>>> e7f2ae3def75dd48d6a8634baa30e2473d34a996:lstmBasic/convLSTM.lua
-- ConvLSTM with nngraph
-- August 2016

require 'nn'
require 'nngraph'

<<<<<<< HEAD:nnGraph/models/convLSTM.lua
-- Set up backend
local backend = nn
local sc = backend.SpatialConvolution
local scNB = backend.SpatialConvolution:noBias()
local sg = backend.Sigmoid

function lstm(inDim, outDim, opt)
=======
local sc = nn.SpatialConvolution
local scNB = nn.SpatialConvolution:noBias()
local sg = nn.Sigmoid

function convLSTM(inDim, outDim, opt)
>>>>>>> e7f2ae3def75dd48d6a8634baa30e2473d34a996:lstmBasic/convLSTM.lua
  local dropout = opt.dropOut or 0
  local kw, kh  = opt.kw, opt.kh
  local stw, sth = opt.st, opt.st
  local paw, pah = opt.pa, opt.pa
  local n = opt.lm
  -- Input  is 1+ 2*#Layer
  -- Output is 1+ 2*#Layer
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- X
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- Cell
    table.insert(inputs, nn.Identity()()) -- Hidden state
  end

  local x
  local outputs = {}
  for L = 1,n do
     -- Container for previous C and H
    local prevH = inputs[L*2+1]
    local prevC = inputs[L*2]
<<<<<<< HEAD:nnGraph/models/convLSTM.lua
    -- Setup input
    if L == 1 then
      x = inputs[1] --This form is from neuraltalk2
    else
    -- Prev hidden output
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
=======
    -- Get input
    if L == 1 then
      x = inputs[1]
    else
    -- Get x from bottom layer as input
      x = outputs[(L-1)*2]
>>>>>>> e7f2ae3def75dd48d6a8634baa30e2473d34a996:lstmBasic/convLSTM.lua
    end
    --Convolutions
    local i2Ig, i2Fg, i2Og, i2It
    if L == 1 then
       i2Ig = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Fg = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Og = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2It = sc(inDim, outDim, kw, kh, stw, sth, paw, pah)(x)
    else
       i2Ig = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Fg = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2Og = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
       i2It = sc(outDim, outDim, kw, kh, stw, sth, paw, pah)(x)
    end

    local h2Ig = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH)
    local h2Fg = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevH)
    local h2Og = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevC)
    local h2It = scNB(outDim, outDim, kw, kh, stw, sth, paw, pah)(prevC)

    local ig = nn.CAddTable()({i2Ig, h2Ig})
    local fg = nn.CAddTable()({i2Fg, h2Fg})
    local og = nn.CAddTable()({i2Og, h2Og})
    local it = nn.CAddTable()({i2It, h2It})

    -- Gates
    local inGate = sg()(ig)
    local fgGate = sg()(fg)
    local ouGate = sg()(og)
<<<<<<< HEAD:nnGraph/models/convLSTM.lua
    local inTanh = backend.Tanh()(it)
    -- perform the LSTM update
    local nextC           = nn.CAddTable()({
=======
    local inTanh = nn.Tanh()(it)
    -- Calculate Cell state
    local nextC = nn.CAddTable()({
>>>>>>> e7f2ae3def75dd48d6a8634baa30e2473d34a996:lstmBasic/convLSTM.lua
        nn.CMulTable()({fgGate, prevC}),
        nn.CMulTable()({inGate, inTanh})
      })
<<<<<<< HEAD:nnGraph/models/convLSTM.lua
    -- gated cells form the output
    local out = nn.CMulTable()({ouGate, nn.Tanh()(nextC)})

    table.insert(outputs, nextC)
   --Apply dropout
   if dropout > 0 then out = nn.Dropout(dropout)(nextH):annotate{name='drop_final'} end
=======
    -- Calculate output
    local out = nn.CMulTable()({ouGate, nn.Tanh()(nextC)})

    table.insert(outputs, nextC)
   -- Dropout if neccessary
   if dropout > 0 then out = nn.Dropout(dropout)(nextH) end
>>>>>>> e7f2ae3def75dd48d6a8634baa30e2473d34a996:lstmBasic/convLSTM.lua
    table.insert(outputs, out)
  end

  -- Extract output
  local lastH = outputs[#outputs]

  return nn.gModule(inputs, outputs)
end


