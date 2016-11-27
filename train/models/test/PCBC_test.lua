--------------------------------------------------------------------------------
-- Testing script for prednet.lua
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 16
--------------------------------------------------------------------------------

local opt = {
   layers = 2, seq = 5, height = 8, width = 8, saveGraph = true,
   channels = {3, 16, 32, 64}
}
local prednet = require '../PCBC'
-- Initialize model generator
prednet:__init(opt)
-- Get the model unwrapped over time as well as the prototype
local model, prototype = prednet:getModel()

--------------------------------------------------------------------------------
-- Testing batch mode
--------------------------------------------------------------------------------

local function getBatchInput(b, seq, h, w, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(b, channels[1], h, w)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(b, seq, channels[1], h, w)        -- Image
   end
   h = h/2
   w = w/2
   x[3] = torch.Tensor()                                   -- C1[0]
   x[4] = torch.zeros(b, channels[2], h, w)                -- H1[0]
   x[5] = torch.zeros(b, channels[2], h, w)                -- E1[0]

   for l = 2, L do
      h = h/2
      w = w/2
      x[3*l]   = torch.Tensor()                            -- C1[0]
      x[3*l+1] = torch.zeros(b, channels[l+1], h,w)        -- Hl[0]
      x[3*l+2] = torch.zeros(b, channels[l+1], h, w)       -- El[0]
   end
   h = h/2
   w = w/2
   x[2] = torch.zeros(b, channels[L+2], h, w)              -- RL+1

   return x
end

local dummyX, o

-- getInput(seq, height, width, L, channels, 1)
dummyX = getBatchInput(4, opt.seq, opt.height, opt.width, opt.layers, opt.channels, 1)
o = prototype:forward(dummyX)

dummyX = getBatchInput(4, opt.seq, opt.height, opt.width, opt.layers, opt.channels, 2)
o = model:forward(dummyX)

-- save populated graphs
graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet-batch')
graph.dot(model.fg, 'PredNet for whole sequence', 'graphs/wholeModel-batch')

local block = {}
local nodes = {17, 8} -- block 1
for a, b in ipairs(prototype.forwardnodes) do
   for i, n in ipairs(nodes) do
      if b.id == n then
         block[i] = b.data.module
         graph.dot(block[i].fg, 'Block tensor', 'graphs/block'..i..'-batch')
      end
   end
end
print('Batch test: ' .. sys.COLORS.green .. 'pass')

--------------------------------------------------------------------------------
-- Testing simple mode
--------------------------------------------------------------------------------

local function getInput(seq, h, w, L, channels, mode)
   -- Input for the gModule
   local b = 1
   local x = getBatchInput(b, seq, h, w, L, channels, mode)
   for i, t in ipairs(x) do if t:dim() > 0 then x[i] = t[1] end end
   return x
end

local dummyX, o

-- getInput(seq, height, width, L, channels, 1)
dummyX = getInput(opt.seq, opt.height, opt.width, opt.layers, opt.channels, 1)
o = prototype:forward(dummyX)

dummyX = getInput(opt.seq, opt.height, opt.width, opt.layers, opt.channels, 2)
o = model:forward(dummyX)
graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet-tensor')
print('Simple test: ' .. sys.COLORS.green .. 'pass')

-- save populated graphs
graph.dot(prototype.fg, 'PredNet Model', 'graphs/predNet-tensor')
graph.dot(model.fg, 'PredNet for whole sequence', 'graphs/wholeModel-tensor')
graph.dot(block[1].fg, 'Block tensor', 'graphs/block1-tensor')
graph.dot(block[2].fg, 'Block tensor', 'graphs/block2-tensor')
