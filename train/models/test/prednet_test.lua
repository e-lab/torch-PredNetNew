--------------------------------------------------------------------------------
-- Testing script for prednet.lua
--------------------------------------------------------------------------------
-- Alfredo Canziani, Nov 16
--------------------------------------------------------------------------------

local opt = {
   layers = 2, seq = 5, height = 8, width = 8, saveGraph = true,
   channels = {3, 16, 32}
}
local prednet = require '../prednet'
-- Initialize model generator
prednet:__init(opt)
-- Get the model unwrapped over time as well as the prototype
local model, prototype = prednet:getModel()

paths.mkdir('graphs')

--------------------------------------------------------------------------------
-- Testing batch mode
--------------------------------------------------------------------------------

local function getBatchInput(b, seq, height, width, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(b, channels[1], height, width)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(b, seq, channels[1], height, width)        -- Image
   end
   x[3] = torch.zeros(b, channels[1], height, width)                -- C1[0]
   x[4] = torch.zeros(b, channels[1], height, width)                -- H1[0]
   x[5] = torch.zeros(b, 2*channels[1], height, width)              -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      x[3*l]   = torch.zeros(b, channels[l], height, width)         -- C1[0]
      x[3*l+1] = torch.zeros(b, channels[l], height,width)          -- Hl[0]
      x[3*l+2] = torch.zeros(b, 2*channels[l], height, width)       -- El[0]
   end
   height = height/2
   width = width/2
   x[2] = torch.zeros(b, channels[L+1], height, width)              -- RL+1

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
print('Batch test: ' .. sys.COLORS.green .. 'pass')

local block
local node = 12 -- block 1
for a, b in ipairs(prototype.forwardnodes) do
   if b.id == node then
      block = b.data.module
      graph.dot(block.fg, 'Block tensor', 'graphs/block-batch')
     break
   end
end

--------------------------------------------------------------------------------
-- Testing simple mode
--------------------------------------------------------------------------------

local function getInput(seq, height, width, L, channels, mode)
   -- Input for the gModule
   local x = {}

   if mode == 1 then
      x[1] = torch.randn(channels[1], height, width)             -- Image
   elseif mode == 2 then
      x[1] = torch.randn(seq, channels[1], height, width)        -- Image
   end
   x[3] = torch.zeros(channels[1], height, width)                -- C1[0]
   x[4] = torch.zeros(channels[1], height, width)                -- H1[0]
   x[5] = torch.zeros(2*channels[1], height, width)              -- E1[0]

   for l = 2, L do
      height = height/2
      width = width/2
      x[3*l]   = torch.zeros(channels[l], height, width)         -- C1[0]
      x[3*l+1] = torch.zeros(channels[l], height,width)          -- Hl[0]
      x[3*l+2] = torch.zeros(2*channels[l], height, width)       -- El[0]
   end
   height = height/2
   width = width/2
   x[2] = torch.zeros(channels[L+1], height, width)              -- RL+1

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
graph.dot(block.fg, 'Block tensor', 'graphs/block-tensor')
