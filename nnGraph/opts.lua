local m = {}

function m.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Elab PredNet')
   cmd:text()
   cmd:text('Options:')
   ---options---
   cmd:option('-insize', 64, 'Input Image size')
   cmd:option('-nlayers', 2,  'PredNet layers')
   cmd:option('-nSeq', 20,   'Number of Sequence')
   cmd:text()

   local opt = cmd:parse(arg or {})
   return opt
end

return m
