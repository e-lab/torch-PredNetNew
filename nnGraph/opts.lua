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
   cmd:option('-dataFile', 'data-small-train.t7' , 'Train file Path')
   cmd:option('-dataFileTest', 'data-small-test.t7', 'Test file path')
   cmd:option('-seed', 1 ,'Seed number')
   cmd:text()

   local opt = cmd:parse(arg or {})
   return opt
end

return m
