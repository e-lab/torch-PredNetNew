local test = {}

function test:__init(opt)
   -- Model parameter
   self.layers = opt.layers

   -- Optimizer parameter
   self.optimState = {learningRate      = opt.learningRate,
                      momentum          = opt.momentum,
                      learningRateDecay = opt.learningRateDecay,
                      weightDecay       = opt.weightDecay}

   -- Dataset parameters
   self.channels = opt.channels

   local dataFile, dataFileTest
   if opt.dataBig then
      dataFile     = opt.datapath .. '/data-big-train.t7'
      dataFileTest = otp.datapath .. 'data-big-test.t7'
   else
      dataFile     = opt.datapath .. '/data-small-train.t7'
      dataFileTest = opt.datapath .. 'data-small-test.t7'
   end
   self.dataset = torch.load(dataFile):float()/255                  -- load MNIST
   print("Loaded " .. self.dataset:size(1) .. " image sequences")

   self.res = self.dataset:size(4)
   self.seq = self.dataset:size(2)
   print("Image resolution: " .. self.res .. " x " .. self.res)

   return self.dataset:size()
end

return test
