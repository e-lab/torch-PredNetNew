-- MatchNet training: predicting future frames in video
-- Eugenio Culurciello, August - September 2016
--
-- code training and testing inspired by: https://github.com/viorik/ConvLSTM
--
-------------------------------------------------------------------------------

require 'nn'
require 'paths'
require 'torch'
require 'image'
require 'optim'
require 'pl'

lapp = require 'pl.lapp'
opt = lapp [[
  Command line options:
  --savedir         (default './results')  subdirectory to save experiments in
  --seed                (default 1250)     initial random seed
  --useGPU                                 use GPU in training
  --GPUID               (default 1)        select GPU
  Data parameters:
  --dataBig                                use large dataset or reduced one

  Training parameters:
  -r,--learningRate       (default 0.001)  learning rate
  -d,--learningRateDecay  (default 0)      learning rate decay
  -w,--weightDecay        (default 0)      L2 penalty on the weights
  -m,--momentum           (default 0.9)    momentum parameter
  --maxEpochs             (default 10)     max number of training epochs

  Model parameters:
  --nlayers               (default 3)     number of layers of MatchNet
  --lstmLayers            (default 1)     number of layers of ConvLSTM
  --inputSizeW            (default 64)    width of each input patch or image
  --inputSizeH            (default 64)    width of each input patch or image
  --nSeq                  (default 20)    input video sequence lenght
  --stride                (default 1)     stride in convolutions
  --padding               (default 1)     padding in convolutions
  --poolsize              (default 2)     maxpooling size

  Display and save parameters:
  -v, --verbose                           verbose output
  --display                               display stuff
  -s,--save                               save models
  --savePics                              save output images examples
]]

if opt.display then require 'env' end
for i =1 , opt.nlayers do
   if i == 1 then
      opt.nFilters  = {1} -- number of filters in the encoding/decoding layers
   else
      table.insert(opt.nFilters, (i-1)*32)
   end
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
os.execute('mkdir '..opt.savedir)
print('Using GPU?', opt.useGPU)
print('How many layers?' ,opt.nlayers)
paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('util.lua')
--Init optimState
local optimState = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
  weightDecay = opt.weightDecay
}
--Get model
model = getModel()
local function main()
   if opt.useGPU then
      require 'cunn'
      require 'cutorch'
      cutorch.setDevice(opt.GPUID)
      model:cuda()
   end
   print('Loading data...')
   local dataFile, dataFileTest = loadData(opt.dataBig)
   local datasetSeq = getdataSeq(dataFile, opt.dataBig) -- we sample nSeq consecutive frames

   print  ('Loaded ' .. datasetSeq:size() .. ' images')

   print('==> training model')
   model:training()
   local w, dE_dw = model:getParameters()
   print('Number of parameters ' .. w:nElement())
   print('Number of grads ' .. dE_dw:nElement())

   local err = 0
   local epoch = 1

   -- set training iterations and epochs according to dataset size:
   opt.dataEpoch = datasetSeq:size()
   opt.maxIter = opt.dataEpoch * opt.maxEpochs

   -- train:
   for t = 1, opt.maxIter do

      -- define eval closure
      local eval_E = function(w)
        local f = 0

        model:zeroGradParameters()
        local sample = datasetSeq[t]
        local inTableG0, targetP, targetC = prepareData(opt,sample)
        --Get output
        -- 1st term is 1st layer of Ahat 2end term is 1stLayer Error
        output = model:forward(inTableG0)
        -- estimate f and gradients
        -- Criterion is embedded
        local dE_dy = {torch.zeros(output[1]:size()):cuda(),output[2]}
        -- Update Model
        model:backward(inTableG0,dE_dy)
        -- Calculate Error and sum
        f = f + output[2]:sum()

        -- return f and df/dX
        return f, dE_dw
      end

      if math.fmod(t, opt.dataEpoch) == 0 then
        epoch = epoch + 1
        print('Training epoch #', epoch)
        optimState.learningRate = optimState.learningRate
      end

      _,fs = optim.adam(eval_E, w, optimState)

      err = err + fs[1]

      --------------------------------------------------------------------
      -- compute statistics / report error
      if math.fmod(t, 1) == 0 then
        print('==> iteration = ' .. t .. ', average loss = ' .. err/(opt.nSeq) .. ' lr '..optimState.learningRate )
        err = 0
        -- Display
        if opt.display then
           display(seqTable,targetC,targetP)
        end
      end
      -- Save file
      if math.fmod(t, opt.dataEpoch) == 1 and t>1 then
         save(target, output, model, optimState, opt)
      end
   end
   print ('Training completed!')
   collectgarbage()
end

main()
