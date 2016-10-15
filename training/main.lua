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
--Call files
paths.dofile('util.lua')
paths.dofile('data.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

local function main()
   print('Loading data...')
   local dataFile, dataFileTest = loadData(opt.dataBig)
   local datasetSeq = getdataSeq(dataFile, opt.dataBig) -- we sample nSeq consecutive frames
   local testDatasetSeq = getdataSeq(dataFileTest, opt.dataBig) -- we sample nSeq consecutive frames
   for epoch = 1 , opt.maxEpochs do
      train(opt, datasetSeq, epoch)
      test(opt, testDatasetSeq, epoch)
   end

   collectgarbage()
end

main()
