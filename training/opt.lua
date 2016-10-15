local o = {}
function o.parse(arg)
   local lapp = require 'pl.lapp'
   local opt = lapp [[
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
     --maxEpochs             (default 100)     max number of training epochs

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
   return opt
end

return o
