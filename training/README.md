# MatchNet training

To run:

1: Download dataset: moving [MNIST examples](https://www.dropbox.com/sh/fvsqod4uv7yp0dp/AAAHoHUjkXg4mW6OvV91TgaEa). Small is a 100-sample test, otherwise use the larger ones with 8000 samples. This dataset originated from: http://mi.eng.cam.ac.uk/~vp344/


## to train MatchNet:

3: run: ```qlua main.lua -useGPU -dataBig -nlayers 3 -display -save -savePics```
   run: ```th main.lua   -useGPU -dataBig -nlayers 3 ```

