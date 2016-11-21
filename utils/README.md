# Utility files to create data compatible with this repository

Three types of datasets are currently supported:
+ CamVid -> camvidToTensor.lua
+ Cityscapes -> cityscapesToTensor.lua
+ Video file -> videoToTensor.lua
+ MNIST -> downloadMNIST.lua

Type `th abcToTensor.lua`, to use any of the script and get your training and testing set.
These scripts will give a .t7 file compatible with this repository.
Modify the file to specify the loaction of video/dataset, desired height/width and sequence length.

In order to download `MNIST` dataset use `sh downloadMNIST.sh`.
