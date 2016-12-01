# CamVid dataset

Go through the following steps to prepare CamVid data for training predictive network and then the classifier for segmentation.

+ Step 1: Convert CamVid videos into tensor in order to be able to use them for training predictive network.
Scripts:
    + [videoToTensor](../videoToTensor.lua)
    + [framevideo](../framevideo.lua)
+ Step 2: Located your trained predictive model and save representation along with corresponding segmented labels.
Scripts:
    + [repNlabel](repNlabel.lua)
    + [opts](opts.lua)
+ Step 3: Use the generated data from Step 2 as input and train a classifier for segmentation using [segmentation repository](https://github.com/e-lab/segmentation).
+ Step 4: Once the classifier is trained, run [viewDemo](viewDemo.lua) to see the segmentated output.
Required files:
    + [viewDemo](viewDemo.lua)
    + [colorMap](colorMap.lua)
    + [framevideo](../framevideo.lua)
    + [opts](opts.lua)

Additional tools are:

+ [camvidToTensor](camvidToTensor.lua): Use it in case its decided to feed images which are not in sequence to predictive model.
+ [viewCamvid](viewCamvid.lua): This can be used to just view CamVid frames with their corresponding labels.
