## Visualization

+ Visualization script takes `.t7` file as input which was generated using `video2Tensor` script.
+ Set `channel[1]` to 1 if its a gray-scale image and to 3 if its an RGB image.
+ Use `nrow` option to control # of images to show in one row.

Sample command to run the script:

```
qlua visualize.lua --input /dataset.t7 --dmodel /media/HDD1/Models/predNet/ --net 100 --nrow 5
```
