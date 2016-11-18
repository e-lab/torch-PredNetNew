## Training MatchNet

Training MatchNet is a two stage process:
* Step 1: First train network in unsupervised manner.
* Step 2: Use the learnt representation to train network for segementation.

#### Step 1

After obtaining the dataset compatible for this repo, point `main.lua` to it and run the following command:

```
qlua main.lua --trainData ./dataset/data-train.t7 --testData ./dataset/data-test.t7 --saveGraph --display --save ./media/matchNet/ --dev cuda
```

`saveGraph` and `display` options are used to save the generated graphs and to display output prediction for every sequence while training respectively.

#### Step 2
