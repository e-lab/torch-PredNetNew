## Training MatchNet

To download mnist .t7 format data run

```
sh downMnist.sh
```

After obtaining the dataset compatible for this repo, point `main.lua` to it and run the following command:

```
qlua main.lua --trainData ./dataset/data-train.t7 --testData ./dataset/data-test.t7 --saveGraph --display --save ./media/matchNet/ --dev cuda
```

`saveGraph` and `display` options are used to save the generated graphs and to display output prediction for every sequence while training respectively.

You can redraw the error charts by typing:

```bash
./showLog.plt /pathOfError/error.log
```
