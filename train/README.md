## Training MatchNet

After obtaining the dataset compatible for this repo, point `main.lua` to it and run the following command:

```
qlua main.lua --datapath ./dataset/ --saveGraph --display --save ./media/matchNet/ --dev cuda
```

`saveGraph` and `display` options are used to save the generated graphs and to display output prediction for every sequence while training respectively.

You can redraw the error charts by typing:

```bash
./showLog.plt /pathOfError/error.log
```
