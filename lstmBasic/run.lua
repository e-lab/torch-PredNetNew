

require 'nn'
require 'LSTM'

--input output lstmOut cellOut #Layer Dropout 
m = lstm(3,5,7,1,0.5)
x = torch.Tensor(3)
--Cell and hidden layer number should be same
c = torch.Tensor(7)
h = torch.Tensor(7)

i = {x,c,h}

o = m:updataOutput(i)
print(o)
