

require 'nn'
require 'LSTM'

--input output lstmOut cellOut #Layer Dropout 
m = lstm(3,6,7,2,0.5)
x = torch.Tensor(3)
--Cell and hidden layer number should be same
c = torch.Tensor(7)
h = torch.Tensor(7)
ht = {h,h,h} 
i = {x,c,unpack(ht)}

o = m:forward(i)
print(o)
