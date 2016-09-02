

require 'nn'
require 'convLSTM'
torch.setdefaulttensortype('torch.FloatTensor')
--inDim, outDim, kw, kh, stride, pad, #Layer, dropout
m = lstm(1,1,7,7,1,3,1,0.5)
--Cell and hidden layer number should be same
x = torch.Tensor(1,32,32)
c = torch.Tensor(1,32,32)
h = torch.Tensor(1,32,32)

i = {x,c,h}
print(i)
o = m:forward(i)
print(o)
