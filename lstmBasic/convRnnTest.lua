require 'convRNN'
require 'nngraph'

torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)

--inDim, outDim, kw, kh, stride, pad, #Layer, dropout
numLayer = 2
m = convRNN(1,3,7,7,1,3,numLayer)

--Cell and hidden layer number should be same
x = torch.Tensor(32,1,32,32)
h = torch.Tensor(32,3,32,32)

ht = {}
for i =1, numLayer do
   table.insert(ht,h:cuda())
end
i = {x:cuda(),unpack(ht)}
--i = {x:cuda(),h:cuda()}

print(i)
m:cuda()
o = m:forward(i)
print(o)
