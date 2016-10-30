

require 'convLSTM'
require 'nngraph'
torch.setdefaulttensortype('torch.FloatTensor')
nngraph.setDebug(true)
--inDim, outDim, kw, kh, stride, pad, #Layer, dropout
local clOpt = {}
clOpt['nSeq'] = 1
clOpt['kw'] = 7
clOpt['kh'] = 7
clOpt['st'] = 1
clOpt['pa'] = 3
clOpt['dropOut'] = 0
clOpt['lm'] = 1
m = lstm(64,32,clOpt)
--Cell and hidden layer number should be same
x = torch.Tensor(64,32,32)
c = torch.Tensor(32,32,32)
h = torch.Tensor(32,32,32)
ht = {}
for i =1, clOpt.lm do
   table.insert(ht,h:cuda())
   table.insert(ht,c:cuda())
end
i = {x:cuda(),unpack(ht)}
--i = {x:cuda(),c:cuda(),h:cuda()}
print(i)
m:cuda()
o = m:forward(i)
print(o)
