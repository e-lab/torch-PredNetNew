require 'image'


frame = torch.load('frames.t7')

f = frame[#frame]


print(f:size())

seqTable = f[1]
print(seqTable:size(1))
local pic = { seqTable[seqTable:size(1)-5]:squeeze(),
              seqTable[seqTable:size(1)-4]:squeeze(),
              seqTable[seqTable:size(1)-3]:squeeze(),
              seqTable[seqTable:size(1)-2]:squeeze(),
              seqTable[seqTable:size(1)-1]:squeeze(),
              seqTable[seqTable:size(1)]:squeeze()}
if true then
   require 'env'
  _im1_ = image.display{image=pic, min=0, max=1, win = _im1_, nrow = 6,
                      legend = 't-5, t-4, t-3, t-2'}
end


