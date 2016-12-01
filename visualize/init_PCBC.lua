local function init(height, width, L, dev)
   local H0 = {}
   local h, w = height, width
   for m = 1, 2 do
      H0[m] = {}
      H0[m][3] = torch.Tensor()                  -- C1[0]
      height = h/2
      width = w/2
      H0[m][4] = torch.zeros(channels[1], height, width)                  -- H1[0]
      H0[m][5] = torch.zeros(channels[1], height, width)                -- E1[0]

      for l = 2, L do
         height = height/2
         width = width/2
         H0[m][3*l]   = torch.Tensor()           -- C1[0]
         H0[m][3*l+1] = torch.zeros(channels[l], height, width)           -- Hl[0]
         H0[m][3*l+2] = torch.zeros(channels[l], height, width)         -- El[0]
      end
      height = height/2
      width = width/2
      H0[m][2] = torch.zeros(channels[L+1], height, width)                -- RL+1
   end

   -- Convert states into CudaTensors if device is cuda
   if dev == 'cuda' then
      for l = 2, 3*L+2 do
         H0[1][l] = H0[1][l]:cuda()
         H0[2][l] = H0[2][l]:cuda()
      end
   end

   return H0

end

return init
