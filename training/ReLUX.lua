local ReLUX, parent = torch.class('nn.ReLUX', 'nn.Module')

function ReLUX:__init(saturation, inplace)
   parent.__init(self)
   
   if inplace == nil then
      self.inplace = false
   else
      self.inplace = inplace
   end
   if saturation == nil then
      self.saturation = 1
   else
      self.saturation = saturation
   end

   if (inplace and type(inplace) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function ReLUX:updateOutput(input)
   input.THNN.HardTanh_updateOutput(
      input:cdata(),
      self.output:cdata(),
      0, self.saturation, self.inplace)
   return self.output
end

function ReLUX:updateGradInput(input, gradOutput)
   input.THNN.HardTanh_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      0, self.saturation, self.inplace)
   return self.gradInput
end