--------------------------------------------------------------------------------
-- Returns the color map for a given set of classes.
--
-- If you want to add more classes then modifiy section "Colors => Classes"
--
-- Written by: Abhishek Chaurasia
--------------------------------------------------------------------------------

local colorMap = {}

-- Map color names to a shorter name
local red = 'red'
local gre = 'green'
local blu = 'blue'

local mag = 'magenta'
local yel = 'yellow'
local cya = 'cyan'

local gra = 'gray'
local whi = 'white'
local bla = 'black'

local lbl = 'lemonBlue'
local bro = 'brown'
local neg = 'neonGreen'
local pin = 'pink'
local pur = 'purple'
local kha = 'khaki'
local gol = 'gold'

-- Create color palette for all the defined colors
local colorPalette = {[red] = {1.0, 0.0, 0.0},
                      [gre] = {0.0, 1.0, 0.0},
                      [blu] = {0.0, 0.0, 1.0},
                      [mag] = {1.0, 0.0, 1.0},
                      [yel] = {1.0, 1.0, 0.0},
                      [cya] = {0.0, 1.0, 1.0},
                      [gra] = {0.3, 0.3, 0.3},
                      [whi] = {1.0, 1.0, 1.0},
                      [bla] = {0.0, 0.0, 0.0},
                      [lbl] = {30/255, 144/255,  255/255},
                      [bro] = {139/255, 69/255,   19/255},
                      [neg] = {202/255, 255/255, 112/255},
                      [pin] = {255/255, 20/255,  147/255},
                      [pur] = {128/255, 0/255,   128/255},
                      [kha] = {240/255, 230/255, 140/255},
                      [gol] = {255/255, 215/255,   0/255}}

-- Default color is chosen as black
local defaultColor = colorPalette[bla]

local function prepCamVidColors(classes)
   local colors = {}

   -- Assign default colors to all the classes
   for i = 1, #classes do
      table.insert(colors, defaultColor)
   end

   -- Colors => Classes
   -- Assign specific color to respective classes
   for i = 1, #classes do
      if classes[i] == 'tunnel' then
         colors[i] = colorPalette[pur]
      elseif classes[i] == 'building' then
         colors[i] = colorPalette[kha]
      elseif classes[i] == 'bicyclist' then
         colors[i] = colorPalette[gre]
      elseif classes[i] == 'car' then
         colors[i] = colorPalette[blu]
      elseif classes[i] == 'animal' then
         colors[i] = colorPalette[mag]
      elseif classes[i] == 'pedestrian' then
         colors[i] = colorPalette[yel]
      elseif classes[i] == 'columnPole' then
         colors[i] = colorPalette[cya]
      elseif classes[i] == 'fence' then
         colors[i] = colorPalette[gra]
      elseif classes[i] == 'lameMkgs' then
         colors[i] = colorPalette[lbl]
      elseif classes[i] == 'miscText' then
         colors[i] = colorPalette[red]
      elseif classes[i] == 'otherMoving' then
         colors[i] = colorPalette[whi]
      elseif classes[i] == 'road' then
         colors[i] = colorPalette[bro]
      elseif classes[i] == 'sidewalk' then
         colors[i] = colorPalette[neg]
      elseif classes[i] == 'signSymbol' then
         colors[i] = colorPalette[pin]
      elseif classes[i] == 'sky' then
         colors[i] = colorPalette[whi]
      elseif classes[i] == 'tree' then
         colors[i] = colorPalette[gre]
      elseif classes[i] == 'train' then
         colors[i] = colorPalette[blu]
      elseif classes[i] == 'othermoving' then
         colors[i] = colorPalette[blu]
      elseif classes[i] == 'archway' then
         colors[i] = colorPalette[pur]
      end
   end
   colorMap.getColors = function()
      return colors
   end
end


function colorMap:init(opt, classes)
   prepCamVidColors(classes)
end

return colorMap
