# Model 

+ Use LSTM
	+ [prednet](prednet.lua) : PredNet implementation
	+ [prednetD](prednetD.lua): `R` is sent to next layer instead of `E` without any delay

+ Use RNN
	+ [PCBC](PCBC.lua)   : Projection of `E` is sent to next layer and `R` of same layer
	+ [PCBCD](PCBCD.lua) : Projection of `E` is sent to `R` and this `R` is sent to next layer without any delay
