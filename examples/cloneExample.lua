 require 'nn';                                                                

 -- make an mlp                                                               
 mlp1=nn.Sequential();                                                        
 mlp1:add(nn.Linear(100,10));                                                 

 -- make a copy that shares the weights and biases                            
 mlp2=mlp1:clone('weight','bias');                                            
 -- we change the bias of the first mlp                                       
 mlp1:get(1).bias[1]=99;                                                      
                                                                              
 -- and see that the second one's bias has also changed..                     
 print(mlp2:get(1).bias[1])                                                   

-- Deep copy not sharing 'weight' and 'bias'
 mlp3=mlp1:clone()                                                            
 mlp1:get(1).bias[1]=98;                                                      
--Will show 99
 print(mlp2:get(1).bias[1])                                                   
--Will show 98
 print(mlp3:get(1).bias[1])                                                   
                                                                              
