%Achilles.inputs{1}
%Achilles.layers{0} ,
%Achilles.layers{1}, Achilles.layers{2}, Achilles.layers{3}

%wb = getwb();
%[b,IW,LW] = separatewb(net,wb);


%Achilles.biases{1}
%Inp_weights = Achilles.IW{1,1},
%H1_weights =  Achilles.LW{2,1}, 
%H2_weights = Achilles.LW{3,2}
%Inp_bias = Achilles.b{1},
%H1_bias = Achilles.b{2},
%H2_bias = Achilles.b{3},
Achilles.LW{3}
%Achilles.IW{1}
%Achilles.LW{2}
%Achilles.LW{3}
%Achilles.LW{4}
%Achilles.outputs{3}

%save w1.txt Inp_weights -ascii
%save b1.txt Inp_bias -ascii
%save w2.txt H1_weights -ascii
%save b2.txt H1_bias -ascii
%save w3.txt H2_weights -ascii
%save b3.txt H2_bias -ascii

writematrix(Inp_weights,'w1_de.txt')
type 'w1_de.txt'
writematrix(H1_weights,'w2_de.txt')
type 'w2_de.txt'
writematrix(H2_weights,'w3_de.txt')
type 'w3_de.txt'
writematrix(Inp_bias,'b1_de.txt')
type 'b1_de.txt'
writematrix(H1_bias,'b2_de.txt')
type 'b2_de.txt'
writematrix(H2_bias,'b3_de.txt')
type 'b3_de.txt'

