
% ++++++++++++++++++++++++++ ROUGH WORK +++++++++++++++++++++++++++++++++++++++++++++++++

clear all
clc
close all
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
load Inputs8.txt
load Inputs7.txt
% opos=ounusma
innu = Inputs8'; %Input
ounu = Inputs7'; %Output
save innu.txt innu -ascii 
save ounu.txt ounu -ascii
%%%
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% [pn,meanp,stdp,tn,meant,stdt] = prestd(innu,ounu); %preprocesses the network training set 
[R,Q] = size(innu);
iitst = 1:3:Q;
iival = 1:3:Q;
iitr = [1:3:Q 1:3:Q];
val.P = innu(:,iival); val.T = ounu(:,iival);
test.P = innu(:,iitst); test.T = ounu(:,iitst);
ptr = innu(:,iitr); ttr = ounu(:,iitr);

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Achilles = newff(ptr,ttr,[36 41],{'tansig' 'tansig' 'purelin'},'trainlm')
Achilles.trainParam.show = 25;
%Achilles.trainParam.mem_reduc =25 Que Vishu dijo ?
Achilles.trainParam.epochs = 1500 %iteraciones original de 2000 pero probar entre 500 y 400
Achilles.trainParam.Mu = 0.005
Achilles.trainParam.mu_max = 1e20
Achilles.trainParam.goal = 0.001 % error permitido - performance goal
Achilles.trainParam.max_fail = 15 %maximum validation failures
[Achilles,tr]=trainlm(Achilles,innu,ounu,[],[],val,test);
save(Achilles)

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% [R,Q] = size(a);
% iitst = 2:4:Q;
% iival = 2:4:Q;
% iitr = [1:3:Q 1:3:Q];
% val.P = a(:,iival); val.T = b(:,iival);
% test.P = a(:,iitst); test.T = b(:,iitst);
% ptr = a(:,iitr); ttr = b(:,iitr);
%str