clear;
clc;
load("Model_all_data.mat");
input = [6.0417400e+02;   1.9853423e+02;  -4.5494120e+01];
%Achilles(input)
output = sim(Achilles, input);






