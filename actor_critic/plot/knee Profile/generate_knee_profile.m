 % This is a simple test demo that applys policy iteration / polynomial function
% approximation to the OpenSim walking problem
% 
% The PI_LQR code is now being put into a tuner class form  
%
% 3/22/2015
clear
close all
clc

%% SIMULATION
addpath(genpath('../tuner'), genpath('../ADP+Model'), genpath('../platform'));

ImpHist = [];
uMag = 1;
targetImpedance = [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];
featureTarget = [-0.3038    0.1133   -0.0461    0.2733   -1.0244    0.3133   -0.0309    0.2500];
impAdjustment = [0.05    0.002   0.01,    0.01    0.001   0.005,    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];

profileHist = [];
%% Tuner structure Initializaion
for i = 1:2
% model
md = human_prosthesis();
impedance = targetImpedance + randn(1,12)*0.001.*targetImpedance;
md = md.initHP(impedance);

%% initialize the state and action
[md, state] = md.getPerformance();

profile = md.profile;

profileHist = [profileHist,profile];
end

figure(1)
plot(profileHist);


