% Reference: Yue et al. 2016 Adaptive control of powered transfemoral prostheses 
% based on adaptive dynamic programming
addpath('H:\PI_OpenSim\OpenSim Model');
% targetImpedance 1~3 are k,B,theta_e parameters in phase 1, 4~6 are phase
% 2, 7~9 are phase 3, 10~12 are phase 4
% targetImpedance = [4.2002    0.1220   -0.2592    0.2572    0.0838   -0.0408    0.0887    0.0088   -1.0492    0.0451    0.0052   -0.2307];
% targetImpedance = [0.5 0.025   -0.4300    0.5 0.025   -0.0495    0.5 0.025   -1.0770    00.5 0.025   -0.2798];
% targetImpedance =  [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];

targetImpedance =   [2.000    0.1000   -0.3300    0.2200    0.0781    -0.0495    0.0400    0.0052   -1.0770    0.0453    0.0058   -0.2798];



% targetImpedance = [0 0 0 0 0 0 0 0 0 0 0 0];

% Impedance = [3.355 0.0891 -0.4660 0.2695 0.0781 -0.0495 0.0823 0.0088 -1.0770 0.0453 0.0058 -0.2798];

% peak error and duration error, 1~2 are phase 1, 3~4 are phase 2, 5~6 are
% phase 3, 7~8 are phase 4.
% featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0174    0.3100   -0.0391    0.2567];

parameterAndErrors = [];
profiles = [];

mpath = 'H:\PI_OpenSim\OpenSim Model\WalkerModel_RK_Torque_Prescribed.osim';
impedance = targetImpedance; % change impedance value here
% impedance = Impedance;
[rawProfile,state,grfo,StepAngle,stagehist] = Lowerlimb_Motion(mpath,impedance,0.40);

% normalize rawProfile from 1-301 to 1-100% of a gait cycle
profile = zeros(1,100);
for k = 1:100
  profile(k) = -mean(rawProfile((3*k-2):(3*k)));
end
  

% get peak error and duration error as defined in Yue 2016
% state = state - featureTarget;
% figure
figure
plot(profile);

title('Knee Angle Profile')
ylabel('knee angle (rad)')
xlabel('gait percentage (%)')

