% Reference: Yue et al. 2016 Adaptive control of powered transfemoral prostheses 
% based on adaptive dynamic programming
% addpath('F:\PI_Opensim_result\OpenSim Model');
% targetImpedance 1~3 are k,B,theta_e parameters in phase 1, 4~6 are phase
% 2, 7~9 are phase 3, 10~12 are phase 4
% targetImpedance = [4.2002    0.1220   -0.2592    0.2572    0.0838   -0.0408    0.0887    0.0088   -1.0492    0.0451    0.0052   -0.2307];
% targetImpedance = [0.5 0.025   -0.4300    0.5 0.025   -0.0495    0.5 0.025   -1.0770    00.5 0.025   -0.2798];
% targetImpedance =  [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];

% targetImpedance = [2.44093328714966,0.329526855153117,-0.0407740128326707,0.480000000000000,0.0608000000000000,0,0.0400000000000000,0.00520000000000000,-1.07700000000000,0.0453000000000000,0.00580000000000000,-0.279800000000000];
% targetImpedance =   [2.43348134294086	0.0957500725054818	-0.386545002432198	0.490000000000000	0.0608000000000000	-0.0100000000000000	0.0400000000000000	0.00520000000000000	-1.07700000000000	0.0453000000000000	0.00580000000000000	-0.279800000000000];
% targetImpedance = [2.2    0.11   -0.2950    0.18    0.1008    -0.0925    0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398];
% targetImpedance = [2.2    0.11   -0.2450    0.18    0.1008    -0.0925    0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398];
targetImpedance = [2.26297596820770,0.0833937685375569,-0.244826969259073,0.180998184549259,0.0970821963663935,-0.0787208274957027,0.0532981747693123,0.00672097131171947,-1.09847219938679,0.0545062879492170,0.00598616064870279,-0.243763729079054];
% targetImpedance = [1.8904    0.2004    0.0377    0.0410    0.0926    0.0735    0.0055    0.0059   -0.3401   -0.0523   -1.0333   -0.2990];
% targetImpedance =  targetImpedance + [-0.00501731099679424 0.00745577084502420 -0.0226044464412941 0 0 0 0 0 0 0 0 0];

% targetImpedance = [0 0 0 0 0 0 0 0 0 0 0 0];

% Impedance = [3.355 0.0891 -0.4660 0.2695 0.0781 -0.0495 0.0823 0.0088 -1.0770 0.0453 0.0058 -0.2798];

% peak error and duration error, 1~2 are phase 1, 3~4 are phase 2, 5~6 are
% phase 3, 7~8 are phase 4.
% featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0174    0.3100   -0.0391    0.2567];
featureTarget =  [ 0 0   0 0.03885 0 0 -0.9888 0 0 -0.1388 0 0];
parameterAndErrors = [];
profiles = [];
% mpath = 'C:\Users\ruofanwu\Downloads\PI_OpenSim\PI_OpenSim\OpenSim Model\WalkerModel_Torque_Prescribed_modified.osim';
mpath = ['C:\Users\wrf-i\Documents\OpenSim_freehip\OpenSim Model\WalkerModel_RK_Torque_dual_ba_short.osim'];  
% mpath = 'C:\Users\ruofanwu\Downloads\PI_OpenSim\PI_OpenSim\OpenSim Model\WalkerModel_Torque_Prescribed.osim';
impedance = targetImpedance; % change impedance value here
% impedance = Impedance;
[rawProfile,rawProfile_l,state,state_l,grfo,StepAngle,TorList] = Lowerlimb_Motion_dual(mpath,impedance,1,1);

% normalize rawProfile from 1-301 to 1-100% of a gait cycle
profile = zeros(1,100);
profile_l = zeros(1,100);

for k = 1:100
  profile(k) = -mean(rawProfile((3*k-2):(3*k)));
  profile_l(k) = -mean(StepAngle((3*k-2):(3*k),3));
end

% ts = linspace(0,1,length(rawProfile));
% [SI_ST, SI_SL] = getSymetryIndex_halfgait(grfo,StepAngle,state_l);
% % [SIST,SISL]=getSymetryIndex(grfo,StepAngle,state_l,2);
% featuretmp = ones(2,4);                % initialize feature vector
% len = length(rawProfile);              % get profile length
% transtp=floor(len*state/2.5);       
% transtp_l=floor(len*state_l/2.5);
% % transition points not equal to 5, something bad happens
% if length(state) >= 10 && length(state) <= 11
%     phacut = ceil((transtp(1:end-1)+transtp(2:end))/2);     % split the profile with transition point
%     phacut_l = ceil((transtp_l(1:end-1)+transtp_l(2:end))/2);     % split the profile with transition point
%     [stflex, stfi] = min(rawProfile(phacut(5):phacut(6)));             % stand flexion peak value and time index
%     stfi = stfi+phacut(5);
%     [stext, stei]= max(rawProfile(stfi:phacut(7)));            % stand extension peak value and time index
%     stei = stei + stfi - 1;                           
%     [swflex,swfi] = min(rawProfile(stei:phacut(8)));           % swing flexion peak value and time index
%     swfi = swfi + stei - 1;
%     [swext, swei]= max(rawProfile(swfi:phacut(9)));                  % swing extension peak value and time index
%     swei = swei + swfi - 1;
% 
%     phaseDuration=state(2:end)-state(1:end-1);
%     peakDelay=phaseDuration(5:8);
%     peakValue=[stflex,stext,swflex,swext];
% %                 peakDelay=[stfi,stei,swfi,swei];
% %                 peakValue=[stflex,stext,swflex,swext];
% %                 peakDelay = ts(peakDelay);
% %                 peakDelay(2:end)=peakDelay(2:end)-peakDelay(1:end-1);
% %                 featuretmp = [peakValue; peakDelay];   
% 
%     [stflex_l, stfi_l] = min(rawProfile_l(phacut_l(3):phacut_l(4)));             % stand flexion peak value and time index
%     stfi_l = stfi_l+phacut_l(3);
%     [stext_l, stei_l]= max(rawProfile_l(stfi_l:phacut_l(5)));            % stand extension peak value and time index
%     stei_l = stei_l + stfi_l - 1;                           
%     [swflex_l,swfi_l] = min(rawProfile_l(stei_l:phacut_l(6)));           % swing flexion peak value and time index
%     swfi_l = swfi_l + stei_l - 1;
%     [swext_l, swei_l]= max(rawProfile_l(swfi_l:phacut_l(7)));                  % swing extension peak value and time index
%     swei_l = swei_l + swfi_l - 1;  
% 
%     peakValue_l=[stflex_l,stext_l,swflex_l,swext_l];
%     phaseDuration_l=state_l(2:end)-state_l(1:end-1);
%     peakDelay_l=phaseDuration_l(3:6);
%     
%     peakValue_l=[stflex_l,stext_l,swflex_l,swext_l];
%     if length(peakValue) == 4 && length(peakValue_l) == 4
%         featuretmp(1,:) = peakValue-peakValue_l;
%         featuretmp(2,:) = peakDelay-peakDelay_l;
%     end
% %                 featuretmp = peakValue-[-0.3944,0.03885, -0.9888, -0.1388];
% 
% else
%     featuretmp=ones(2,4); 
% 
% end
% [tt,l_grf,r_grf] = getGRF(grfo);
% % [SI_pulse,SI_brake]=getAPSI(grfo,state,state_l,2);
% 
% LK=[-0.143117 -0.197048 -0.265116 -0.352033 -0.454833 -0.570199 -0.69115 -0.805644 -0.901986 -0.97145 -1.00932 -1.01665 -0.999725 -0.960978 -0.900241 -0.82013 -0.723963 -0.614181 -0.494103 -0.370533 -0.252026 -0.148004 -0.0750492 -0.039619 -0.0410152 -0.0722566 -0.068766 -0.116239 -0.164934 -0.213977 -0.257611 -0.28571 -0.297404 -0.295484 -0.283441 -0.264417 -0.242077 -0.217468 -0.191114 -0.16441 -0.138928 -0.115017 -0.0937242 -0.0748746 -0.0593412 -0.047473 -0.0413643 -0.042586 -0.0523599 -0.0712094 -0.101753];
% LKt=[0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 0.24 0.26 0.28 0.3 0.32 0.34 0.36 0.38 0.4 0.42 0.44 0.46 0.48 0.5 0.52 0.54 0.56 0.58 0.6 0.62 0.64 0.66 0.68 0.7 0.72 0.74 0.76 0.78 0.8 0.82 0.84 0.86 0.88 0.9 0.92 0.94 0.96 0.98 1];
% 
% RK=[LK(27:51) LK(1:26)];
% 
% % get peak error and duration error as defined in Yue 2016
% % state = state - featureTarget;
% figure
% tl=linspace(0,2.5,length(StepAngle(:,3)));
% plot(tl,-StepAngle(:,3));
% 
% hold on
% plot(tl,-StepAngle(:,4));
% % plot([LKt(1:end-1) 1+LKt(1:end-1)],[-RK(1:end-1) -RK(1:end-1)]);
% plot([state(2) state(2)],[0 1],'g--');
% plot([state(3) state(3)],[0 1],'g--');
% plot([state(4) state(4)],[0 1],'g--');
% plot([state(5) state(5)],[0 1],'g--');
% 
% 
% title('Knee Angle Profile')
% ylabel('knee angle (rad)')
% xlabel('gait percentage (%)')
% legend('Left Limb','Right Limb')
% 
% figure
% plot(tl(1:end-150),-StepAngle(151:end,3));
% hold on
% plot(tl(1:end-150),-StepAngle(1:end-150,4));
% 
% 
% figure
% plot(tl(1:end-150),-StepAngle(1:end-150,3));
% hold on
% plot(tl(1:end-150),-StepAngle(151:end,4));
% % plotgrf
% 
% % figure
% % plot(tt,l_grf);
% % hold on
% % plot(tt,r_grf);
% % title('vertical ground reaction force')
% % ylabel('ground reaction force')
% % xlabel('gait percentage')
% % legend('left foot','right foot')
% 
% function [tt,l_grf,r_grf] = getGRF(grfo)
%     tt = sort(unique(grfo(:,1)));
%     l_grf=zeros(length(tt),1);
%     r_grf=zeros(length(tt),1);
%     for i = 1:length(tt)
%         l_grf(i) = mean(grfo(grfo(:,1)==tt(i),3));
%         r_grf(i) = mean(grfo(grfo(:,1)==tt(i),6));
%     end
%     l_grf = smooth(tt,l_grf,20,'sgolay');
%     r_grf = smooth(tt,r_grf,20,'sgolay');
%     
% end