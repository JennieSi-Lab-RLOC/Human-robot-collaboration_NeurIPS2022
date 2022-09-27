%Impedances = [3.5672   0.0875  -0.4162 0.2146  0.0456  -0.0350 0.0921  0.0106  -1.1285 0.0489  0.0057  -0.2792];
%Impedances = [3.5672   0.0775  -0.4162  0.2146  0.0656  -0.0550 0.0921  0.0106  -1.1285 0.0489  0.0057  -0.2792];
%Impedances = [3.7140    0.0872   -0.4146    0.2417    0.0752   -0.0470    0.0874    0.0094   -1.0899    0.0547    0.0063   -0.2575];
clear
Impedances = [2.2    0.11   -0.2950    0.18000    0.1008    -0.0925    0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398];
addpath('C:\Users\wrf-i\Documents\OpenSim_tracking\OpenSim Model')
%% peak value and peak-to-peak duration status
%TargetState =[-1.0142, 0.2267, -0.0202, 0.2367,  -0.2875, 0.1700,  -0.0326, 0.2867];
%% peak-to-peak value and peak-to-peak duration status
%TargetState =[-1.0142, 0.2267, 0.9939, 0.2367   -0.2672, 0.1700, 0.2550, 0.2833];
%% peak value and peak timing status
%TargetState =[-1.0142, 0.2267, -0.0202, 0.4633, -0.2875, 0.6333, -0.0325, 0.9167];
%TargetState = [-0.2979    0.1167   -0.0394    0.2767   -1.0215    0.3033   -0.0258    0.2500];
%TargetState = [-0.2979    0.1167   -0.0108    0.2800   -1.0083    0.3100   -0.0520    0.2467];
%TargetState = [-0.3038    0.1133   -0.0461    0.2733   -1.0244    0.3133   -0.0309    0.2500];
%TargetState = [-0.3035    0.1133   -0.0558    0.2700   -1.0296    0.3100   -0.0373    0.2533]
% TargetState = [ -0.3035    0.1133   -0.0593    0.2700   -1.0307    0.3100   -0.0511    0.2600];
TargetState = [ -0.2921 0.13 -0.0514 0.2667 -1.0507 0.3067 -0.0628 0.2533];
mpath = ['.\OpenSim Model\WalkerModel_RK_Torque_dual_ba_short.osim'];   
Angle = 0.1431;
InitU = Impedances;
samples = 30;
totalcnt = 1;
rng(10);
InitialImpedance=[];
safedegree1sd =[7,5,6,4]; % [6,5,7,4];
rate = 1.5;
saferadian = [3,3,3,3]*3.14/180;

Initialstates = [];
InitialError = [];
InitialMotion = [];
InitialAngle = [];
it = 0;
while(it<samples)
    totalcnt = totalcnt + 1;
    InitI = Impedances + Impedances.*((rand(1,12)-0.5).*0.2);
    InitA = Angle;
%     InitA = (rand(1,1)-0.5)*2*0.0872*0;
%     InitA = InitA + Angle;
    
%     [Motion,Xf] = Lowerlimb_Motion_XiangEdited(mpath,InitI);
    [rawProfile,rawProfile_l,state,state_l,grfo,StepAngle,TorList] = Lowerlimb_Motion_dual(mpath,InitI,1);
%     [SI_ST, SI_SL] = getSymetryIndex_halfgait(grfo,StepAngle,state_l);
%     [SI_pulse,SI_brake]=getAPSI_halfgait(grfo,state,state_l);
    [featuretmp,STSI,SLSI]=getPeakError(rawProfile,rawProfile_l,state,state_l,StepAngle);
%     Initerror = Xf-TargetState;
    feature = [featuretmp(:,1)' ,STSI,SLSI];
%     IniterrorR = reshape(Initerror',2,4)';
    Initperror = abs(featuretmp');
%     maxerror = max(abs(feature(1:6)));
    if length(feature) == 6
        if ~max(Initperror(1,:) > saferadian) &&  ~max(Initperror(2,:) > 0.06) && mean(Initperror(1,:))> 0.02 
            it = it + 1;
            InitialImpedance=[InitialImpedance;InitI];
%             InitialAngle = [InitialAngle; InitA];
%             Initialstates = [Initialstates; featuretmp(:)];
            InitialError = [InitialError; feature];
            InitialMotion = [InitialMotion,rawProfile];
        end
    end
end
[irow,icol] = size(InitialImpedance);
rng(1);
sdlist = round(rand(1,irow)*500);
% figure();
% plot(InitialError(:,2:2:end)','r*');
% figure();
% plot(InitialError(:,1:2:end)','r*');
% figure();
% plot(InitialMotion);
% save('InitImpedanceSet_Xmas_dual_new.mat','InitialImpedance','sdlist','Initialstates','InitialError','InitialMotion','InitialAngle','totalcnt','rate');
