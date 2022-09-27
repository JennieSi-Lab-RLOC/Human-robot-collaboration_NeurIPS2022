function [Motion,Motion_l, Trans,Trans_l, grfo,StepAngle,TorList]= Lowerlimb_Motion_dual(modelPath, Impedances)
import org.opensim.modeling.*;
global stage;
global stage_l;
global stagehist_l;
global Impedance;
global Impedance_l;
global stagehist;
global tmp;
global tmp_l;
global simsteps;        %Used by IntegrateOpenSimPlant to set the frames/second
global force;
global flag;
global tmp_u;
global tmp_p;
global tmp_ul;
global tmp_pl;
global grf;
global TorqueList; 
grf = [];
simsteps = 300;
stage=2;
stage_l=0;
tmp=0;
tmp_l=0;
flag = 0;
tmp_u = 0;
tmp_p = 0;
tmp_ul = 0;

tmp_pl = 0;
stagehist=[];
stagehist_l=[];
force = [];
Impedance = [Impedances(7:12),Impedances(1:6)];
TorqueList = [];
% Impedance_l = [Impedances(7:12),Impedances(1:6)];
Impedance_l = [0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398    2.2    0.11   -0.2950    0.18000    0.1008    -0.0925];

%% stiffness1 duamping1 equilibrium1
% Open a Model by name
%tfolder = cd(cd('..'));
%modelfolder = [tfolder '\ADP+Model\Prosthesis_model\WalkerModel_RK_Torque_Prescribed.osim'];
osimModel = Model(modelPath);  
% Use the visualizer (must be done before the call to init system)
osimModel.setUseVisualizer(false);

% Initialize the system and get the initial state
osimState = osimModel.initSystem();

% Set the initial states of the model
editableCoordSet = osimModel.updCoordinateSet();
editableCoordSet.get('Pelvis_ty').setValue(osimState, 0.905);
editableCoordSet.get('Pelvis_ty').setSpeedValue(osimState, -0.5);
editableCoordSet.get('Pelvis_tx').setSpeedValue(osimState,1.77);
editableCoordSet.get('RKnee_rz').setSpeedValue(osimState, -3.58);
editableCoordSet.get('RKnee_rz').setValue(osimState, -0.12);%0.1431
editableCoordSet.get('LKnee_rz').setSpeedValue(osimState, -3.32);
editableCoordSet.get('LKnee_rz').setValue(osimState, -0.205);%0.1431

% Recalculate the derivatives after the coordinate changes
stateDerivVector = osimModel.computeStateVariableDerivatives(osimState);

% Controls function
controlsFuncHandle = @OpenSimPlantControlsFunction;

% Integrate plant using Matlab Integrator
timeSpan = [0 2.2]; %[0 5];
integratorName = 'ode15s';
integratorOptions = odeset('AbsTol', 1E-5);

% Run Simulation
stagehist = [stagehist;0, stage];
stagehist_l = [stagehist_l;0, stage_l];
motionData = IntegrateOpenSimPlant(osimModel, controlsFuncHandle, timeSpan, ...
    integratorName, integratorOptions);

labelname = 'RKnee_rz';
indy = strcmp(motionData.labels(:),labelname);
Motion = motionData.data(:,indy);
StepAngle(:,4)= motionData.data(:,indy);

Trans = stagehist(:,1)';
Trans_l = stagehist_l(:,1)';

grfo = force;


labelname = 'LKnee_rz';
indy = strcmp(motionData.labels(:),labelname);
Motion_l = motionData.data(:,indy);
StepAngle(:,3)= motionData.data(:,indy);

labelname = 'RHip_rz';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,2)= motionData.data(:,indy);

labelname = 'LHip_rz';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,1)= motionData.data(:,indy);

labelname = 'Pelvis_ty';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,5)= motionData.data(:,indy);


labelname = 'Pelvis_tx';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,6)= motionData.data(:,indy);

labelname = 'Pelvis_tx_u';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,7)= motionData.data(:,indy);

labelname = 'Pelvis_ty_u';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,8)= motionData.data(:,indy);


labelname = 'LKnee_rz_u';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,9)= motionData.data(:,indy);

labelname = 'RKnee_rz_u';
indy = strcmp(motionData.labels(:),labelname);
StepAngle(:,10)= motionData.data(:,indy);

TorList = TorqueList; 

end