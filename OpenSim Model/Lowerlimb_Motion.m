function [Motion, Trans, grfo ]= Lowerlimb_Motion(modelPath, Impedances)
import org.opensim.modeling.*;
global stage;
global Impedance;
global stagehist;
global tmp;
global simsteps;        %Used by IntegrateOpenSimPlant to set the frames/second
global force;
global flag;
global tmp_u;
global tmp_p;
global grf;
grf = [];
simsteps = 300;
stage=2;
tmp=0;
flag = 0;
tmp_u = 0;
tmp_p = 0;
stagehist=[];
force = [];
Impedance = [Impedances(7:12),Impedances(1:6)];
%% stiffness1 duamping1 equilibrium1
% Open a Model by name

osimModel = Model(modelPath);
% Use the visualizer (must be done before the call to init system)
osimModel.setUseVisualizer(false);

% Initialize the system and get the initial state
osimState = osimModel.initSystem();

% Set the initial states of the model
editableCoordSet = osimModel.updCoordinateSet();
editableCoordSet.get('Pelvis_ty').setValue(osimState, 0.912);
editableCoordSet.get('RKnee_rz').setValue(osimState, -0.0687);%0.1431
editableCoordSet.get('RKnee_rz').setSpeedValue(osimState, -2.3724);  

% Recalculate the derivatives after the coordinate changes
stateDerivVector = osimModel.computeStateVariableDerivatives(osimState);

% Controls function
controlsFuncHandle = @OpenSimPlantControlsFunction;

% Integrate plant using Matlab Integrator
timeSpan = [0 1]; %[0 5];
integratorName = 'ode15s';
integratorOptions = odeset('AbsTol', 1E-5);

% Run Simulation
stagehist = [stagehist;0, stage];
motionData = IntegrateOpenSimPlant(osimModel, controlsFuncHandle, timeSpan, ...
    integratorName, integratorOptions);
stagehist = [stagehist;1, stage];

labelname = 'RKnee_rz';
indy = strcmp(motionData.labels(:),labelname);
Motion = motionData.data(:,indy);
Trans = stagehist(:,1)';
grfo = force;


end