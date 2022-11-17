% ----------------------------------------------------------------------- 
% The OpenSim API is a toolkit for musculoskeletal modeling and           
% simulation. See http://opensim.stanford.edu and the NOTICE file         
% for more information. OpenSim is developed at Stanford University       
% and supported by the US National Institutes of Health (U54 GM072970,    
% R24 HD065690) and by DARPA through the Warrior Web program.             
%                                                                         
% Copyright (c) 2005-2013 Stanford University and the Authors             
% Author(s): Daniel A. Jacobs, Chris Dembia                                             
%                                                                         
% Licensed under the Apache License, Version 2.0 (the "License");         
% you may not use this file except in compliance with the License.        
% You may obtain a copy of the License at                                 
% http://www.apache.org/licenses/LICENSE-2.0.                             
%                                                                         
% Unless required by applicable law or agreed to in writing, software     
% distributed under the License is distributed on an "AS IS" BASIS,       
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or         
% implied. See the License for the specific language governing            
% permissions and limitations under the License.                          
% ----------------------------------------------------------------------- 
%OpenSimPlantControlsFunction  
%   outVector = OpenSimPlantControlsFunction(osimModel, osimState)
%   This function computes a control vector which for the model's
%   actuators.  The current code is for use with the script
%   DesignMainStarterWithControls.m
%
% Input:
%   osimModel is an org.opensim.Modeling.Model object 
%   osimState is an org.opensim.Modeling.State object
%
% Output:
%   outVector is an org.opensim.Modeling.Vector of the control values
% -----------------------------------------------------------------------
function modelControls = OpenSimPlantControlsFunction(osimModel, osimState)
    % Load Library
    import org.opensim.modeling.*;
    global stage;
    global Impedance;
    global stagehist;
    global tmp;
    global tmp_u;
    global tmp_p;
    global force;
    global flag;
    global grf;
    % Check Size
    if(osimModel.getNumControls() < 1)
       error('OpenSimPlantControlsFunction:InvalidControls', ...
           'This model has no controls.');
    end
    
    % Get a reference to current model controls
    modelControls = osimModel.updControls(osimState);
    
    % Initialize a vector for the actuator controls
    % Most actuators have a single control.  For example, muscle have a
    % signal control value (excitation);
    actControls = Vector(1, 0.0);
        
    % Calculate the controls based on any proprty of the model or state 
    RKnee_rz = osimModel.getCoordinateSet().get('RKnee_rz').getValue(osimState);
    RKnee_rz_u = osimModel.getCoordinateSet().get('RKnee_rz').getSpeedValue(osimState);
    LFoot_f = osimModel.getForceSet().get('LFootForce').getRecordValues(osimState);
    RFoot_f = osimModel.getForceSet().get('RFootForce').getRecordValues(osimState);
    %if RFoot_f.get(1) > 0
    %    
    %end
    %
    switch(stage)
        case 0
            index = stage*3+1;
            kp = Impedance(index);
            kv = Impedance(index+1);
            RKnee_rz_des =Impedance(index+2);% -60*pi/180;
            if(RKnee_rz < -10*pi/180 && RKnee_rz_u >= -0.08 && tmp_u >= -0.08)
                tt=osimState.getTime();
                stage = 1;
                stagehist = [stagehist;tt, stage];
                stagestr = 'stage 1'; % swing extension
            end
        case 1
            %RKnee_rz_des = -5*pi/180;
            index = stage*3+1;
            kp = Impedance(index);
            kv = Impedance(index+1);
            RKnee_rz_des =Impedance(index+2);
            if( (RFoot_f.get(1)> 0.5 && grf(end,3) >= 0.5))
                tt=osimState.getTime();
                stage = 2;
                stagehist = [stagehist;tt, stage];
                stagestr = 'stage 2'; % stand flexion
            end
        case 2
            %RKnee_rz_des = -13*pi/180;
            index = stage*3+1;
            kp = Impedance(index);
            kv = Impedance(index+1);
            RKnee_rz_des =Impedance(index+2);
            tt=osimState.getTime();

            if(RKnee_rz < -5*pi/180 && RKnee_rz_u >= -0.08 && tmp_u >= -0.08 && LFoot_f.get(1) == 0)
            %if(LFoot_f.get(1) == 0 && grf(end,3) == 0) %&&RKnee_rz < -5*pi/180 && (RKnee_rz_u >= 0 && tmp_u>=0) RKnee_rz_u > -0.01
                tt=osimState.getTime();
                stage = 3;
                stagehist = [stagehist;tt, stage];
                stagestr = 'stage 3'; % stand extension
            end
        case 3
            %RKnee_rz_des = 2*pi/180;
            index = stage*3+1;
            kp = Impedance(index);
            kv = Impedance(index+1); 
            RKnee_rz_des =Impedance(index+2);
            if( LFoot_f.get(1) >= 0.5 && grf(end,3) >= 0.5 ) % && RKnee_rz > -2*pi/180&&RFoot_f.get(1)> 0
                %display('stage 0');
                tt=osimState.getTime();
                stage = 0;
                stagehist = [stagehist;tt, stage];
                stagestr = 'stage 0';  % swing flexion
            end

    end
    %display(stagestr);
    tt=osimState.getTime();
    grf = [grf;tt,RFoot_f.get(1),LFoot_f.get(1)];
    force = [force ;tt,LFoot_f.get(1),LFoot_f.get(2),LFoot_f.get(3),RFoot_f.get(1)];
    tmp_u = (tmp_u + RKnee_rz_u)/2;
    tmp_p = (tmp_p + RKnee_rz)/2;
    tmp = RFoot_f.get(1);

    val = kv * RKnee_rz_u + kp * (RKnee_rz - RKnee_rz_des);
    % Set Actuator Controls
    actControls.set(0, val);
    %actControls
    %modelControls

    % Update modelControls with the new values
    
    osimModel.updActuators().get('torqueAct_RK').addInControls(actControls, modelControls);

    
 end