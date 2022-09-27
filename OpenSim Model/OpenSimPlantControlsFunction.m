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
    global stage_l;
    global Impedance;
    global Impedance_l;
    global stagehist;
    global stagehist_l;
    global TorqueList;   
    global tmp;
    global tmp_l;
    global tmp_u;
    global tmp_p;
    global tmp_ul;
    global tmp_pl;
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
    actControls_l = Vector(1, 0.0);
    
        
    % Calculate the controls based on any proprty of the model or state 
    RKnee_rz = osimModel.getCoordinateSet().get('RKnee_rz').getValue(osimState);
    RKnee_rz_u = osimModel.getCoordinateSet().get('RKnee_rz').getSpeedValue(osimState);
    LKnee_rz = osimModel.getCoordinateSet().get('LKnee_rz').getValue(osimState);
    LKnee_rz_u = osimModel.getCoordinateSet().get('LKnee_rz').getSpeedValue(osimState);
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
%             if( (RFoot_f.get(1)> 0.5 && grf(end,3) >= 0.5))% || (RKnee_rz_u <= -0.05))% || (RKnee_rz > -0.4 && tmp_p >= RKnee_rz && RKnee_rz_u < 0.5 )) %RKnee_rz > -1*pi/180 && RKnee_rz > -5*pi/180
            if RFoot_f.get(1)> 0.5
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
%             if RKnee_rz<=-0.2
%                 RKnee_rz
%                 LFoot_f.get(1)
%             end
%               if tt>0.1
%                   RKnee_rz_u
%               end
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
%             if( LFoot_f.get(1) >= 0.5 && grf(end,3) >= 0.5 ) % && RKnee_rz > -2*pi/180&&RFoot_f.get(1)> 0
            if LFoot_f.get(1)> 0.5
                %display('stage 0');
                tt=osimState.getTime();
                stage = 0;
                stagehist = [stagehist;tt, stage];
                stagestr = 'stage 0';  % swing flexion
            end
%         case 4
%             index = stage*3+1;
%             kp = Impedance(index);
%             kv = Impedance(index+1); 
%             RKnee_rz_des =Impedance(index+2);
%             if(LFoot_f.get(1)> 0 ) % && RKnee_rz > -2*pi/180&&RFoot_f.get(1)> 0
%                 %display('stage 0');
%                 tt=osimState.getTime();
%                 stage = 0;
%                 stagehist = [stagehist;tt, stage];
%                 stagestr = 'stage 0';
%             end

    end
%% Left limb
    switch(stage_l)
        case 0
            index = stage_l*3+1;
            kp_l = Impedance_l(index);
            kv_l = Impedance_l(index+1);
            
            LKnee_rz_des =Impedance_l(index+2);% -60*pi/180;
            if(LKnee_rz < -10*pi/180 && LKnee_rz_u >= -0.08 && tmp_ul >= -0.08)
                tt=osimState.getTime();
                stage_l = 1;
                stagehist_l = [stagehist_l;tt, stage_l];
                stagestr_l = 'stage 1'; % swing extension
            end
        case 1
            %LKnee_rz_des = -5*pi/180;
            index = stage_l*3+1;
            kp_l = Impedance_l(index);
            kv_l = Impedance_l(index+1);
            LKnee_rz_des =Impedance_l(index+2);
%             if( (LFoot_f.get(1)> 0.5 && grf(end,2) >= 0.5))% || (LKnee_rz_u <= -0.05))% || (LKnee_rz > -0.4 && tmp_p >= LKnee_rz && LKnee_rz_u < 0.5 )) %LKnee_rz > -1*pi/180 && LKnee_rz > -5*pi/180
            if LFoot_f.get(1)> 0.5
                tt=osimState.getTime();
                stage_l = 2;
                stagehist_l = [stagehist_l;tt, stage_l];
                stagestr_l = 'stage 2'; % stand flexion
            end
        case 2
            %LKnee_rz_des = -13*pi/180;
            index = stage_l*3+1;
            kp_l = Impedance_l(index);
            kv_l = Impedance_l(index+1);
            LKnee_rz_des =Impedance_l(index+2);
            tt=osimState.getTime();
%             if LKnee_rz<=-0.2
%                 LKnee_rz
%                 LFoot_f.get(1)
%             end
%               if tt>0.1
%                   LKnee_rz_u
%               end
            if(LKnee_rz < -5*pi/180 && LKnee_rz_u >= -0.08 && tmp_ul >= -0.08 && RFoot_f.get(1) == 0)
            %if(LFoot_f.get(1) == 0 && grf(end,3) == 0) %&&LKnee_rz < -5*pi/180 && (LKnee_rz_u >= 0 && tmp_u>=0) LKnee_rz_u > -0.01
                tt=osimState.getTime();
                stage_l = 3;
                stagehist_l = [stagehist_l;tt, stage_l];
                stagestr_l = 'stage 3'; % stand extension
            end
        case 3
            %LKnee_rz_des = 2*pi/180;
            index = stage_l*3+1;
            kp_l = Impedance_l(index);
            kv_l = Impedance_l(index+1); 
            LKnee_rz_des =Impedance_l(index+2);
%             if( LFoot_f.get(1) >= 0.5 && grf(end,2) >= 0.5 ) % && LKnee_rz > -2*pi/180&&LFoot_f.get(1)> 0
            if RFoot_f.get(1)> 0.5
                %display('stage 0');
                tt=osimState.getTime();
                stage_l = 0;
                stagehist_l = [stagehist_l;tt, stage_l];
                stagestr_l = 'stage 0';  % swing flexion
            end

    end
    
%%
    %display(stagestr);
    tt=osimState.getTime();
    grf = [grf;tt,RFoot_f.get(1),LFoot_f.get(1)];
    force = [force ;tt,LFoot_f.get(0),LFoot_f.get(1),LFoot_f.get(2),RFoot_f.get(0),RFoot_f.get(1),RFoot_f.get(2),stage];
    tmp_u = (tmp_u + RKnee_rz_u)/2;
    tmp_p = (tmp_p + RKnee_rz)/2;
    
    tmp_ul = (tmp_ul + LKnee_rz_u)/2;
    tmp_pl = (tmp_pl + LKnee_rz)/2;
    
    tmp = RFoot_f.get(1);
    tmp_l = LFoot_f.get(1);
   %stagehist = [stagehist, stage];
%     switch(stage)
%     case 0
%         RKnee_rz_des = -90*pi/180;
%         kp = 2.0;       %2.0
%         kv = 0.3;         %0.4
%         if(RKnee_rz < -60*pi/180)
%             stage = 1;
%             display('stage 1');
%         end
%     case 1
%         RKnee_rz_des = 10*pi/180;
%         kp = 3;       %2.0
%         kv = 0.3;        %0.4
%         if(RKnee_rz > 0*pi/180 && LFoot_f.get(1)> 0 )
%             stage = 2;
%             display('stage 2');
%         end
%     case 2
%         RKnee_rz_des = -20*pi/180;
%         kp = 3.5;         %2.0
%         kv = 0.4;         %0.4
%         if(RKnee_rz < -12*pi/180)
%             stage = 3;
%             display('stage 3');
%         end   
%     case 3
%         RKnee_rz_des = 5*pi/180;
%         kp = 3.5;       %2.0
%         kv = 0.4;         %0.4  
%         if( RFoot_f.get(1)> 0 )
%             display('stage 0');
%             %stage = 0;
%         end
% end

%     if(LFoot_f.get(1)>0 && RFoot_f.get(1)==0)
%         stage = 1;
%     elseif(RFoot_f.get(1)>0)
%         if(stage == 1 && RKnee_rz > -1*pi/180)
%             stage = 0;
%         elseif(stage == 0 && RKnee_rz < -50*pi/180)
%             stage = 1;
%         end
%     end
% 
%     if(stage == 0)
%     
%     else
%         RKnee_rz_des = 0*pi/180;
%         kp = 0.01;       %2.0
%         kv = 0.04;         %0.4
%     end
%     
%     if abs(LFoot_f.get(1)) > 0
%        display('Touch Ground');
%     end
    
% RightShank.force.X RightShank.force.Y  RightShank.force.Z
% RightShank.torque.X  RightShank.torque.Y RightShank.torque.Z
% Platform.force.X    Platform.force.Y    Platform.force.Z
% Platform.torque.X    Platform.torque.Y    Platform.torque.Z

%     RKnee_f_label = osimModel.getForceSet().get('RFootForce').getRecordLabels() 
%     for i=0:RKnee_f.getSize()-1
%         char(RKnee_f_label.get(i))
%     end
    %lfoot = Vector(3, 0.0);
    %lfoot = osimModel.getBodySet().get('LeftShank').getMassCenter()

    % Position Control to slightly flexed Knee
    %wn = 5.0;%5.0;

%     RKnee_rz_des = -90*pi/180;
%     if(RKnee_rz >= -5 )
%         RKnee_rz_des = -90*pi/180; 
%     elseif(RKnee_rz < -85 && )
%         RKnee_rz_des = -0*pi/180; 
%     end
%     
    %RKnee_rz_des = -0*pi/180;
    val = kv * RKnee_rz_u + kp * (RKnee_rz - RKnee_rz_des);
    val_l = kv_l * LKnee_rz_u + kp_l * (LKnee_rz - LKnee_rz_des);
    TorqueList = [TorqueList;tt, val,val_l];
    % Set Actuator Controls
    actControls.set(0, val);
    actControls_l.set(0, val_l);
    %actControls
    %modelControls
    tt=osimState.getTime();
    
    
    % Update modelControls with the new values
    %display('enter functions!');
    osimModel.updActuators().get('torqueAct_LK').addInControls(actControls_l, modelControls);

    osimModel.updActuators().get('torqueAct_RK').addInControls(actControls, modelControls);
    
    %modelControls
    %LFoot_f = osimModel.getForceSet().get('LFootForce').getRecordValues(osimState)
    
 end