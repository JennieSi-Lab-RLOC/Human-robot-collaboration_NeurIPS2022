% this code is a complete simulation program used to generate data 
% for the OpenSim walking problem
%
% Si Lab
% Sept 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc

%% Model parameter initialization
% loading initial impedance set
load('InitImpedanceSet_Xmas_twosides.mat');

addpath('.\OpenSim Model')
addpath('.\actor_critic')
addpath('.\multilayerperceptron')


%% Tuner structure Initializaion
for testID=5
%     previous_result=load('freehip_1109_dhdp_random.mat');
    %Desired number of runs
    MaxRun=1; 
    %Desired number of trials
    MaxTr=50;
    %Desired number of steps
    MaxSt=300; % was set to 500 in previous OpenSim experiments

    targetImpedance = [ 2.2    0.11   -0.2950    0.18000    0.1008    -0.0925    0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398];
    featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0224    0.3133   -0.0491    0.2567];
    actionbound=[  0.1     0.01     0.005      0.005;
                   0.01    0.01     0.001     0.001;
                   0.02    0.02     0.02      0.02 ];
    GameActionBound = 0.15;           
    sensorNoise = 0.2*0.6351/180*pi;
    actuatorNoise = 0.01.*actionbound;

    
    sd = testID;
    rng(sd,'twister');
    initImpedance = InitialImpedance(testID, :);
    InitImpedanceHuman = InitialImpedanceH(testID, :);
    
    
    % model

    mpath = ['C:\Users\wrf-i\Documents\OpenSim_twoagent\OpenSim Model\WalkerModel_RK_Torque_dual_ba_short.osim'];  
    
    %% Here a new human_prosthesis class should be overload with human data
    md = human_prosthesis(mpath);% 
    
    md = md.initHP(initImpedance,InitImpedanceHuman);
    %% target profile
    [md, state] = md.getPerformance();
    targetProfile = md.profile;
    normFactor = md.performanceStateNorm;

    Mag = actionbound';
    %% Fail states

    %%==========ADP Settings=====================================
    %% Binary control ------ CTL_TYPE=1
    %% Analylog control ---- CTL_TYPE=0.
    CTL_TYPE = 0;	

    nstate_x = 4;
    nstate_z=2;
    nstate_h=1;
    naction = 3;
    naction_i = 1;
    annhidden = 6;
    cnnhidden = 6;
    Bias = 0;

    desired_SL = 0.75;
    % create a Qlearning object
    HumanControls=load('your_own_training_Data.mat'); % please load the trained human controller here

    Human_ADP=HumanControls.ADP_agent;
    # initialize marl agent
    ADP_agent_game = MARL_multi(nstate_x, naction, naction_i,annhidden, cnnhidden,Bias);
    # initialize human control
    ADP_agent_h = dHDP_multi(nstate_h, naction, annhidden, cnnhidden,Bias);
   
    
    costhist=[];
    rewardhist=[];
    ADP_agent_game = ADP_agent_game.setControlType(CTL_TYPE);
    ADP_agent_h = ADP_agent_h.setControlType(CTL_TYPE);

%%

    runs=1;
    
    disp(['It is ' int2str(runs) ' run.']);

    resetHist = [];
    statusHist = [];

    cnt=0;
    resetNN = 0;
    mu=0;
    batchSize=1;
    
    for  trial = 1:MaxTr
        failure=0;
        failReason=0;
        steps=1;

        
        if trial == 1 % first trial
            %% initial profile
            md = md.simulate([],[],initImpedance,InitImpedanceHuman);
            [md, state] = md.getPerformance();  % raw state
            initRawState = state;
            
            [state, kneeValue,status] = md.getState(); % normalized state
            [ST, SL, STSI,SLSI] = md.getSymmetry();
            initState = state;
        else % reset impedance if needed
            % which phase fails? then reset that phase
            for phase = 1:4
               if status(phase) == -1
                    imp = initImpedance((phase*3-2):(phase*3));
                    md = failureReset(md, phase, imp);
               end
            end
            [state,kneeValue, status] = md.getState(); % normalized state
            [ST, SL, STSI,SLSI] = md.getSymmetry();
            if min(status) == -1
                md = md.simulate([],[],initImpedance,ImpHist_h(end,:));
                [md, state] = md.getPerformance();
                [state, kneeValue,status] = md.getState();
                [ST, SL, STSI,SLSI] = md.getSymmetry();
            end

        end

        state_z = [repmat(STSI,4,1) repmat(SLSI,4,1)];
        state_h = repmat(SL(1)-desired_SL,4,1);
        %% add noise
        for p=1:4

            state(p,1) = state(p,1)+sensorNoise*randn(); 
        end
        state_x = [state state_z];
        %%
        % get action based on the ANN and get cost based on CNN   
        innerValue = zeros(1,4);
        
        [action_u,action_v, value_p, ADP_agent_game] = ADP_agent_game.getAction(state_x, 0);
        [action_h, value_h, ADP_agent_h] = ADP_agent_h.getAction(state_h, 0);

       
        if length(resetHist)>= 3 && sum(resetHist(end-1:end)) <=(batchSize*4)
            ADP_agent_game=ADP_agent_game.resetANN([1 2 3 4]);
            ADP_agent_game=ADP_agent_game.resetNetwork([1 2 3 4]);

        end
        batchCnt=0;
        TrainPreState_x=zeros(16,batchSize);
        TrainState_x=zeros(16,batchSize);
        TrainState_z=zeros(8,batchSize);
        TrainReward_x=zeros(4,batchSize);
        TrainStatus_x = zeros(4,batchSize);
        
        
        while(steps<MaxSt)  

            force_u = action_u.*Mag+ actuatorNoise'.*randn(4,3).*(1-repmat(status,1,3)); % the force applied to the model with actuator noise

            force_h = action_h.*Mag.*0.3;
            prevstate_z =5*[repmat(STSI,4,1) repmat(SLSI,4,1)];
            prevstate_x = 5*[state	prevstate_z];	%prev state 

            %Plug in the model

            md = md.simulate(force_u,force_h);

            [state, kneeValue,status] = md.getState(); % normalized state   
            [ST, SL, STSI,SLSI] = md.getSymmetry();
            
    %% add noise
            for p=1:4
                state(p,1) = state(p,1)+sensorNoise*randn(); 
            end
            if min(status) == -1 % failure
                disp(['Trial # ' int2str(trial) ' has  ' int2str(steps) ' steps.']);
                resetHist = [resetHist;cnt steps];

                break;
            end
            mu=SL(1)-action_v.*GameActionBound-desired_SL;
            muHist=[muHist;mu'];

            muc= 1;

            % reward in LQR format with scaling
            reward_U=1*state(:,1).^2+1*state(:,2).^2+0.25*STSI.^2+0.25*SLSI.^2+0.01*action_u(:,1).^2+0.01*action_u(:,2).^2+0.01*action_u(:,3).^2+0.1*action_v(:,1).^2+muc*mu.^2;
            reward_U = 25*reward_U;

            state_x = 5*[state	repmat(STSI,4,1) repmat(SLSI,4,1)];
            state_z = 5*[repmat(STSI,4,1) repmat(SLSI,4,1)];
            state_h = repmat(SL(1)-desired_SL,4,1);
            batchCnt =batchCnt+1;
            TrainPreState_x(:,batchCnt)=reshape(prevstate_x',[],1);
            TrainState_x(:,batchCnt)=reshape(state_x',[],1);
            
            TrainState_z(:,batchCnt)=reshape(state_z',[],1);
            TrainReward_x(:,batchCnt) = reward_U';
            TrainStatus_x(:,batchCnt)=status';
            
            if batchCnt == batchSize
                update_phases = [];
                for phase = 1:4
                    
                    if status(phase) ~= 1 
                        update_phases = [update_phases,phase];
                    end
                end
                  
                ADP_agent_game = ADP_agent_game.update(TrainPreState_x, TrainState_x ,TrainReward_x,TrainStatus_x, update_phases);%[1 2 3 4]); % update for phase =1,2,3,4

                
                batchCnt = 0;
            end
            
            % get action based on the updated ANN        
            [action_u, action_v,value_p, ADP_agent_game] = ADP_agent_game.getAction(state_x, 0); % next action, #1 is the active block

            [action_h, value_h, ADP_agent_h] = ADP_agent_h.getAction(state_h, 0);

            for phase = 1:4
                if max(isnan(action_u(phase,:))) == 1
                    ADP_agent_game=ADP_agent_game.resetNetwork(phase);
                    status(phase) = -1;
                end
            end
            for phase = 1:4
                if max(isnan(action_v(phase,:))) == 1
                    ADP_agent_h=ADP_agent_h.resetNetwork(phase);
                    status(phase) = -1;
                end
            end
            if abs(SL(1)-desired_SL) >0.02 
                status =  [0 0 0 0]';
            end
            if min(status) == -1 % failure
                disp(['Trial # ' int2str(trial) ' has  ' int2str(steps) ' steps.']);
                resetHist = [resetHist;cnt steps];

                break;
            end


            for phase = 1:4
                if status(phase) == 1 % if success, make a smaller action
                    action_u(phase,:) = action_u(phase,:)*0.2;
%                     action_v(phase,:) = action_v(phase,:)*0.2;
                end
            end

            statusHist = [statusHist;reshape(status,1,[])];% record learning status
            cnt=cnt+1; % for this run
            steps=steps+1;  % for this trial



            if cnt >= MaxSt
               status = [0 0 0 0]'; 
               break;
            end

        end %end of steps

        if cnt >= MaxSt
%            status = [0 0 0 0]'; 
           break;
        end

    end % end of trials


    totalSteps = size(normstateHist_p,1); % total number of steps before success
    totalTrials = trial - 1;
    

end
