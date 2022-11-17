% this code is a complete simulation program used to generate data 
% for the OpenSim walking problem
%
% Xiang Gao/Ruofan Wu
% July, 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc

load('InitImpedanceSet_2019.mat');
addpath('.\OpenSim Model')
addpath('.\actor_critic')
addpath('.\multilayerperceptron')
addpath('.\Agent')

% TEST SEQUENCE INDEX -- set both initial impedance # and random seed #
for testID=1:5

    %Desired number of runs
    MaxRun=1; 
    %Desired number of trials
    MaxTr=50;
    %Desired number of steps
    MaxSt=100; % was set to 500 in previous OpenSim experiments
    
    %% Model parameter initialization
    % impedance = [3.4141    0.0792   -0.4021]; % manually assigned impedance
    
    initImpedance = InitialImpedance(testID, :);
    

    targetImpedance = [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];
    featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0224    0.3133   -0.0491    0.2567];
    %% Tuner structure Initializaion
    
    sd = testID;
    rng(sd);
    
    % model
    
    mpath = 'WalkerModel_RK_Torque_Prescribed.osim';  
    md = human_prosthesis(mpath);
    md = md.initHP(targetImpedance, featureTarget);
    
    %% target profile
    [md, state] = md.getPerformance();
    targetProfile = md.profile;
    
    
    Mag = abs(targetImpedance)*0.05; % for all phases, ANN output magnitude
    Mag = reshape(Mag,[4,3]);
    
    %% Fail states
    % md.sminus % failure
    % md.splus  % success
    
    %%==========ADP Settings=====================================
    %% Binary control ------ CTL_TYPE=1
    %% Analylog control ---- CTL_TYPE=0.
    CTL_TYPE = 0;	
    
    nstate = 2;
    naction = 3;
    annhidden = 6;
    cnnhidden = 6;
    Bias = 0;
    
    % create a Qlearning object
    ADP_agent = dHDP_multi(nstate, naction, annhidden, cnnhidden,Bias);
    ADP_agent = ADP_agent.setControlType(CTL_TYPE);
    
    for runs = 1:MaxRun
        uhist=[];    
        disp(['It is ' int2str(runs) ' run.']);
        
        stateHist = [];
        actionHist = [];
        valueHist = [];
        forceHist = [];
        resetHist = [];
        statusHist = [];
        
        cnt=0;
        resetNN = 0;
        for trial=1:MaxTr
            failure=0;
            failReason=0;
            steps=1;
                 
            if trial == 1 % first trial
                %% initial profile
                md = md.simulate([],initImpedance);
                [md, state] = md.getPerformance();  % raw state
                initRawState = state;
    
                [state, reward, status] = md.getState(); % normalized state
                initState = state;
            else % reset impedance if needed
                % which phase fails? then reset that phase
                for phase = 1:4
                   if status(phase) == -1
                        imp = initImpedance((phase*3-2):(phase*3));
                        md = failureReset(md, phase, imp);
                   end
                end
    
                [state, reward, status] = md.getState(); % normalized state
                if min(status) == -1
                    md = md.simulate([],initImpedance);
                    [md, state] = md.getPerformance();
                    [state, reward, status] = md.getState();
                end
                
            end
            
            % get action based on the ANN and get cost based on CNN         
            [action, value, ADP_agent] = ADP_agent.getAction(state, 0);
            if length(resetHist)>= 3 && sum(resetHist(end-2:end)) <=6
                ADP_agent=ADP_agent.resetANN([1 2 3 4]);
            end

            statusHist = [statusHist;reshape(status,1,[])];
            
            while(steps<MaxSt)           
                force = action.*Mag; % the force applied to the model

                prevState = state;		%prev state 
                
                %Plug in the model
                md = md.simulate(force);
                [state, reward, status] = md.getState(); % normalized state      
                
                if min(status) == -1 % failure
                    disp(['Trial # ' int2str(trial) ' has  ' int2str(steps) ' steps.']);
                    resetHist = [resetHist;cnt steps];
                    break;
                end
                % reward was quadratic form. adding URU
                reward=reward+0.01*sum(force.^2,2);
                ADP_agent = ADP_agent.update(prevState, state, reward, [1,2,3,4]); % update for phase =1,2,3,4
                
                % get action based on the updated ANN        
                [action, value, ADP_agent] = ADP_agent.getAction(state, 0); % next action, #1 is the active block
                
    
    
                for phase = 1:4
                    if status(phase) == 0 % if success, make a smaller action
                        action(phase,:) = action(phase,:)*0.2;
                    end
                end
                

                statusHist = [statusHist;reshape(status,1,[])];
                cnt=cnt+1; % for this run
                steps=steps+1;  % for this trial
                
                % next trial if exceed max steps
                if cnt >= MaxSt
                   status = [0 0 0 0]'; 
                   break;
                end
                
                % success if 7 out of 10 steps within the bound
                if cnt > 10
                    if min(sum(statusHist(end-9:end,:))>6) >=1
                        disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);
    
                        break;
                    end
                end
                
            end %end of steps
            
            % success if 7 out of 10 steps within the bound
            if cnt > 10
        %         if min(status) >= 1
                if min(sum(statusHist(end-9:end,:))>6) >=1
                    disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);
                    break;
                end
            end
    
            
        end % end of trials
    end % end of run
    
    totalSteps = size(stateHist,1) % total number of steps before success
    totalTrials = trial - 1
end