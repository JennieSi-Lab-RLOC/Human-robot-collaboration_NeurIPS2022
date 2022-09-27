% this code is a complete simulation program used to generate data 
% for the OpenSim walking problem
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
close all
clc

% TEST SEQUENCE INDEX -- set both initial impedance # and random seed #
% testID = 20;	


%% Model parameter initialization

% load('InitImpedanceSet_Xmas_dual_new.mat');
load('InitImpedanceSet_Xmas_dual_v20_bin.mat');

% load('trainedIndex')

addpath('.\OpenSim Model')
addpath('.\actor_critic')
addpath('.\multilayerperceptron')
addpath('.\Agent')
%% Tuner structure Initializaion
for testID=1:5
    
    %Desired number of runs
    MaxRun=1; 
    %Desired number of trials
    MaxTr=50;
    %Desired number of steps
    MaxSt=500; % was set to 500 in previous OpenSim experiments

    targetImpedance = [ 2.2    0.11   -0.2950    0.18000    0.1008    -0.0925    0.0400    0.0052   -1.040    0.0553    0.0058   -0.2398];
    featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0224    0.3133   -0.0491    0.2567];
    actionbound=[  0.01     0.01     0.001      0.001;
                   0.01    0.01     0.001     0.001;
                   0.02    0.02     0.02      0.02 ];
                     
    sensorNoise = 0.25*0.6351/180*pi;
    actuatorNoise = 0.01.*reshape(targetImpedance,3,4)';
    

%     Finaltrials = 0;
%     FinalStep = 0;
%     SuccessCnt=[];
%     trackingCnt= 0;
    sd = testID;
    rng(sd,'twister');
    initImpedance = InitialImpedance(testID, :);
    
    
    % OpenSim model loading and initialization
    mpath = ['C:\Users\wrf-i\Documents\OpenSim_tracking\OpenSim Model\WalkerModel_RK_Torque_dual_ba_short.osim'];  
    md = human_prosthesis(mpath);
    md = md.initHP(initImpedance);

    %% target profile
    [md, state] = md.getPerformance();
    targetProfile = md.profile;
    % geting normalization factor on human model
    normFactor = md.performanceStateNorm;
    % set max action magnitude
    Mag = actionbound';

    %% ==========ADP Settings=====================================
    % Binary control ------ CTL_TYPE=1
    % Analylog control ---- CTL_TYPE=0.
    CTL_TYPE = 0;	

    % # of state and action
    nstate = 2;
    naction = 3;
    % # of hidden neuron in actor and critic
    annhidden = 8;
    cnnhidden = 20;
    Bias = 0;

    % create a Qlearning object
    ADP_agent = dHDP_multi(nstate, naction, annhidden, cnnhidden,Bias);
    ADP_agent = ADP_agent.setControlType(CTL_TYPE);% setting control type


%% simualtion start

    runs=1;
    resetHist = []; % record reset times
    statusHist = [];
    disp(['It is ' int2str(runs) ' run.']);
    succADP=[];
    cnt=0;
    resetNN = 0;
    ADPHist=[];

    for  trial = 1:MaxTr
        failure=0; % fail flag
        failReason=0;
        steps=1;
        paceflag = false;
        if trial == 1 % first trial
            %% initial profile
%             trackingCnt = trackingCnt+1;
            md = md.simulate([],initImpedance);
            [md, state] = md.getPerformance();  % raw state
            initRawState = state;
            [state, reward, status] = md.getState(); % normalized state
            initState = state;
        else % reset impedance if needed
            % reset the fail phase
            for phase = 1:4
               if status(phase) == -1
                    imp = initImpedance((phase*3-2):(phase*3));
                    md = failureReset(md, phase, imp);
               end
            end
            % obtain the latest state after reset
            [state, reward, status] = md.getState(); % normalized state
            if min(status) == -1
                md = md.simulate([],initImpedance);
                [md, state] = md.getPerformance();
                [state, reward, status] = md.getState();
            end

        end
        % reset both actor and critic if the model keep falling
        if length(resetHist)>= 3 && sum(resetHist(end-2:end)) <=6
            ADP_agent=ADP_agent.resetANN([1 2 3 4]);
            ADP_agent=ADP_agent.resetNetwork([1 2 3 4]);
        end

        %% add sensor noise
        for p=1:4
            state(p,1) = state(p,1)+sensorNoise*randn()/md.performanceStateNorm(p,1); 
        end
        %%
        % get action based on the ANN and get cost based on CNN   
        [action, value, ADP_agent] = ADP_agent.getAction(state, 0);
        statusHist = [statusHist;reshape(status,1,[])];
        while(steps<MaxSt)  
            
%             if trackingCnt == 20
%                 paceflag = true;
%                 trackingCnt=0;
%                 intactImIndex_new = randi([1,5],1,1);
%                 while intactImIndex == intactImIndex_new
%                     intactImIndex_new = randi([1,5],1,1);
%                 end
%                 intactImIndex = intactImIndex_new;
%                 ImIndexHist=[ImIndexHist;cnt intactImIndex];
%             end
                
            force = action.*Mag; % the force applied to the model

            prevState = state;		%record prev state 

            %Plug in the model, simulate next gait
            md = md.simulate(force);
            
            [state, reward, status] = md.getState(); % normalized state   
    %% add sensor noise
            for p=1:4
                state(p,1) = state(p,1)+sensorNoise*randn()./md.performanceStateNorm(p,1); 
            end
            if min(status) == -1 % failure
                disp(['Trial # ' int2str(trial) ' has  ' int2str(steps) ' steps.']);
                resetHist = [resetHist;cnt steps];
                break;
            end
            
            % calulate LQR reward
            reward=reward+0.1*sum(action.^2,2);
            if ~paceflag
            	ADP_agent = ADP_agent.update(prevState, state, reward,status, [1 2 3 4]); % update for phase =1,2,3,4
            else
                paceflag=false;
            end
            % get action based on the updated ANN        
            [action, value, ADP_agent] = ADP_agent.getAction(state, 0); % next action, #1 is the active block

            for phase = 1:4
                if max(isnan(action(phase,:))) == 1
                    ADP_agent=ADP_agent.resetNetwork(phase);
                    status(phase) = -1;
                end
            end

            if min(status) == -1 % failure
                disp(['Trial # ' int2str(trial) ' has  ' int2str(steps) ' steps.']);
                resetHist = [resetHist;cnt steps];

                break;
            end
            statusHist = [statusHist;reshape(status,1,[])];
            cnt=cnt+1; % for this run
            steps=steps+1;  % step counting for this trial
            
            % fail if exceed the max steps
            if cnt >= MaxSt
               status = [0 0 0 0]'; 
               break;
            end
            
            if cnt > 10
                if min(sum(statusHist(end-9:end,:))>7) >=1
                    disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);

                    break;
                end
            end
%             if cnt > 10
%                 if Finaltrials == 0
%                     if min(sum(statusHist(end-9:end,:))>6) >=1
%                         disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);
%                         succADP=[succADP;ADP_agent];
%                         Finaltrials=Finaltrials+1;  
%                         SuccessCnt=[SuccessCnt,cnt];
%                         trackingCnt=0;
%                         intactImIndex_new = randi([1,5],1,1);
%                         while intactImIndex == intactImIndex_new
%                             intactImIndex_new = randi([1,5],1,1);
%                         end
%                         intactImIndex = intactImIndex_new;
%                         ImIndexHist=[ImIndexHist;cnt intactImIndex];
%                         paceflag=true;
%                         FinalStep=0;
%                     end
%                 else
%                     FinalStep = FinalStep+1;
%                     if min(sum(statusHist(end-9:end,:))>6) >=1 && FinalStep>11
%                         disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);
%                         succADP=[succADP;ADP_agent];
%                         SuccessCnt=[SuccessCnt,cnt];
%                         if Finaltrials <3
%                             trackingCnt=0;
%                             intactImIndex_new = randi([1,5],1,1);
%                             while intactImIndex == intactImIndex_new
%                                 intactImIndex_new = randi([1,5],1,1);
%                             end
%                             intactImIndex = intactImIndex_new;
%                             ImIndexHist=[ImIndexHist;cnt intactImIndex];
%                             paceflag=true;
%                             Finaltrials=Finaltrials+1;
%                             FinalStep=0;
%                         else
%                             Finaltrials=Finaltrials+1;
%                             break;
%                         end
%                     end
%                 end
%             end

        end %end of steps
%         trial = trial+1;
        if cnt > 10
    %         if min(status) >= 1
            if min(sum(statusHist(end-9:end,:))>7) >=1 && Finaltrials==4
                disp(['Trial # ' int2str(trial) ' has succeed in ' int2str(steps) ' steps ']);
                break;
            end
        end
    end % end of trials




%%
    totalSteps = size(normstateHist,1); % total number of steps before success
    totalTrials = trial - 1;
    
end

