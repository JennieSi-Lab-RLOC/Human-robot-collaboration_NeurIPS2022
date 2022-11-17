%%
classdef human_prosthesis
    properties
        modelPath;
        initialImpedance;
        targetImpedance;
        phaseFlag;
        phaseCnt;
        
        subEval;
        subEvalCnt;
        subThreshold;
        Pmav;               % mean absolute value
        Dmav;   
        Pmean;
        Dmean;
        Pmin;
        Dmin;
        subReward;
        subCost;           
        subReinf;
        
        
        subRflag;
        autoFlag;
        autoStatus;
        autoScaleCnt;
        autoIP;
        autoDeviation;
        
        
        %% control varibles
        ui;
        u;                  %action
        ulist;
        utype;
        unoise;
        uMagScale;
        uMag;
        uMagInit;

        %% state varibles
        x;
        xname;
        
        prosthesisState;
        impedance;          % 12 elements row vector(3 per phase)
        
        humanState;
        bodyWeight;
        height;
        
        profile;                % knee profile
        profileInit;            % initial knee profile
        transition;             % transition timing
        feature;                % profile feature(duration, peakvalue)
        featurePast;            % profile feature of last step
        featureTarget;          % target feature(duration, peakvalue)
        featureState;           % feature state(featureRaw - featureTarget; duration error, peak error)
        featureStateS;
        featureStatePast;       % feature state of last gait cycle  
        performanceStateNorm;   % performance state normalization( feature state normalization )       
        performanceState;       % (featureState, featureState-featureStatePast)
        
        %% reward varibles   
        desire;         % desire value
        sindex;         % target state index
        splus;          % safe range
        sminus;
        rewardtype;     % rewardtype: 1 for smooth reward, 0 for not smooth(1 or 0.1) reward
        costconstant;
        costfailure;
        costsucceed;
        w;
        rewards;
        status;
        
        %% history varibles        
        uhist;          % action
        xhist;          % state
        shist;          % status
        ihist;          % impedance
        rhist;          % reward
        scalehist;
        histcnt;
        savecnt;
        
    end
    methods
        
        %% create human prosthesis platform
        function obj = human_prosthesis(mpath)
            % initial impedance featureTarget and model path parameters
            if nargin == 0
                tfolder = cd(cd('.'));
                mpath = [tfolder,'\OpenSim Model\WalkerModel_RK_Torque_Prescribed.osim'];                
            end
    
            obj.autoFlag = 0;
            obj.subRflag = 1;           
            
            obj.modelPath = mpath;
            obj.phaseCnt = 4;
            obj.histcnt = 0;
            obj.savecnt = 0;
            obj.autoStatus = zeros(4,1);
            obj.status = zeros(4,1);
            for i = 1 : obj.phaseCnt
                obj.uhist{i} = [];
                obj.xhist{i} = [];
            end
            obj.ihist = [];
            obj.shist = [];
            obj.rhist = [];
            obj.scalehist = [];
            obj.subReinf = 0.6;
            obj.uMagInit = 0.1;
        end
       
        %% Set initial impedance parameters and target features
        % impedance 1*12 or 4*3
        % feature target 1*8
        function md = initHP(md, impedance, featureTarget)
            %% initial impedance featureTarget and model path parameters
            if nargin == 1
                impedance = [3.5672   0.0775  -0.4162  0.2146  0.0656  -0.0550	0.0921  0.0106  -1.1285 0.0489  0.0057  -0.2792];
                featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0174    0.3100   -0.0391    0.2567];
            elseif nargin == 2
                featureTarget = [-0.3033    0.1100   -0.0356    0.2733   -1.0174    0.3100   -0.0391    0.2567];

            end
            % resize impedance parameters for a matrix format as
            % impedancematrix
            if size(impedance,1) == 1
                impedancemaxtrix = reshape(impedance,3,4)';
            else
                impedancemaxtrix = impedance;
                impedance = reshape(impedance', 1, numel(impedance));
            end
            
            targetimpedance = [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];
            
            %% Initialize model
            md.subEvalCnt = 0;
            md.initialImpedance = impedancemaxtrix;
            md.targetImpedance = reshape(targetimpedance,3,4)';
            md.featureTarget = reshape(featureTarget,2,4)';
            
            %% control initialization
            md.impedance = impedancemaxtrix;    
            md.uMagScale =  md.uMagInit*ones(4,1);
            md.ui = zeros(size(impedancemaxtrix));                    %initial control(scaler or colum vector)
            md.u = md.ui;
            md.unoise = 0;
            md.ulist = [];                                     %action range or action list;             
            md.utype = 0;                                      %0 for continue action space; action number for discrete action space
            md.uMag = diag(md.uMagScale)*impedancemaxtrix;                         %action magnitude
            
            %% state initialization
            [md.profile, md.transition] = Lowerlimb_Motion(md.modelPath, impedance);
            md.profileInit = md.profile;
            md = md.getFeature();
            md.featurePast = md.feature;
            md.xname = {'peakE','DpeakE','durationE','DdurationE'};
            
            %% reward initialization
            deviation =  md.feature - md.featureTarget;
            md.rewardtype = 0;                              %rewardtype: 1 for smooth reward, 0 for not smooth(1 or 0.1) reward
            md.costsucceed = 0;
            md.costconstant = 0;                            % cost of states between splus and sminus
            md.costfailure = 1; 
            md.splus = 0.02*ones(size(deviation));
            md.splus(:,1) = 0.0175;                      % +- 1 degree success
            if md.autoFlag
                md.sminus = abs(deviation);             % limitation range is 2*md.featureState(Initial featureState)
                md.sminus(md.sminus <= md.splus) = md.splus(md.sminus <= md.splus);
                md.sminus = md.sminus*2;
            else
                safedegree1sd =[7,5,6,4]'; % [6,5,7,4];             % fixed failure range
                saferadian = 2*safedegree1sd*3.14/180;

                targetD = [0.1133    0.2733    0.3133    0.2500]';

                safetime = [0.2; 0.2 ;0.2; 0.2];
                md.sminus = [saferadian, safetime];
            end         
            md.w = atanh(sqrt(0.95))./md.splus;                 % if rewardtype is one, uncomment this line.
            
            ft = md.sminus;
            md.performanceStateNorm = [ft,ft*2];
            derivative = md.feature - md.featurePast;
            md.performanceState = [deviation, derivative]; 
            
            % sub reward initialization
            md.subEvalCnt = 0;
            md.autoDeviation =  md.performanceState(:,1:2);
            md.subCost = zeros(4,2);
            md.subReward = zeros(4,2);
            md.Pmav = md.performanceState(:,3)*0; 
            md.Dmav = md.performanceState(:,4)*0;
            md.Pmin = ones(size(md.performanceState(:,1))); 
            md.Dmin = ones(size(md.performanceState(:,2)));
            md.Pmean = md.performanceState(:,1)*0; 
            md.Dmean = md.performanceState(:,2)*0;
        end
        
        %% set sub reinforcement signal(-0.4,-0.6,-0.8)
        function md = setSubReinf(md, subreinf)
            md.subReinf = subreinf;
        end   
        
        %%
        function md = failureReset(md, i, impedance)
            % reset only phase i
%             md.impedance(i,:) = md.initialImpedance(i,:);
            md.impedance(i,:) = impedance;
%             for i = 1: 4
%                if md.status(i) == -1
%                     md.impedance(i,:) = md.initialImpedance(i,:);
%                     md.uMagScale(i) = md.uMagInit;
% %                     md.uMag(i,:) =  md.uMagScale(i)*md.initialImpedance(i,:);
%                     md.uMag(i,:) =  md.uMagScale(i)*md.targetImpedance(i,:);
%                     break;
%                end
%             end
            impedancev = reshape(md.impedance', 1, numel(md.impedance));
            [md.profile, md.transition] = Lowerlimb_Motion(md.modelPath, impedancev);
            md = md.getFeature();
            md.featurePast = md.feature;
            deviation =  md.feature - md.featureTarget;
            
            %% auto flag is one for auto scaling state magnitude

            derivative = md.feature - md.featurePast;
            md.performanceState = [deviation, derivative]; 
                 
            md.subEvalCnt = 0;              
            md.autoDeviation =  md.performanceState(:,1:2);
            md.subCost = zeros(4,2);
            md.subReward = zeros(4,2);
            md.Pmav = md.Pmav*0;          % mean absolute value of Pe'
            md.Dmav = md.Dmav*0;          % mean absolute value of De'
            md.Pmean = md.Pmean*0;
            md.Dmean = md.Dmean*0;                     
            md.Pmin = ones(size(md.Pmin));
            md.Dmin = ones(size(md.Dmin));
        end

        %%
        function [md, status]= subEvaluate(md)
            md.subEvalCnt = md.subEvalCnt + 1;
            md.Pmav = md.Pmav + abs(md.performanceState(:,3));          % mean absolute value of Pe'
            md.Dmav = md.Dmav + abs(md.performanceState(:,4));          % mean absolute value of De'
            md.Pmean = md.Pmean + md.performanceState(:,1); 
            md.Dmean = md.Dmean + md.performanceState(:,2); 


            status = zeros(4,1);
            % Evaluate the system when subEvalCnt is greater than 6.
            if md.subEvalCnt >= 6
                error = md.autoDeviation - md.performanceState(:,1:2); %% ???????????
                grad = error.*md.performanceState(:,1:2);
                md.Pmav = md.Pmav./md.subEvalCnt;
                md.Dmav = md.Dmav./md.subEvalCnt;
                md.Pmean = md.Pmean./md.subEvalCnt;
                md.Dmean = md.Dmean./md.subEvalCnt;                
                Pe = abs(md.performanceState(:,1));
                De = abs(md.performanceState(:,2));
                for i = 1:md.phaseCnt
                    % evaluate peak error
                    if  Pe(i) > md.splus(i,1) && Pe(i) < md.sminus(i,1)
                        if abs(error(i,1)) >= md.Pmav(i)                % variation is bigger than the absolute mean value
                            if sign(grad(i,1)) >= 0                     % right direction
                                md.subCost(i,1) = 0;
                                md.subReward(i,1) = md.subReward(i,1) + 2;
                            else                                        % wrong direction
                                md.subCost(i,1) = md.subCost(i,1) + 2;
                                md.subReward(i,1) = 0;
                            end
                        else                                            % variation is small than the absolute mean value
                            if sign(grad(i,1)) >= 0                     % right direction
                                %md.subCost(i,1) = 0;
                                md.subReward(i,1) = md.subReward(i,1) + 1;
                            else                                        % wrong direction
                                md.subCost(i,1) = md.subCost(i,1) + 1;
                                %md.subReward(i,1) = 0;
                            end 
                        end
                    end
                    
                    % evaluate duration error
                    if  De(i) > md.splus(i,2) && De(i) < md.sminus(i,2)
                        if abs(error(i,2)) >= md.Dmav(i)                % variation is bigger than the absolute mean value
                            if sign(grad(i,2)) >= 0                     % right direction
                                md.subCost(i,2) = 0;
                                md.subReward(i,2) = md.subReward(i,2) + 2;
                            else                                        % wrong direction
                                md.subCost(i,2) = md.subCost(i,2) + 2;
                                md.subReward(i,2) = 0;
                            end
                        else                                            % variation is small than the absolute mean value
                            if sign(grad(i,2)) >= 0                     % right direction
                                %md.subCost(i,2) = 0;
                                md.subReward(i,2) =  md.subReward(i,2) + 1;
                            else                                        % wrong direction
                                md.subCost(i,2) = md.subCost(i,2) + 1;
                                %md.subReward(i,2) = 0;
                            end 
                        end
                    end
                    
                    % when Pmin and Dmin are both in target range, scale
                    % down the uMag.
                    if  abs(md.Pmin(i)) <= md.splus(i,1) &&  abs(md.Dmin(i)) <= md.splus(i,2) && md.savecnt >= 50 %sum(md.status(i,:)) == 2 
                        md.subCost(i,:) = [0, 0];
                        md.subReward(i,:) = [0, 0];
                        status(i) = 1;
                        % scale down uMag with lower limitation
                        md.uMagScale(i) = md.uMagScale(i)*0.5;
                        if md.uMagScale(i) < 0.001
                             md.uMagScale(i) = 0.001;
                        end
%                         md.uMag(i,:) =  md.uMagScale(i)*md.initialImpedance(i,:);
                        md.uMag(i,:) =  md.uMagScale(i)*md.targetImpedance(i,:);
                    end 
               
                    % scale up the uMag when reward score is greater than 4
                    if sum(md.subReward(i,:)) >= 4
                        md.subReward(i,:) = [0, 0];
                        status(i) = 0.5;
                        % scale up uMag with higher limitation
                        md.uMagScale(i) = md.uMagScale(i)*1.2;
                        if md.uMagScale(i) > 0.1
                             md.uMagScale(i) = 0.1;
                        end
%                         md.uMag(i,:) =  md.uMagScale(i)*md.initialImpedance(i,:);
                        md.uMag(i,:) =  md.uMagScale(i)*md.targetImpedance(i,:);
                    end
                    
                    % sub reinforcement when cost score is greater than 4
                    if sum(md.subCost(i,:)) >= 4 
                        md.subCost(i,:) = [0, 0];
                        status(i) = md.subReinf;
                        if md.status(i) == 0 && md.subRflag
                            md.status(i) = md.subReinf;
                            md.rewards(i) = md.subReinf;
                        end
                    end
                end
                [md.subReward,md.subCost,status]

                md.autoDeviation =  md.performanceState(:,1:2);
                md.subEvalCnt = 0;
                md.Pmav = md.Pmav*0;          % mean absolute value of Pe'
                md.Dmav = md.Dmav*0;          % mean absolute value of De'
                md.Pmean = md.Pmean*0;
                md.Dmean = md.Dmean*0;                     
                md.Pmin = ones(size(md.Pmin));
                md.Dmin = ones(size(md.Dmin));
            end      
        end
        
        
        %% simulate system under action, which is 4-by-3 matrix
        %   4 phase in one gait cycle
        %   3 impedance parameters for each phase
        function md = simulate(md, action, impedance)  
            if nargin > 2
                md.impedance = reshape(impedance,3,4)';  % fix a problem here
                impedances =  impedance;
            else
                md.u = action;
                md.impedance = md.impedance + md.u;
                impedances =  md.impedance;
            end
            % add impedance increment to the impedance parameters
%             md.u = action.*md.uMag+md.unoise*2*(rand(size(action)) - 0.5);

            
            % reshape the impedance parameters to 1*12 vector for simulation
            if size(impedances,1) == 4
                impedances = reshape(impedances', 1, numel(impedances));
            end
            
            [md.profile, md.transition] = Lowerlimb_Motion(md.modelPath, impedances);
            md = md.getFeature();
            md = md.getPerformance();
            %
            md = md.reward(md.feature - md.featureTarget);
           md = md.subEvaluate();
            
            % save action, performance
            for i = 1 : size(action,1)
                md.uhist{i} = [md.uhist{i}; action(i,:)];
                md.xhist{i} = [md.xhist{i}; md.performanceState(i,:)];
            end
            % save impedance and scale number
            md.ihist= [md.ihist; impedances];
            md.shist= [md.shist; md.status'];
            md.rhist= [md.rhist; md.rewards'];
            md.scalehist = [md.scalehist; md.uMagScale'];
            md.savecnt =  md.savecnt + 1;
            state = md.performanceState
        end
        
        %% 
        function [state, reward, status] = getState(md)
            performance = md.performanceState./md.performanceStateNorm;
            md.x = performance;
            %take impedance parameters into consideration
%             icp = reshape(md.impedance, 3, 4)';                 
%             md.x = [icp, performance]; %
            reward = md.rewards;
            status = md.status;
            state = md.x(:,1:2);
        end
        
        %% rewardtype: 1 for smooth reward, 0 for not smooth(1 or 0.1) reward    
        function  [md, cost, status]= reward(md, error)
            cost = zeros(size(error));                  %[md.costconstant.*ones(1,size(error,2)); zeros(1,size(error,2))];
            status = zeros(size(error));
            if md.rewardtype == 0
                cost(abs(error) >= md.sminus) = md.costfailure;
                status(abs(error) >= md.sminus) = -1;
                cost(abs(error) < md.splus) = md.costsucceed;
                status(abs(error) < md.splus) = 1;
            elseif md.rewardtype == 1
                cost = md.costconstant*(tanh(abs(error).*md.w)).^2;
                cost(abs(error) >= md.sminus) = md.costfailure;
                status(abs(error) >= md.sminus) = -1;
                status(abs(error) < md.splus) = 1;
            else
                cost(abs(error) >= md.sminus) = md.costfailure;
                status(abs(error) >= md.sminus) = -1;
                cost(abs(error) < md.splus) = md.costsucceed;
                status(abs(error) < md.splus) = 1;
                % with operational range, give a small cost
                cost(abs(error) > md.splus & abs(error) < md.sminus) = 0.003;
            end
%             cost = mean(cost,2);
            cost = max(cost, [], 2);
            status = min(status, [], 2);
            md.rewards = cost;
            md.status = status;
        end
        
        %% get the feature of the profile along with transition timeing
        % profile starts from heel strike
        % feature - [peakvalue, duration] 4*2 matrix
        function [md, feature] = getFeature(md, profile, transition, ts)
            if nargin == 1
                profile = md.profile;
                transition = md.transition;
                ts = linspace(0,1,length(profile));             % simulation last for 1s, get time stamp for each point                
            end
            feature = ones(1,8);                % initialize feature vector      
            len = length(profile);              % get profile length
            transtp=floor(len*transition);            
            % transition points not equal to 5, something bad happens
            if length(transition) >= 5
                phacut = ceil((transtp(1:end-1)+transtp(2:end))/2);     % split the profile with transition point
                [stflex, stfi] = min(profile(1:phacut(2)));             % stand flexion peak value and time index
                [stext, stei]= max(profile(   stfi:phacut(3)));            % stand extension peak value and time index
                stei = stei + stfi - 1;                           
                [swflex,swfi] = min(profile(stei:phacut(4)));           % swing flexion peak value and time index
                swfi = swfi + stei - 1;
                [swext, swei]= max(profile(swfi:end));                  % swing extension peak value and time index
                swei = swei + swfi - 1;

                peakDelay=[stfi,stei,swfi,swei];
                peakValue=[stflex,stext,swflex,swext];
                peakDelay = ts(peakDelay);
                peakDelay(2:end)=peakDelay(2:end)-peakDelay(1:end-1);
                featuretmp = [peakValue; peakDelay];                
                if size(featuretmp, 2) == 4
                    feature = reshape(featuretmp,1,[]);          
                end
            end   
            md.feature = reshape(feature, 2, 4)';
        end
        
        %% get the performance state of the human prosthesis system
        % [deviation of feature, derivative of feature] 4*4 matrix
        % feature is [peak value, duration time] 4*2
        function [md, performance] = getPerformance(md)
             deviation = md.feature - md.featureTarget;
             derivative = md.feature - md.featurePast;
             md.featurePast = md.feature;
             performance = deviation;
%              performance = [deviation, derivative];
             md.performanceState = [deviation, derivative];
        end

        %% evaluate a set of impedance parameter to get profile and feature
        function [profile,feature] = evaluateIP(md, impedance)
            if size(impedance,1) == 4
                impedance = reshape(impedance', 1, numel(impedance));
            end
            [md.profile, md.transition] = Lowerlimb_Motion(md.modelPath, impedance);
            profile = md.profile;
            [~, feature] = md.getFeature();
        end
        
        %% generate initial impedance set that within a range(saferadian)      
        function [InitialImpedance] = generateInitIP(md, samples, saveflag)
            if nargin == 2
               saveflag = 0; 
            end
            TargetImpedances = [3.3054    0.0807   -0.4300    0.2695    0.0781   -0.0495    0.0823    0.0088   -1.0770    0.0453    0.0058   -0.2798];
            TargetState = [-0.3038    0.1133   -0.0461    0.2733   -1.0244    0.3133   -0.0309    0.2500];
            %TargetState = [ -0.3035    0.1133   -0.0593    0.2700   -1.0307    0.3100   -0.0511    0.2600];
            it = 0;
            totalcnt = 1;
            InitialImpedance=[];
            safedegree1sd =[7,5,6,4];                       % standard derivation of the normal gait
            rate = 1.5;
            saferadian = rate*safedegree1sd*3.14/180;       % safety limitation of the impedance
            Initialstates = [];
            InitialError = [];
            InitialMotion = [];
            while(it<samples)
                totalcnt = totalcnt + 1;
                InitI = TargetImpedances;
                InitI = InitI.*((rand(1,12)-0.5).*0.2);     % add deviation to the TargetImpedance to generate initial impedance that fit in the safty limitation
                InitI = InitI + TargetImpedances;
                
                %[md.profile, md.transition] = Lowerlimb_Motion(md.modelPath,InitI);
                %[md,Xf] = md.getFeature();
                %Motion = md.profile;
                [Motion, Xf] = md.evaluateIP(InitI);
                Initerror = Xf-TargetState;

                IniterrorR = reshape(Initerror',2,4)';
                Initperror = abs(IniterrorR(:,1))';
                maxerror = max(abs(IniterrorR));
                if ~max(Initperror > saferadian) && (maxerror(2) <= 0.05)   % duration error smaller than 0.05
                    it = it + 1;
                    InitialImpedance=[InitialImpedance;InitI];
                    %InitialAngle = [InitialAngle; InitA];
                    Initialstates = [Initialstates; Xf];
                    InitialError = [InitialError; Initerror];
                    InitialMotion = [InitialMotion,Motion(:,1)];
                end
            end
%             [irow,icol] = size(InitialImpedance);
%             rng(1);
%             sdlist = round(rand(1,irow)*500);
            figure();
            plot(InitialError(:,2:2:end)','r*');
            figure();
            plot(InitialError(:,1:2:end)','r*');
            figure();
            plot(InitialMotion);
            if saveflag
                save('InitImpedanceSet2.mat','InitialImpedance','Initialstates','InitialError','InitialMotion','totalcnt','rate');
            end
        end
        
        %% show the history state and control of the simulation
        function md = showHist(md, clean)
            if nargin == 1
                clean = 0;
            end
            for i = 1 : length(md.phaseFlag)
                if md.phaseFlag(i) == 1
                    figure(i);
                    n = size(md.xhist{1},2) + size(md.uhist{1},2);
                    for j = 1:size(md.xhist{i},2)
                        subplot(n,1,j);
                        plot(md.xhist{i}(:,j));
                        grid on;
                    end
                    k = 1;
                    for j = size(md.xhist{i},2)+1:n
                        subplot(n,1,j);
                        plot(md.uhist{i}(:,k));
                        k = k + 1;
                        grid on;
                    end
                end  
            end          
            if clean >= 1 %&& md.histcnt >= 3000
               save([num2str(clean),'data.mat'],'md');
               md.uhist = [];
               md.xhist = [];
            end
        end
        
    end
end