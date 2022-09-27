 %% Q-Learning
%  
%  Copyright (c) 2019 Xiang Gao
%  $Revision: 0.10 $
%
%
classdef dHDP
    properties
        cnn;                        % critic neural network
        cNeurons;                   % neurons of each layer in CNN
        cActFunction;               % activation function of each layer in CNN
        ann;                        % action neural network
        aNeurons;                   % neurons of each layer in ANN
        aActFunction;               % activation function of each layer in ANN
        bias;
        
        nState;
        nAction;
        nSA;
        Action;
        
        gamma;
        controlType;
        steps;
        
        reinfHist;
        renewHist;
        Phase;
        
        knowledgeGuidedMode;
        exADP;
        costhist;
        rewardhist;
        fv1hist;
        fv2hist;
        fv3hist;
        afv3hist;
        w1hist;
        w2hist;
        bv1hist;
        exValueHist;
        naValueHist;
        aouthist
    end
    methods
        function obj = dHDP(nstate, naction, annhidden, cnnhidden,Phase, Bias)
            if nargin == 5
                Bias = 0;                                   % set bias for multilayer perceptron
            end
            obj.knowledgeGuidedMode = false;
            obj.bias = Bias;
            obj.nState = nstate;
            obj.nAction = naction;
            obj.nSA = nstate + naction;
            obj.Phase =Phase;
            obj.costhist = [];
            obj.rewardhist=[];
            obj.exValueHist=[];
            obj.naValueHist=[];
            if obj.knowledgeGuidedMode
                if Phase == 1
                    load('C:\Users\wrf-i\Documents\KGQL_OpenSim_echo_guide_square\KGQL_OpenSim_2020\exADP8.mat')
                    exADP = exADP.blocks{1,1};
                elseif Phase == 2
                    load('C:\Users\wrf-i\Documents\KGQL_OpenSim_echo_guide_square\KGQL_OpenSim_2020\exADP8.mat')
                    exADP = exADP.blocks{1,2};
                elseif Phase == 3
                    load('C:\Users\wrf-i\Documents\KGQL_OpenSim_echo_guide_square\KGQL_OpenSim_2020\exADP8.mat')
                    exADP = exADP.blocks{1,3};
                else
                    load('C:\Users\wrf-i\Documents\KGQL_OpenSim_echo_guide_square\KGQL_OpenSim_2020\exADP8.mat')
                    exADP = exADP.blocks{1,4};
                end
                obj.exADP=exADP; 
            end
                
                
            % create cnn network
            neurons=[nstate+naction, cnnhidden, 1];                 % Structure of mlp, nodes of input layer, nodes of hidden layer 1, ..., nodes of output layer
            actfunction = cell(1, length(neurons));
            actfunction{1} = 'linear';                              % activation function of input layer
            for i = 2:length(neurons)-1                             % activation function of hidden layers
               actfunction{i} = 'tansig';
            end
            actfunction{end} = 'linear';                            % activation function of output layer      
            obj.cNeurons = neurons;
            obj.cActFunction = actfunction; 
            obj.cnn = CNN(naction, neurons, actfunction, Bias);        % create CNN without bias term
            if obj.knowledgeGuidedMode
                ratio = 0.5;
                obj.cnn.nn.W{1} = (1-ratio)*obj.cnn.nn.W{1} + ratio*exADP.cnn.nn.W{1};
                obj.cnn.nn.W{2} = (1-ratio)*obj.cnn.nn.W{2} + ratio*exADP.cnn.nn.W{2}; 
            end
            % create ann network 
            neurons=[nstate, annhidden, naction];                   % Structure of mlp, nodes of input layer, nodes of hidden layer 1, ..., nodes of output layer
            actfunction = cell(1, length(neurons));
            actfunction{1} = 'linear';                              % activation function of input layer
            for i = 2:length(neurons)-1                             % activation function of hidden layers
               actfunction{i} = 'tansig';
            end
            actfunction{end} = 'tansig';                            % activation function of output layer      
            obj.aNeurons = neurons;
            obj.aActFunction = actfunction;
            obj.ann = ANN(neurons, actfunction, Bias, obj.knowledgeGuidedMode);                 % create ANN without bias term
%             obj.ann = ANN(neurons, actfunction, Bias, false);
            obj.reinfHist = 0;
            obj.fv1hist=[];
            obj.fv2hist=[];
            obj.fv3hist=[];
            obj.afv3hist=[];
            obj.w1hist=[];
            obj.w2hist=[];
            obj.bv1hist=[];
            
            obj.aouthist=[];
            obj.renewHist = 0;
            obj.gamma = 0.90;
            obj.steps = 0;
            obj.controlType = 0;
        end       
        
        % reset multilayer perceptron
        function mQlearning = resetNetwork(mQlearning)
            mQlearning.cnn = CNN(mQlearning.nAction, mQlearning.cNeurons, mQlearning.cActFunction, mQlearning.bias);
            mQlearning.ann = ANN(mQlearning.aNeurons, mQlearning.aActFunction, mQlearning.bias,mQlearning.knowledgeGuidedMode);
        end
        
        % set controlType to one for discrete control problem
        function mQlearning = setControlType(mQlearning,controlType)
            mQlearning.controlType = controlType;
        end
        
        function mQlearning = initialize(mQlearning, state)
            [action, value, mQlearning] = bestAction(mQlearning, state);
        end
        
        %update the action network and critic network from reinforcement signal
        function mQlearning = update(mQlearning, prevState, state, reinforcement,status, exADP)

            mQlearning.steps = mQlearning.steps + 1;
            exADP = mQlearning.exADP;
            %% update the critic network
            if status == 1
                prevAction = mQlearning.ann.getAction(prevState);
            else
                prevAction = mQlearning.ann.getAction(prevState);
            end
            action = mQlearning.ann.getAction(state);
            value = mQlearning.cnn.getCost([state, action]); % value for the next state
            mQlearning.costhist = [mQlearning.costhist;mQlearning.cnn.getCost([prevState,prevAction])];
            mQlearning.rewardhist = [mQlearning.rewardhist;reinforcement];
%             ratio = 1;`

            ratio = mQlearning.getRatio(); 
            if mQlearning.knowledgeGuidedMode 
%                 [~,exValue] = mQlearning.ann.getActionGradient(state, action);
                exValue = exADP.cnn.getCost([state, action]);% external value from another ADP
                mQlearning.exValueHist = [mQlearning.exValueHist;exValue];
                mQlearning.naValueHist = [mQlearning.naValueHist;value];
%                 value = value + ratio*exValue;
                value = (1-ratio)*value + ratio*exValue;
            end
            
            target = reinforcement + mQlearning.gamma*value;
            
            mQlearning.cnn = mQlearning.cnn.bpUpdateQLearning([prevState, prevAction], target);
            
            %% update the action network
            if status ~=1
                [mQlearning.ann, J ] = mQlearning.ann.bpUpdateQLearning(mQlearning.cnn, state, exADP, ratio);
            end
            % save reinforcement signal and J   
            mQlearning.reinfHist = [mQlearning.reinfHist; reinforcement];
            mQlearning.fv1hist = [mQlearning.fv1hist; mQlearning.cnn.nn.FV{1,1}];
            mQlearning.fv2hist = [mQlearning.fv2hist; mQlearning.cnn.nn.FV{1,2}];
            mQlearning.fv3hist = [mQlearning.fv3hist; mQlearning.cnn.nn.FV{1,3}];
            mQlearning.afv3hist = [mQlearning.afv3hist; mQlearning.ann.nn.FV{1,3}];
            mQlearning.w1hist = [mQlearning.w1hist; reshape(mQlearning.cnn.nn.W{1,1},1,[])];
            mQlearning.w2hist = [mQlearning.w2hist; mQlearning.cnn.nn.W{1,2}'];
            mQlearning.bv1hist = [mQlearning.bv1hist; mQlearning.cnn.nn.BV{1,1}];
            mQlearning.aouthist = [mQlearning.aouthist; mQlearning.ann.output];
            
        end
        
        function ratio = getRatio(mQlearning)
           if mQlearning.knowledgeGuidedMode && mQlearning.steps < 20
%            if mQlearning.knowledgeGuidedMode && mQlearning.steps == 1 % kick start
%                 ratio = 1; 
                ratio = 0.5*(0.8^mQlearning.steps); 
%                 ratio = 0.5^mQlearning.steps;
           else
              ratio = 0;
           end
        end
        
        % get best action at state or random action without state parameter
        function [action, value, mQlearning] = getAction(mQlearning, state)
            value = 1;
            if nargin == 1
                action = mQlearning.randAction();
            else
                [action, value, mQlearning] = mQlearning.bestAction(state);
            end
            % action type is 1 for discrete control
            daction = ones(size(action));
            if mQlearning.controlType
                daction( action<0 ) = -1;
                action = daction;
            end
        end
        
        % get action based on the ANN and get cost based on CNN         
        function [action, value, mQlearning] = bestAction(mQlearning, state)
            [action, mQlearning.ann] = mQlearning.ann.getAction(state);
            [value, mQlearning.cnn] = mQlearning.cnn.getCost([state, action]);
        end
        
        % get random action between -1 to 1        
        function [action] = randAction(mQlearning)
            action = 2*(rand(size(1,mQlearning.nAction))-0.5);
        end
        
        % plot reinforcement signal history and J history
        function mQlearning = showHist(mQlearning, clean)
            if nargin == 1
                clean = 0;
            end
            figure(2);
            subplot(2,1,1);
            plot(mQlearning.JpreHist);
            title('J history');
            grid on;
            subplot(2,1,2);
            plot(mQlearning.reinfHist);
            title('R history');
            grid on;           
            if clean == 1
                mQlearning.JpreHist = [];
                mQlearning.reinfHist = [];
            end   
        end
        
        function mQlearning = cleanHDP(mQlearning)
            mQlearning.reinfHist = [];
            mQlearning.steps = 0;
            mQlearning.cnn = mQlearning.cnn.cleancnn();
            mQlearning.ann = mQlearning.ann.cleanann();  
        end
            
        function mQlearning = resetANN(mQlearning)
            mQlearning.ann = ANN(mQlearning.aNeurons, mQlearning.aActFunction, mQlearning.bias,mQlearning.knowledgeGuidedMode);
        end
    end
end