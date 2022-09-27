%% direct Heuristic Dynamic Programming
%  
%  Copyright (c) 2015 Yue Wen
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
        
        Jpre;
        gamma;
        controlType;
        steps;
        
        JpreHist;
        reinfHist;
        renewHist;
    end
    methods
        function obj = dHDP(nstate, naction, annhidden, cnnhidden, Bias)
            if nargin == 4
                Bias = 0;                                   % set bias for multilayer perceptron
            end
            obj.bias = Bias;
            obj.nState = nstate;
            obj.nAction = naction;
            obj.nSA = nstate + naction;
            
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
            obj.ann = ANN(neurons, actfunction, Bias);                 % create ANN without bias term
            
            obj.reinfHist = 0;
            obj.JpreHist = 0;
            obj.renewHist = 0;
            obj.gamma = 0.95;
            obj.Jpre = 0;
            obj.steps = 0;
            obj.controlType = 0;
        end       
        
        % reset multilayer perceptron
        function mdHDP = resetNetwork(mdHDP)
            mdHDP.cnn = CNN(mdHDP.nAction, mdHDP.cNeurons, mdHDP.cActFunction, mdHDP.bias);
            mdHDP.ann = ANN(mdHDP.aNeurons, mdHDP.aActFunction, mdHDP.bias);
            mdHDP.Jpre = 0;
        end
        
        % set controlType to one for discrete control problem
        function mdHDP = setControlType(mdHDP,controlType)
            mdHDP.controlType = controlType;
        end
        
        function mdHDP = initialize(mdHDP, state)
            [action, value, mdHDP] = bestAction(mdHDP, state);
            mdHDP.Jpre = value;
        end
        
        %update the action network and critic network from reinforcement signal
        function mdHDP = update(mdHDP, state, reinforcement)
            mdHDP.steps = mdHDP.steps + 1;
            % if reinforcement signal is -1, reset learning rate. Or
            % annealing learning rate
%             if reinforcement == 1
%                 mdHDP.cnn = mdHDP.cnn.learnRateReset();
%                 mdHDP.ann = mdHDP.ann.learnRateReset();
%             elseif rem(mdHDP.steps,5)==0
%                 mdHDP.cnn = mdHDP.cnn.learnRateDecay();
%                 mdHDP.ann = mdHDP.ann.learnRateDecay();
%             end
            
            %% update the critic network
            target = mdHDP.Jpre - reinforcement;
            %state = state + 0.0001*rand(size(state));
            action = mdHDP.ann.getAction(state);
%             if target > 1
%                target = 1; 
%             end
            
            mdHDP.cnn = mdHDP.cnn.bpUpdate([state, action], target, mdHDP.gamma);
            
            %% update the action network
            [mdHDP.ann, J] = mdHDP.ann.bpUpdate(mdHDP.cnn, state);
            
            % update Jpre
            if reinforcement == 1
                mdHDP.Jpre = 0;
            else        
                mdHDP.Jpre = J;
            end
            
%             % check ann network weight
%             ann_std1 = std(std(abs(mdHDP.ann.nn.W{1})));
%             ann_std2 = std(std(abs(mdHDP.ann.nn.W{2})));
%             if ann_std1 < 10^-3 || ann_std2< 10^-3
%                 mdHDP.ann = ANN(mdHDP.aNeurons, mdHDP.aActFunction, mdHDP.bias);
%                 [mdHDP.ann, J] = mdHDP.ann.bpUpdate(mdHDP.cnn, state);
%                 mdHDP.renewHist = [mdHDP.renewHist; 1];
%             else
%                 mdHDP.renewHist = [mdHDP.renewHist; 0];
%             end

            % save reinforcement signal and J   
            mdHDP.reinfHist = [mdHDP.reinfHist; reinforcement];
            mdHDP.JpreHist = [mdHDP.JpreHist;  mdHDP.Jpre];
        end
        
        % get best action at state or random action without state parameter
        function [action, value, mdHDP] = getAction(mdHDP, state)
            value = 1;
            if nargin == 1
                action = mdHDP.randAction();
            else
                [action, value, mdHDP] = mdHDP.bestAction(state);
            end
            % action type is 1 for discrete control
            daction = ones(size(action));
            if mdHDP.controlType
                daction( action<0 ) = -1;
                action = daction;
            end
        end
        
        % get action based on the ANN and get cost based on CNN         
        function [action, value, mdHDP] = bestAction(mdHDP, state)
            [action, mdHDP.ann] = mdHDP.ann.getAction(state);
            [value, mdHDP.cnn] = mdHDP.cnn.getCost([state, action]);
        end
        
        % get random action between -1 to 1        
        function [action] = randAction(mdHDP)
            action = 2*(rand(size(1,mdHDP.nAction))-0.5);
        end
        
        % plot reinforcement signal history and J history
        function mdHDP = showHist(mdHDP, clean)
            if nargin == 1
                clean = 0;
            end
            figure(2);
            subplot(2,1,1);
            plot(mdHDP.JpreHist);
            title('J history');
            grid on;
            subplot(2,1,2);
            plot(mdHDP.reinfHist);
            title('R history');
            grid on;           
            if clean == 1
                mdHDP.JpreHist = [];
                mdHDP.reinfHist = [];
            end   
        end
        
        function mdHDP = cleanHDP(mdHDP)
            mdHDP.JpreHist = [];
            mdHDP.reinfHist = [];
            mdHDP.Jpre = 0;
            mdHDP.steps = 0;
            mdHDP.cnn = mdHDP.cnn.cleancnn();
            mdHDP.ann = mdHDP.ann.cleanann();  
        end
            
    end
end