%% Neural fitted Q literation method for continous action
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%
classdef NFQAC
    properties
        
        cnn;                        % critic neural network
        cNeurons;                   % neurons of each layer in CNN
        cActFunction;               % activation function of each layer in CNN
        ann;                        % action neural network
        aNeurons;                   % neurons of each layer in ANN
        aActFunction;               % activation function of each layer in ANN
        
        actionList;
        actionType;
        nState;
        nAction;
        nSA;
        
        gamma;
        tupleList;
        tupleCount;
        epoches;
        epslon;
    end
    methods
        function obj = NFQAC(nstate, naction, annhidden, cnnhidden)
            obj.nState = nstate;
            obj.nAction = naction;
            obj.nSA = nstate + naction;
            % create ann network 
            neurons=[nstate, annhidden, naction];                    % Structure of mlp, nodes of input layer, nodes of hidden layer 1, ..., nodes of output layer
            actfunction = cell(1, length(neurons));
            actfunction{1} = 'linear';                              % activation function of input layer
            for i = 2:length(neurons)-1                             % activation function of hidden layers
               actfunction{i} = 'tansig';
            end
            actfunction{end} = 'tansig';                            % activation function of output layer      
            obj.aNeurons = neurons;
            obj.aActFunction = actfunction;
            obj.ann = ANN(neurons, actfunction);   
            
            % create cnn network
            neurons=[nstate+naction, cnnhidden, 1];                    % Structure of mlp, nodes of input layer, nodes of hidden layer 1, ..., nodes of output layer
            actfunction = cell(1, length(neurons));
            actfunction{1} = 'logsig';                              % activation function of input layer
            for i = 2:length(neurons)-1                             % activation function of hidden layers
               actfunction{i} = 'logsig';
            end
            actfunction{end} = 'logsig';                            % activation function of output layer      
            obj.cNeurons = neurons;
            obj.cActFunction = actfunction;            
            obj.cnn = CNN(naction, neurons, actfunction); 
            
            obj.tupleList = [];            
            obj.tupleCount = 0;
            obj.gamma = 0.95;
            obj.epoches = 20;
            obj.epslon = 0.0;
        end    
        
        % reset CNN and ANN, and clean history tuples
        function nfq = resetNetwork(nfq)
            nfq.cnn = CNN(nfq.nAction, nfq.cNeurons, nfq.cActFunction);
            nfq.ann = ANN(nfq.aNeurons, nfq.aActFunction);
            nfq.tupleList = [];
            nfq.tupleCount = 0;
        end

        % set parameters
        function nfq = setMisc(nfq, name, value)
            if strcmp(name,'gamma')                                 % set gamma for discount factor
                nfq.gamma = value;
            elseif strcmp(name,'epoch')                             % set epoches for fitting   
                nfq.epoches = value;
            elseif strcmp(name,'epslon')                            % set epslon for epslon-greedy
                nfq.epslon = value;
            end
        end
        
        % save tuple in the tupleList        
        function nfq = saveTuple(nfq, state, action, statenext, reward, status)
            % if there are just two parameters pass through, the second one
            % is the combination as a tuple
            if nargin == 2
               tuple = state;
            else
               tuple = [state, action, statenext, reward, status];
            end
            nfq.tupleList = [nfq.tupleList; tuple];
            nfq.tupleCount = nfq.tupleCount + 1;
        end
        
        % train the cnn network with all history tuples
        function nfq = train(nfq, epoches)
            %fprintf('number of tuples: %d\n', nfq.tupleCount);
            if nargin == 2
               nfq.epoches = epoches;
            end
            %fprintf('number of tuples: %d\n', nfq.tupleCounter);
            % separate the data
            inputs = nfq.tupleList(:, 1:nfq.nSA);
            states = nfq.tupleList(:, 1:nfq.nState);
            nextState = nfq.tupleList(:, nfq.nSA+1:end-2);            
            reward = nfq.tupleList(:, end-1);
            status = nfq.tupleList(:, end);
            for i = 1:nfq.epoches                                           % value update with one iteration; policy update with multiple iteration until converge
                [~, Qpre, nfq] = nfq.bestAction(nextState);  
                Qvalue = reward + nfq.gamma.*Qpre;                          %( reward >= nfq.cc).*(nfq.gamma.*Qpre);
                Qvalue(status == -1) = reward(status == -1);
                nfq.cnn = nfq.cnn.fit(inputs, Qvalue);
            end
            
            nfq.ann = nfq.ann.rpUpdate(nfq.cnn, states);
            nfq.epslon = nfq.epslon*0.98;
        end
        
        % get best action at state or random action without state parameter
        function [action, value] = getAction(nfq, state)
            value = 1;
            eps = rand(1);
            if nargin == 1 || eps < nfq.epslon
                action = nfq.randAction();
            else
                [action, value] = nfq.bestAction(state);
            end
        end
        
        % get action based on the ANN and get cost based on CNN 
        function [action, value, nfq] = bestAction(nfq, state)
            [action, nfq.ann] = nfq.ann.getAction(state);
            [value, nfq.cnn] = nfq.cnn.getCost([state, action]);          
        end
        
        % get random action between -1 to 1
        function [action] = randAction(nfq)
            action = 2*(rand(size(1,nfq.nAction))-0.5);
        end     
    end
end