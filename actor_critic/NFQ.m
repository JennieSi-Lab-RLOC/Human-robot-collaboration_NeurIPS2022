%% Neural fitted Q literation method
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%
classdef NFQ
    properties
        %% critic network 
        cnn;
        activationFunction;                 % activation function for critic network
        inputNode;                          % number of nodes in input layer
        hiddenNodes;                        % numbers of nodes in hidden layers
        outputNode;                         % number of nodes in output layer 
        
        nState;                             % number of state                     
        nAction;                            % number of action
        actionList;                         % action list for discrete control
        actionCount;                        % number of elements in actionList

        %% tuple is consisted of [state, action, stateNext, reward, status]
        % reward is the instantanous cost
        % status is simulation state, failure as -1, succeed as 1, normal working as 0
        tupleList;                          % all past tuples are saved in the tupleList
        tupleCounter;                       % number of tuples in the tupleList
        
        epoches;

        gamma;                              % Discount factor for J(t) = reward(t) + gamma*J(t-1); default as one
        epslon;                             % random action with epslon-greedy explore method; default as zero
    end
    methods
        function obj = NFQ(nstate,naction,hidden,activefunction)
            %% initial variables for multilayer perceptron
            obj.nState = nstate;
            obj.nAction = naction;
            obj.inputNode = nstate + naction;
            obj.hiddenNodes = hidden;
            obj.outputNode = 1;                                         % output nodes is 1 here
            neurons=[obj.inputNode, obj.hiddenNodes, obj.outputNode];                % Structure of mlp, nodes of input layer, nodes of hidden layer 1, ..., nodes of output layer
            actfunction = cell(1, length(neurons));
            if nargin == 3                                              % The activation function is logsig in default 
                actfunction{1} = 'logsig';                              % activation function of input layer
                for i = 2:length(neurons)-1                             % activation function of hidden layers
                    actfunction{i} = 'logsig';
                end
                actfunction{end} = 'logsig';                            % activation function of output layer
            else
                actfunction = activefunction;
            end
            obj.activationFunction = actfunction;
            obj.cnn = CNN(naction, neurons, actfunction);               % create critic neural network
            
            %% initial tuple list and 
            obj.tupleList = [];
            obj.tupleCounter = 0;
            obj.epoches = 20;                                           % epoches for fitting
            obj.gamma = 0.95;                                           % discount factor for history cost(reward)
            obj.epslon = 0;                                             % epslon-greedy explore parameter
        end
        
        % reset multilayer perceptron and clean history tuples
        function nfq = resetNetwork(nfq)
            nfq.cnn = CNN(nfq.nAction, [nfq.inputNode, nfq.hiddenNodes, nfq.outputNode], nfq.activationFunction);
            nfq.tupleList = [];
            nfq.tupleCounter = 0;
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

        % set actionList and number of actions in the list
        function nfq = setActionList(nfq, actionlist)
           nfq.actionList = actionlist;
           nfq.actionCount = size(actionlist, 1);
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
            nfq.tupleCounter = nfq.tupleCounter + 1;
        end
        
        % train the cnn network with all history tuples
        function nfq = train(nfq, epoches)
            if nargin == 2
               nfq.epoches = epoches;
            end
            %fprintf('number of tuples: %d\n', nfq.tupleCounter);
            % separate the data
            inputs = nfq.tupleList(:, 1:nfq.inputNode);
            nextState = nfq.tupleList(:, nfq.inputNode+1:end-2);
            reward = nfq.tupleList(:, end-1);
            status = nfq.tupleList(:, end);
            %N = (floor(nfq.tupleCounter/500)+1)*30;
            for i = 1:nfq.epoches   %N                                          % value update with one iteration; policy update with multiple iteration until converge
                Qpre = nfq.minQ(nextState);                                     % find Q of next state
                Qvalue = reward + nfq.gamma.*Qpre;                              % update Q of current state as reward + gamma*Q(nextState)
                Qvalue(status == -1) = reward(status == -1);                    % set Q of failure state as -1
                [goalState, goalValue] = nfq.hintToGoal(0);                     % add pre-knowledge points, not used yet
                nfq.cnn = nfq.cnn.fit([inputs; goalState],[Qvalue; goalValue]); % fit the [state, action] and Q
            end
            nfq.epslon = nfq.epslon*0.9;                                        % discount the epslon
        end

        % get the minimal J at state
        function Qpre = minQ(nfq, states)
            Qpre = zeros(size(states,1),1);
            for i = 1:size(states,1)
                [~, Qtmp] = nfq.bestAction(states(i,:));            % calculate Q of next state
                Qpre(i) = Qtmp;
            end    
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
        
        % get best action based on the critic network
        function [action, value] = bestAction(nfq, state)
            inputs = zeros(nfq.actionCount, nfq.inputNode);
            % construct all [state action] pairs
            for i = 1:nfq.actionCount
               inputs(i,:) = [state, nfq.actionList(i)]; 
            end
            [Qvalue, nfq.cnn] = nfq.cnn.getCost(inputs);            % get Q value of all state-action pair
            [value,index] = min(Qvalue);                            % get the action that minimize the Q value at the state
            action = nfq.actionList(index);  
        end
        
        % get random action in the action list
        function [action] = randAction(nfq)
            eps = rand(1);
            index = mod(round(eps*1000),nfq.actionCount)+1;
            action = nfq.actionList(index);  
        end   
        
        % add artificial tuples(where we know is good or bad) to help learning, not in use yet
        function [goalState, goalValue] = hintToGoal(nfq,num)
            state = zeros(1,nfq.nState);
            goal = zeros(nfq.actionCount, nfq.inputNode);
            for i = 1:nfq.actionCount
               goal(i,:) = [state, nfq.actionList(i)]; 
            end
            gvalue = zeros(nfq.actionCount, 1);
            cnt = num/nfq.actionCount;
            goalState = [];
            goalValue = [];
            for i = 1:cnt
                goalState = [goalState; goal];
                goalValue = [goalValue; gvalue];
            end
        end
        
    end
end