%% Critic Neural Network
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%
classdef CNN
    properties
        nn;                 % multilayer perceptron
        nstate;             % number of state
        naction;            % number of action
        ninput;             % number of nodes in the input layer
        noutput;            % number of nodes in the output layer
        output;             % output value of the CNN
        knowledgeGuidedMode;
    end
    
    methods
        function obj = CNN(naction, neurons, actfunction, bias)
            if nargin == 3
                bias = 1;                                   % set bias for multilayer perceptron
            end
            obj.nn = mlp(neurons,actfunction,bias);
            obj.ninput = neurons(1);
            obj.naction = naction;
            obj.nstate = neurons(1)-naction;
            obj.noutput = neurons(end);
            
            % set parameters for resilient propagation
            obj.nn.rpPara.iterations = 100;                 % Maximum number of iterations
            obj.nn.rpPara.epoches = 100;                   % Maximum number of epoches
            obj.nn.rpPara.mse = 10^-4;                      % Target MSE
            obj.nn.rpPara.mu_pos = 1.2;                     % Increase factor mu
            obj.nn.rpPara.mu_neg = 0.5;                     % Decrease factor mu
            obj.nn.rpPara.dmax = 50;                        % Upper bound of step size
            obj.nn.rpPara.dmin = 0.000001;                  % Lower bound of step size
            obj.nn.rpPara.delta0 = 0.07;                    % Initial step size
            obj.nn.rpPara.method = 'Rprop-';                % RProp method used, including 'Rprop-' 'Rprop+' 'IRprop-' 'IRprop+'
            for i = 1: obj.nn.nbridge                       % Initial step size matrix for weight
                obj.nn.rpPara.delta{i} = obj.nn.rpPara.delta0*ones(size([obj.nn.W{i};obj.nn.B{i}]));
            end
            
            % set parameters for the back propagation
            obj.nn.bpPara.epoches = 100;                    % Maximum number of epoches(iterations)
            obj.nn.bpPara.lr = 0.1;                         % Learning rate
            obj.nn.bpPara.target = 0.001;                    % Target MSE
            obj.nn.bpPara.lro = 0.2;                        % Learning rate backup
            obj.nn.momentum = 0;
            obj.nn.W{1} = 0.1*obj.nn.W{1};
            obj.nn.W{2} = 0.1*obj.nn.W{2};
        end       
        
        %% back propagation method (dHDP)
        % annealing of the learning rate for back propagation
        function cnn = learnRateDecay(cnn)
            if (cnn.nn.bpPara.lr >0.01)
                cnn.nn.bpPara.lr = cnn.nn.bpPara.lr - 0.05; % 0.05(si),0.005(yue)
            else
                cnn.nn.bpPara.lr = 0.005;
            end       
        end
        
        % reset learning rate for the back propagation
        function cnn = learnRateReset(cnn)
            cnn.nn.bpPara.lr = cnn.nn.bpPara.lro;
        end
        
        % back propagation training method
        function cnn = bpUpdate(cnn, input, target, gamma)
            for ep=1:cnn.nn.bpPara.epoches
                [cnn.nn, J] = mlp_forward(cnn.nn,input);                % get the J value
                feedback = gamma*J - target;                            % comput the error between J and (Jpre - reinforcementr)
                mse = 0.5*sum(feedback'*feedback)/size(feedback,1);
                if mse < cnn.nn.bpPara.target
                    break
                end
                [cnn.nn, ~] = mlp_backward(cnn.nn, feedback, 1);
%                 if mod(ep,10) == 0
%                    pause(0.02);
%                 end
            % normalize the weights if necessary -- not working?
%                 if (max(max(abs(cnn.nn.W{1})))>1.5)
%                     cnn.nn.W{1}=cnn.nn.W{1}/max(max(abs(cnn.nn.W{1})));
%                 end
%                 if max(max(abs(cnn.nn.W{2})))>1.5
%                     cnn.nn.W{2}=cnn.nn.W{2}/max(max(abs(cnn.nn.W{2})));
%                 end                
            end
            cnn.nn.bpResult.epoches = ep;
            cnn.nn.bpResult.mse = mse;
            cnn.nn.bpResult.error = feedback;
            if mse >= cnn.nn.bpPara.target
                fprintf('cnn bpUpdate with %d epoches: mse %f\n', ep, mse);
            end
        end 
        
                % back propagation training method for Q learning
        function cnn = bpUpdateQLearning(cnn, input, target)
%             disp('BP updating Q-learning CNN');
            for ep=1:cnn.nn.bpPara.epoches
                [cnn.nn, Q] = mlp_forward(cnn.nn,input);                % get the Q value
                feedback = Q - target;                            % comput the error between current Q and the target Q
                mse = 0.5*sum(feedback'*feedback)/size(feedback,1);
                if mse < cnn.nn.bpPara.target
                    break
                end
                [cnn.nn, ~] = mlp_backward(cnn.nn, feedback, 1);

            % normalize the weights if necessary -- not working?
%                 if (max(max(abs(cnn.nn.W{1})))>1.5)
%                     cnn.nn.W{1}=cnn.nn.W{1}/max(max(abs(cnn.nn.W{1})));
%                 end
%                 if max(max(abs(cnn.nn.W{2})))>1.5
%                     cnn.nn.W{2}=cnn.nn.W{2}/max(max(abs(cnn.nn.W{2})));
%                 end                
            end
            cnn.nn.bpResult.epoches = ep;
            cnn.nn.bpResult.mse = mse;
            cnn.nn.bpResult.error = feedback;
            if mse >= cnn.nn.bpPara.target
                fprintf('cnn bpUpdate with %d epoches: mse %f\n', ep, mse);
            end
        end 
        
        %% resilient propagation method (dHDP)
        % resilient propagation training method
        function cnn = rpUpdate(cnn, input, target, gamma)
            for ep=1:cnn.nn.rpPara.iterations
                [cnn.nn, J] = mlp_forward(cnn.nn,input);  
                feedback = gamma*J - target;
                mse = 0.5*sum(feedback'*feedback)/size(feedback,1);
                if mse < 0.01   %cnn.nn.rpPara.mse
                    break
                end
                [cnn.nn, ~] = mlp_backward(cnn.nn,feedback);
                cnn.nn = rp_update(cnn.nn);
            end
            cnn.nn.rpResult.epoches = ep;
            cnn.nn.rpResult.mse = mse;
            cnn.nn.rpResult.error = feedback;
            if mse >= 0.01  % cnn.nn.rpPara.mse
                fprintf('cnn rpUpdate with %d epoches: mse %f\n', ep, mse);
            end
        end
        
        %% back propagate the error to action (dHDP and NFQCA)
        function [cnn, action_grad, J]= bp2ann(cnn, input)
            [cnn.nn, J] = mlp_forward(cnn.nn,input);
            [cnn.nn, grad] = mlp_backward(cnn.nn,J);%ruofan modify: don't update the network
            action_grad = grad(:,cnn.nstate+1:end);
        end
        
        
        %% fit the inputs and targets with resilient propagation (NFQ and NFQCA)
        function cnn = fit(cnn, inputs, targets)
            [cnn.nn, error] = mlp_rp(cnn.nn, inputs, targets);  
            if cnn.nn.rpResult.mse >= cnn.nn.rpPara.mse
                fprintf('cnn fitting with %d epoches: mse %f\n', cnn.nn.rpResult.epoches, cnn.nn.rpResult.mse);
            end
        end
        
        function [outputs, cnn] = getCost(cnn, inputs)
            [cnn.nn, outputs] = mlp_forward(cnn.nn,inputs);
            cnn.output = outputs;
        end
        
        function cnn = cleancnn(cnn)           
            cnn.nn.bpPara.lr = cnn.nn.bpPara.lro;
            for i = 1: cnn.nn.nbridge                       % Initial step size matrix for weight
                cnn.nn.rpPara.delta{i} = cnn.nn.rpPara.delta0*ones(size([cnn.nn.W{i};cnn.nn.B{i}]));
                cnn.nn.GW{i} = zeros([cnn.nn.neurons(i)+1, cnn.nn.neurons(i+1)]);
                cnn.nn.GWO{i} = zeros([cnn.nn.neurons(i)+1, cnn.nn.neurons(i+1)]); 
            end
        end
    end
end