%% Action Neural Network
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%
classdef ANN
    properties
        nn;                 % multilayer perceptron
        nstate;             % number of state
        naction;            % number of action
        output;             % output value of the ANN   
        
        knowledgeGuidedMode;
    end
    
    methods
        function obj = ANN(neurons, actfunction, bias, knowledgeGuidedMode)
            if nargin == 2
                bias = 1;
            end
            %% create multilayer perceptron
            obj.nn = mlp(neurons,actfunction,bias,1);                  
            obj.naction = neurons(end);  
            obj.nstate = neurons(1);
            
            %% initialize the resilient propagation parameters
            obj.nn.rpPara.iterations = 600;                 % Maximum number of iterations
            obj.nn.rpPara.epoches = 1200;                   % Maximum number of epoches
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
            
            %% initialize the back propagation parameters 
            obj.nn.bpPara.epoches = 100;                    % Maximum number of epoches(iterations)
            obj.nn.bpPara.lr = 0.1;                         % Learning rate
            obj.nn.bpPara.target = 0.01;                    % Target MSE
            obj.nn.bpPara.lro = 0.3;
            obj.nn.momentum = 0;
%             obj.nn.W{1} = 0.1*obj.nn.W{1};
%             obj.nn.W{2} = 0.1*obj.nn.W{2};
            
            obj.knowledgeGuidedMode = knowledgeGuidedMode;
        end       
        
        %% back propagation method (dHDP)
        % annealing of the learning rate for back propagation
        function ann = learnRateDecay(ann)
            if (ann.nn.bpPara.lr > 0.01)
                ann.nn.bpPara.lr = ann.nn.bpPara.lr - 0.05;  % 0.05(si),0.005(yue)?
            else
                ann.nn.bpPara.lr = 0.005;
            end
        end
        
        % reset learning rate for the back propagation
        function ann = learnRateReset(ann)
            ann.nn.bpPara.lr = ann.nn.bpPara.lro;
        end
  
        % action network take state as parameters and generate action
        function [action, ann] = getAction(ann, state)
            [ann.nn, action] = mlp_forward(ann.nn,state);
            ann.output = action;
        end
        
        % backpropagate through critic network
        function [ann, J]= bpUpdate(ann, cnn, states)
%             disp('BP updating ANN');
            for ep = 1:ann.nn.bpPara.epoches
                [ann.nn, actions] = mlp_forward(ann.nn,states);       
                [cnn, action_grad, J]=cnn.bp2ann([states, actions]);
                mse = 0.5*sum(J'*J)/size(J,1);
                if mse < ann.nn.bpPara.target
                    break
                end
                [ann.nn, ~] = mlp_backward(ann.nn,action_grad,1);
%                 if mod(ep,10) == 0
%                    pause(0.02);
%                 end
            end
            ann.nn.bpResult.epoches = ep;
            ann.nn.bpResult.mse = mse;
%             if mse >= ann.nn.bpPara.target
%                 fprintf('ann bpUpdate with %d epoches: mse %f\n', ep, mse);
%             end
        end
        
                % backpropagate through critic network
        function [ann, J]= bpUpdateQLearning(ann, cnn, states, target,exADP, ratio)
%             disp('BP updating Q-learning ANN');
            for ep = 1:ann.nn.bpPara.epoches
                [ann.nn, actions] = mlp_forward(ann.nn,states);       
                [cnn, action_grad, J]=cnn.bp2ann([states, actions]);
                
                if ann.knowledgeGuidedMode   
                    [~, exaction_grad, exValue]=exADP.cnn.bp2ann([states, actions]);
                    J = (1-ratio)*J + ratio*exValue;
                    action_grad = action_grad + ratio*exaction_grad;
                end
                feedback = J-target;
                mse = 0.5*sum(feedback'*feedback)/size(J,1);
                if mse < ann.nn.bpPara.target
                    break
                end
                [ann.nn, ~] = mlp_backward(ann.nn,action_grad,1);
%                 if mod(ep,10) == 0
%                    pause(0.02);
%                 end
            end
            ann.nn.bpResult.epoches = ep;
            ann.nn.bpResult.mse = mse;
            if mse >= ann.nn.bpPara.target
                fprintf('ann bpUpdate with %d epoches: mse %f\n', ep, mse);
            end
        end
        
        % external knowledge-guided Q' and action gradient
        function [action_grad, Qp] = getActionGradient(ann, state, action)
            ratio = -0.5;  % a small ratio
            global NF;
            global angVelocityModel;
            rawState = state.*NF;
            Angle = rawState(1);  % in rad
            AngVelocity = rawState(2);
            Force = action*10; % continuous case
            
            % external value Q'
            regTable = table(Angle, AngVelocity, Force); % regression table
            yfit = angVelocityModel.predictFcn(regTable);
            Qp = ratio*yfit; 
            
            % numerical derivative
            h = 0.1;       % step size
            Force = Force+h;    % domain            
            regTable = table(Angle, AngVelocity, Force); % regression table
            yfit2 = angVelocityModel.predictFcn(regTable);
            force_grad = (yfit2-yfit)/h;   % first derivative
            action_grad = ratio * force_grad/10;
        end
        
        %% resilient propagation method (dHDP, NFQCA)
        % resilient propagation training method through critic network
        function [ann, mse]= rpUpdate(ann, cnn, states)
            for ep=1:ann.nn.rpPara.iterations
                [ann.nn, actions] = mlp_forward(ann.nn,states);
                [cnn, action_grad, J] = cnn.bp2ann([states, actions]);              
                mse = 0.5*sum(J'*J)/size(J,1);
                if mse < 0.01  %ann.nn.rpPara.mse
                    break
                end
                [ann.nn, ~] = mlp_backward(ann.nn,action_grad);
                ann.nn = rp_update(ann.nn);
            end
            ann.nn.rpResult.epoches = ep;
            ann.nn.rpResult.mse = mse;
            ann.nn.rpResult.error = J;
            if mse >= 0.01     %ann.nn.rpPara.mse
                fprintf('ann rpUpdate with %d epoches: mse %f\n', ep, mse);
            end
        end
        
        function ann = cleanann(ann)      
            ann.nn.bpPara.lr = ann.nn.bpPara.lro;
            for i = 1: ann.nn.nbridge                       % Initial step size matrix for weight
                ann.nn.rpPara.delta{i} = ann.nn.rpPara.delta0*ones(size([ann.nn.W{i};ann.nn.B{i}]));
                ann.nn.GW{i} = zeros([ann.nn.neurons(i)+1, ann.nn.neurons(i+1)]);
                ann.nn.GWO{i} = zeros([ann.nn.neurons(i)+1, ann.nn.neurons(i+1)]); 
            end
        end
        function ann=mlp_reset(ann,neurons, actfunction, bias, knowledgeGuidedMode)
            ann.nn = mlp(neurons,actfunction,bias,1);  
            %% initialize the resilient propagation parameters
            ann.nn.rpPara.iterations = 600;                 % Maximum number of iterations
            ann.nn.rpPara.epoches = 1200;                   % Maximum number of epoches
            ann.nn.rpPara.mse = 10^-4;                      % Target MSE
            ann.nn.rpPara.mu_pos = 1.2;                     % Increase factor mu
            ann.nn.rpPara.mu_neg = 0.5;                     % Decrease factor mu
            ann.nn.rpPara.dmax = 50;                        % Upper bound of step size
            ann.nn.rpPara.dmin = 0.000001;                  % Lower bound of step size
            ann.nn.rpPara.delta0 = 0.07;                    % Initial step size
            ann.nn.rpPara.method = 'Rprop-';                % RProp method used, including 'Rprop-' 'Rprop+' 'IRprop-' 'IRprop+'
            for i = 1: ann.nn.nbridge                       % Initial step size matrix for weight
                ann.nn.rpPara.delta{i} = ann.nn.rpPara.delta0*ones(size([ann.nn.W{i};ann.nn.B{i}]));
            end
            
            %% initialize the back propagation parameters 
            ann.nn.bpPara.epoches = 100;                    % Maximum number of epoches(iterations)
            ann.nn.bpPara.lr = 0.1;                         % Learning rate
            ann.nn.bpPara.target = 0.01;                    % Target MSE
            ann.nn.bpPara.lro = 0.3;

%             ann.nn.W{1} = 0.1*ann.nn.W{1};
%             ann.nn.W{2} = 0.1*ann.nn.W{2};
            
            ann.knowledgeGuidedMode = knowledgeGuidedMode;
        end
    end
end