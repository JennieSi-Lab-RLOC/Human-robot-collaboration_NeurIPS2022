%% mlp_backward - backpropagate through multilayer perceptron
%   
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%  useage:
%       [nn, output] = mlp_backward(nn,feedback,method)
%
%  inputs:
%       nn - multilayer perceptrons
%
%       feedback - the error between the output and the target
%           feedback = output - target; minimize E = 1/2*feedback^2 
%
%       method - the indicator for using backpropagation method
%           default setting is 0 for resilient propagation. set as one for backpropagation. 
%
%  outputs:
%       nn - multilayer percetron, update nn.BV{i} nn.GW{i} 
%           gradient of nodes and weight
%
%       output - gradient of the input nodes
%
function [nn, output] = mlp_backward(nn, feedback, method)
if nargin == 2
   method = 0;          % calculate the gradient for resilient propagation
end
if size(feedback,2) ~= nn.noutput
    fprintf('Feedback to the MLP does not match!\n');
end

if nn.bias
    nn = mlp_backward_bias(nn, feedback);
    if method == 1                  % method equals to one for updating the weight directly with back propagation method.
        for j=1:nn.nbridge
            nn.W{j} = nn.W{j} - nn.bpPara.lr*nn.GW{j}(1:end-1,:);
            nn.B{j} = nn.B{j} - nn.bpPara.lr*nn.GW{j}(end,:);
        end
    end
else
    nn = mlp_backward_NObias(nn, feedback);
    if method == 1                  % method equals to one for updating the weight directly with back propagation method.
        for j=1:nn.nbridge
            
            nn.W{j} = nn.W{j} - nn.bpPara.lr*(nn.momentum*nn.GWO{j}+(1-nn.momentum)*nn.GW{j});
            nn.GWO{j}=(nn.momentum*nn.GWO{j}+(1-nn.momentum)*nn.GW{j});
        end
        
    end
end
output = nn.BV{1};
end

%% Backward calculation for neural network without bias term
function [nn, output] = mlp_backward_NObias(nn, feedback)
nn.BV{nn.nlayer} = feedback;        % gradient for the output layer
% calculate the gradient of each layer
for i = nn.nlayer:-1:2
    vt = nn.BV{i};
    % back propagate through activation function
    if strcmp(nn.activefunction{i}, 'tansig')
        delta = (1-nn.FV{i}).*(1+nn.FV{i}).*vt;         %    nn.V{1} = (exp(vt) - exp(-vt))./(exp(vt) + exp(-vt));
    elseif strcmp(nn.activefunction{i}, 'logsig')
        delta = 1*(nn.FV{i}.*(1-nn.FV{i})).*vt;           %    nn.V{1} = (exp(vt) - exp(-vt))./(exp(vt) + exp(-vt));        
    elseif strcmp(nn.activefunction{i}, 'linear')
        delta = vt;
    elseif strcmp(nn.activefunction{i}, 'relu')
        delta=zeros(size(vt));
        delta(nn.FV{i}>=0) = vt(nn.FV{i}>=0);
    else
        fprintf('activefunction does not match!\n');
        output = zeros([nn.ninput,1]);
        return
    end   
    
    %delta(isnan(delta)) = -1;
    flag = sum(sum(isnan(delta)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    
    % gradient of nodes outputs
%     nn.BV{i-1} = nn.BV{i}*[nn.W{i-1}]';
    nn.BV{i-1} = delta*[nn.W{i-1}]';
    % gradient of weight vector and bias vector
%     dw = nn.FV{i-1}'*nn.BV{i};
%     if i ==2
%         dw= (nn.FV{i-1}.^2)'*delta;
%     else
%         dw= nn.FV{i-1}'*delta; % Ruofan modified
%     end
    dw= nn.FV{i-1}'*delta;
    % check for NaN number in gradient of weight
    flag = sum(sum(isnan(dw)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    nn.GW{i-1} = dw;        %Gradiant of both weight and bias
end
% gradient of the input node
% nn.BV{1} = 2*(nn.BV{2}*nn.W{1}').*nn.FV{1};
nn.BV{1} = nn.BV{2}*nn.W{1}';
end

%% Backward calculation for neural network with bias term
function [nn, output] = mlp_backward_bias(nn, feedback)
nn.BV{nn.nlayer} = feedback;        % gradient for the output layer
% calculate the gradient of each layer
for i = nn.nlayer:-1:2
    vt = nn.BV{i};
    % back propagate through activation function
    if strcmp(nn.activefunction{i}, 'tansig')
        delta = (1-nn.FV{i}).*(1+nn.FV{i}).*vt;         %    nn.V{1} = (exp(vt) - exp(-vt))./(exp(vt) + exp(-vt));
    elseif strcmp(nn.activefunction{i}, 'logsig')
        delta = (nn.FV{i}.*(1-nn.FV{i})).*vt;           %    nn.V{1} = (exp(vt) - exp(-vt))./(exp(vt) + exp(-vt));        
    elseif strcmp(nn.activefunction{i}, 'linear')
        delta = vt;
    elseif strcmp(nn.activefunction{i}, 'relu')
        delta=zeros(size(vt));
        delta(vt>=0) = vt(vt>=0);
    else
        fprintf('activefunction does not match!\n');
        output = zeros([nn.ninput,1]);
        return
    end   
    
    %delta(isnan(delta)) = -1;
    flag = sum(sum(isnan(delta)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    
    % BV is the gradient of the outputs of each nodes
    % exclude bias for each layer except for output layer
    if i == nn.nlayer
        nn.BV{i} = delta;
    else
        nn.BV{i} = delta(:,1:end-1);
    end
    % gradient of nodes outputs
    nn.BV{i-1} = nn.BV{i}*[nn.W{i-1};nn.B{i-1}]';
    % gradient of weight vector and bias vector
    dw = nn.FV{i-1}'*nn.BV{i};
    
    % check for NaN number in gradient of weight   
    flag = sum(sum(isnan(dw)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    nn.GW{i-1} = dw;        %Gradiant of both weight and bias
end
% gradient of the input node
nn.BV{1} = 2*(nn.BV{2}*nn.W{1}').*nn.FV{1};
end
