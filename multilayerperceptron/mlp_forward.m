%% mlp_forward - calculate through multilayer perceptron
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%   
%  useage:
%       [nn, outputs] = mlp_forward(nn, inputs, full)
%
%  inputs:
%       nn - multilayer perceptrons
%
%       inputs - a row vector with one sample each row
%           this could be used as batch mode
%
%       full - the indicator for output selection
%           default setting is 0 for just output nodes value. 
%           set as one for output all hidden nodes value 
%
%  outputs:
%       nn - multilayer percetron, update nn.FV{i}
%           input value for each layer
%
%       output - values of output nodes
%           one row for each sample in bach mode.
%
function [nn, outputs] = mlp_forward(nn, inputs, full)
if size(inputs,2) ~= nn.ninput
    fprintf('Input to the MLP does not match!\n');
    return;
end
if nargin == 2
   full = 0; 
end
if nn.bias
    nn = mlp_forward_bias(nn, inputs);
else
    nn = mlp_forward_NObias(nn, inputs);
end
% output mode
if full
    outputs = nn.FV;
else
    outputs = nn.FV{nn.nlayer};
end
end

%% Forward calculation for neural network without bias term
function nn = mlp_forward_NObias(nn, inputs)
% add bias for the first hidden layer
nn.FV{1} = inputs;
for i=1:nn.nbridge
%     if i == 1
%     % inputs of nodes in layer i+1
%         vt = (nn.FV{i}.^2)*nn.W{i};
%     else
%         vt = nn.FV{i}*nn.W{i};
%     end
	vt = nn.FV{i}*nn.W{i};
    % output of nodes in layer i+1 through activation function
    if strcmp(nn.activefunction{i+1}, 'tansig')
%         fv = (1 - exp(-vt))./(1 + exp(-vt));        % from Si
        
        % from Yue -- with weight normalization
%         if i == nn.nbridge
%             vtscale = 0.5*max(abs(vt));     % 0.5*min(abs(vt));
%         else
%             vtscale = 0.5*max(abs(vt));
%         end
%         if vtscale > 2
%             nn.W{i} = nn.W{i}./vtscale;
%             vt = nn.FV{i}*nn.W{i};
%         end
        fv = 2./(1 + exp(-0.5*vt)) - 1;
        
    elseif strcmp(nn.activefunction{i+1}, 'logsig')
        fv = 1./(1 + exp(-vt));
    elseif strcmp(nn.activefunction{i+1}, 'linear')
        fv = vt;
    elseif strcmp(nn.activefunction{i+1}, 'relu')
        fv = zeros(size(vt));
        fv(vt>=0) = vt(vt>=0);
       
    else
        fprintf('activefunction does not match!\n');
        fv = zeros(size(vt));  
    end  
    % replace NaN with -1
    % fv(isnan(fv)) = -1;
    % check for NaN number in output of nodes
    flag = sum(sum(isnan(fv)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    % add bias to the input for next layer
    nn.FV{i+1} = fv;
end
% delete the bias for output layer
% nn.FV{nn.nlayer} = nn.FV{nn.nlayer}(:,1:end-1);
end

%% Forward calculation for neural network with bias term
function nn = mlp_forward_bias(nn, inputs)
snum = size(inputs, 1);
% add bias for the first hidden layer
nn.FV{1} = [inputs, nn.bias*ones(snum,1)];
for i=1:nn.nbridge
    % inputs of nodes in layer i+1
%     if i == 1
%     % inputs of nodes in layer i+1
%         vt = (nn.FV{i}.^2)*nn.W{i};
%     else
%         vt = nn.FV{i}*nn.W{i};
%     end
    vt = nn.FV{i}*nn.W{i};
    % output of nodes in layer i+1 through activation function
    if strcmp(nn.activefunction{i+1}, 'tansig')
        fv = (1 - exp(-2*vt))./(1 + exp(-2*vt));        %nn.V{1} = (exp(vt) - exp(-vt))./(exp(vt) + exp(-vt));
    elseif strcmp(nn.activefunction{i+1}, 'logsig')
        fv = 1./(1 + exp(-vt));
    elseif strcmp(nn.activefunction{i+1}, 'linear')
        fv = vt;
    elseif strcmp(nn.activefunction{i+1}, 'relu')
        fv = zeros(size(vt));
        fv(vt>=0) = vt(vt>=0);
    else
        fprintf('activefunction does not match!\n');
        fv = zeros(size(vt));  
    end  
    % replace NaN with -1
    % fv(isnan(fv)) = -1;
    % check for NaN number in output of nodes
    flag = sum(sum(isnan(fv)),2);
    if flag > 0
        fprintf('Bad Data');
    end
    % add bias to the input for next layer
    nn.FV{i+1} = [fv, nn.bias*ones(snum,1)];       
end
% delete the bias for output layer
nn.FV{nn.nlayer} = nn.FV{nn.nlayer}(:,1:end-1);
end

%% test of sigmoid
% t=-3:0.01:3;
% y1 = (1-exp(-t))./(1+exp(-t));
% y2 = (exp(t)-exp(-t))./(exp(t)+exp(-t));
% y3 = (1-exp(-2*t))./(1+exp(-2*t));
% plot(t,y1,'-r');hold on; plot(t,y2,'-b')
