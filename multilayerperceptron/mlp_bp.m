%% mlp_bp - train multilayer perceptron with backpropagation
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%  useage:
%       nn = mlp_bp(nn)
%
%  inputs:
%       nn - multilayer perceptrons created by mlp
%
%       input - a row vector with one sample each row
%           this could be used as batch mode
%
%       target - a row vector with one sample each row
%           this could be used as batch mode
%
%  outputs:
%       nn - multilayer percetron, update nn.W{i} nn.B{i} for inputs and
%       targets mapping
%
%       errors - error between network output and target
%
function [nn, errors] = mlp_bp(nn, inputs, targets)
if size(inputs,2) ~= nn.ninput
    fprintf('Input to the MLP does not match!\n');
end
if size(targets,2) ~= nn.noutput
    fprintf('Target to the MLP does not match!\n');
end
snum = size(inputs,1);
% batch mode update for multilayer perceptron with backpropagation
for ep=1:nn.bpPara.epoches
    [nn, outputs] = mlp_forward(nn, inputs);
    errors = outputs - targets;
    mse = 0.5*sum(diag((errors'*errors)))/snum;
    %mse = 0.5*sum(error'*error)/size(error,1);
    if mse < nn.bpPara.target
        break
    end
    [nn, ~ ] = mlp_backward(nn, errors, 1);
end
nn.bpResult.epoches = ep;
nn.bpResult.errors = errors;
nn.bpResult.mse = mse;
end
