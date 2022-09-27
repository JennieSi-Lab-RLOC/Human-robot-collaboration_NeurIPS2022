%% mlp_rp - train multilayer perceptron with resilient propagation
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%  useage:
%       nn = mlp_rp(nn)
%
%  inputs:
%       nn - multilayer perceptrons created by mlp
%
%       inputs - a row vector with one sample each row
%           this could be used as batch mode
%
%       targets - a row vector with one sample each row
%           this could be used as batch mode
%
%  outputs:
%       nn - multilayer percetron, update nn.W{i} nn.B{i} for inputs and
%       targets mapping
%
%       errors - error between network output and target
%
function [nn, errors] = mlp_rp(nn, inputs, targets)
if size(inputs,2) ~= nn.ninput
    fprintf('Input to the MLP does not match!\n');
    return;
end
if size(targets,2) ~= nn.noutput
    fprintf('Target to the MLP does not match!\n');
    return;
end
snum = size(inputs,1);
% batch mode update for multilayer perceptron with resilient propagation
for ep=1:nn.rpPara.epoches
    [nn, outputs] = mlp_forward(nn, inputs);
    errors = outputs - targets;
    mse = 0.5*sum(diag((errors'*errors)))/snum;
    if mse < nn.rpPara.mse
        break
    end
    [nn, ~ ] = mlp_backward(nn,errors);
    nn = rp_update(nn);
end
nn.rpResult.epoches = ep;
nn.rpResult.errors = errors;
nn.rpResult.mse = mse;
end
