%% mlp_validate multi layer perceptron in batch
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $

function [se, errors, nn] = mlp_validate(nn, inputs, targets)
snum = size(inputs,1);
[nn, outputs] = mlp_forward(nn, inputs);
error = outputs - targets;
errors = error;
se = 0.5*sum(diag((error'*error)))/snum;
end