%% shuffle samples for multi layer perceptron input
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
function [inputs, targets] = shuffle(inputs, targets)
ni = size(inputs,1);
data=[inputs;targets];
[n_row, n_col] = size(data);

shuffle_seq = randperm(n_col);
for i = (1:n_col),
    data_shuffled(:,i) = data(:,shuffle_seq(i));
end
inputs = data_shuffled(1:ni,:);
targets = data_shuffled(ni+1:end,:);
end
