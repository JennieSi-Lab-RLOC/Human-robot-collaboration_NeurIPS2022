%% zero mean samples for multi layer perceptron input
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
function [outputs] = zeromean(inputs)
n_col = size(inputs, 2);
nor_data = zeros(size(inputs));
mean_data = mean(inputs,2);
for ii = 1:n_col
    nor_data(:, ii) = inputs(:, ii) - mean_data;
end
max_nor_data = max(abs(nor_data), [], 2);
max_nor_data(max_nor_data == 0) = 1;
for ii = 1:n_col
    nor_data(:, ii) = nor_data(:, ii)./(max_nor_data/0.8);
end
outputs = nor_data;
end
