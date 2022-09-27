%% mlp_reset - random reset multilayer perceptron
%
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%  useage:
%       nn = mlp_reset(nn)
%
%  inputs:
%       nn - multilayer perceptrons created by mlp
%
%  outputs:
%       nn - multilayer percetron, reset nn.W{i} nn.B{i} nn.GW{i} nn.GWO{i}
%
function nn = mlp_reset(nn)
for i=1:nn.nbridge
    w_scale = 1/sqrt(nn.neurons(i)+nn.neurons(i+1));            % weight scale factor
    nn.W{i} = 2*w_scale*(rand([nn.neurons(i), nn.neurons(i+1)])-0.5);
    nn.B{i} = biasflag*(rand([1, nn.neurons(i+1)])-0.5);        % B are zeros when biasflag is zero.
    nn.GW{i} = zeros([nn.neurons(i)+1, nn.neurons(i+1)]);       % GW(gradient of weight) includes weight and bias as [nn.W{i}; nn.B{i}]
    nn.GWO{i} = zeros([nn.neurons(i)+1, nn.neurons(i+1)]);      % GWO(gradient of weight in the last iteration) includes weight and bias
end
end