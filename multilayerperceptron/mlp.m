%% mlp - construct multilayer perceptron
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%  
%  usage:
%       nn = mlp(neurons, actfunction, biasflag)
%
%  inputs:
%       neurons - the structure of the multilayer perceptron
%           such as [3, 5 , 1];
%
%       actfunction - the activation function of each layer
%           such as {'linear','tansig','logsig'};
%
%       biasflag - the indicator for network to include or exculde bias term.
%           default setting is 1. set as zero for zero mean inputs.
%
%  outputs:
%       nn - multilayer percetron, with nn.W{i} nn.B{i} nn.GW{i} nn.GWO{i} ...
%
function nn = mlp(neurons, actfunction, biasflag,ANN)
if nargin == 2
    biasflag = 1;                   % bias is applied to the network by default
    ANN=0;
end

if nargin == 3
    ANN = 0;                   
    
end
nn.neurons = neurons;               % neurons in the network from input layer to output layer
nn.ninput = neurons(1);             % input nodes
nn.noutput = neurons(end);          % output nodes
nn.nlayer = size(nn.neurons, 2);    % layers of network
nn.nbridge = nn.nlayer - 1;         % weight connects in the network, number of weight vectors
nn.bias = biasflag;                 
% nn.seeds = round(1000*nn.nbridge*rand([1,nn.nbridge]));

if size(actfunction,2) == nn.nlayer
    nn.activefunction = actfunction;
elseif size(actfunction,2) == 1
    for i=1:nn.nlayer
       nn.activefunction{i} = actfunction;
    end
else
    fprintf('actfunction parameter does not match!');
    return;
end

% Initialize Weights to be justified
for i=1:nn.nbridge
%     rng(nn.seeds(i));                                              % specify the seeds for repeat study 
    w_scale = 1;%1/sqrt(nn.neurons(i)+nn.neurons(i+1));            % weight scale factor
    if biasflag
%         if ANN==1
%             nn.W{i} = 2*w_scale*(rand([nn.neurons(i)+1, nn.neurons(i+1)])-0.5); % make sure it's positive for cnn ruofan
%         else
%             nn.W{i} = 2*w_scale*(rand([nn.neurons(i)+1, nn.neurons(i+1)]));
%         end
    else
        if ANN==1
            nn.W{i} = 2*w_scale*(rand([nn.neurons(i), nn.neurons(i+1)])-0.5); % make sure it's positive for cnn ruofan
        else
            if i==1 
                nn.W{i} = 2*w_scale*(rand([nn.neurons(i), nn.neurons(i+1)])-0.5);
                
            else
                nn.W{i} = 2*w_scale*(rand([nn.neurons(i), nn.neurons(i+1)])-0.5);
            end
        end
    end
    if biasflag
        nn.GW{i} = zeros([nn.neurons(i)+1, nn.neurons(i+1)]);       % GW(gradient of weight) includes weight and bias as [nn.W{i}; nn.B{i}]
        nn.B{i} = rand([1, nn.neurons(i+1)])-0.5;        % B are zeros when biasflag is zero.
    else
        nn.GW{i} = zeros([nn.neurons(i), nn.neurons(i+1)]);       % GW(gradient of weight) includes weight and bias as [nn.W{i}; nn.B{i}]
        nn.B{i} = zeros([1, nn.neurons(i+1)])-0.5;        % B are zeros when biasflag is zero.
    end
    nn.GWO{i} = nn.GW{i};   %zeros([nn.neurons(i)+1, nn.neurons(i+1)]);      % GWO(gradient of weight in the last iteration) includes weight and bias
    nn.FV{i} = zeros(1, nn.neurons(i));
    nn.BV{i} = zeros(1, nn.neurons(i));
end
nn.FV{i+1} = zeros(1, nn.neurons(i+1));
nn.BV{i+1} = zeros(1, nn.neurons(i+1));
end