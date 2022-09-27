%% mlp_rp - update multilayer perceptron with resilient propagation
%  
%  Copyright (c) 2015 Yue Wen
%  $Revision: 0.10 $
%
%  useage:
%       nn = rp_update(nn)
%
%  inputs:
%       nn - multilayer perceptrons created by mlp
%
%  outputs:
%       nn - multilayer percetron, update nn.W{i} nn.B{i} based on nn.GW{i}
%       nn.GWO{i} and nn.rpPara
%
function [nn] = rp_update(nn)
% update weight vector and bias vector layer by layer
for i= 1:nn.nbridge
    gg = nn.GW{i}.*nn.GWO{i};
    delta{i} = nn.rpPara.delta{i};
    delta{i} = min(delta{i}*nn.rpPara.mu_pos,nn.rpPara.dmax).*(gg>0) +...
    max(delta{i}*nn.rpPara.mu_neg,nn.rpPara.dmin).*(gg<0) +...
    delta{i}.*(gg==0);
    
    % only Rprop- is tested
    switch nn.rpPara.method
        case 'Rprop-'
            deltaW{i}       = -sign(nn.GW{i}).*delta{i};

        case 'Rprop+'
            deltaW{i}       = -sign(nn.GW{i}).*delta{i}.*(gg>=0) -...
                nn.GWO{i}.*(gg<0);
            nn.GW{i}        = nn.GW{i}.*(gg>=0);
            nn.GWO{i}   = deltaW{i};

        case 'IRprop-'
            nn.GW{i}    = nn.GW{i}.*(gg>=0);
            deltaW{i}   = -sign(nn.GW{i}).*delta{i};

        case 'IRprop+'
            deltaW{i}   = -sign(nn.GW{i}).*delta{i}.*(gg>=0) -...
                nn.GWO{i}.*(gg<0)*(E>old_E);
            nn.GW{i}    = nn.GW{i}.*(gg>=0);
            nn.GWO{i}   = deltaW{i};
            old_E       = E;

        otherwise
            error('Unknown method')

    end
    % update the W{i} and B{i}
    nn.GWO{i} = nn.GW{i};
    if nn.bias
        nn.W{i}   = nn.W{i} + deltaW{i}(1:end-1,:);
        nn.B{i}   = nn.B{i} + nn.bias*deltaW{i}(end,:);
    else
        nn.W{i}   = nn.W{i} + deltaW{i}(1:end,:);
    end
    nn.rpPara.delta{i} = delta{i};
end
        
end
