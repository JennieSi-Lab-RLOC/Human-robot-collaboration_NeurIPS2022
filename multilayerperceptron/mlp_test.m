neurons=[3 5 4];
actfunction={'linear','tansig','tansig'};
nn = mlp(neurons,actfunction);
x = [1 2 3; 1 4 6];
d = [0.5 -0.3 0.4 -0.1; 0.6 -0.8 0.2 -0.4];
[nn, o] = mlp_forward(nn, x);
[nn, oo] = mlp_backward(nn, o*0.2);
[se, error, nn] = mlp_validate(nn, x, d);