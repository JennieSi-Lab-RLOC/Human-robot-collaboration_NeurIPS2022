close all
%load the data file first, it's in the same folder
addpath('D:\Dropbox (ASU)\Research\Dynamic Programming\MATLAB\TNNLS code\PaperADP(PC141)\ADP+Model') 


% %% RMSE
% for ploti = 1:4
%     phindex = ploti;
%     
%     
%     ann = madp.ann(phindex);
%     cnn = madp.cnn(phindex);
%     data = ann.Ivhist;
%     peakE(1:size(data,1),phindex) = data(:,1);
%     
%     rmse = rms(peakE');
%     
% 
% end
% 
% % load data first and then plot RMSE
% RMSE_array(:,5) = rmse(1:500)';

load('other\1.mat')
%% FOUR PHASE STATE
for ploti = 1:4
    phindex = ploti;
    ADP_plot_new;
    if phindex == 4
       legend([la,lb],'without KG','with KG','Location','best') 
    end
    print(['ICRA_Final2020_Phase_',num2str(phindex)], '-dpng', '-r300');
end