close all

savePrintFigure = 1;

%% after tuning plot
array = [88,89,93];
% array = [88,89,92,93,94,96,99];

figure
plot(-targetProfile/pi*180, 'r','LineWidth',2);
hold on
for i = 1:size(array,2);
    fileName = strcat(num2str(array(i)),'.mat');
    load(fileName);
    plot(-md.profile/pi*180, 'Color',[0,0.7,0.9]);
    hold on
end

load('profileHist.mat')
plot(-profileHist/pi*180, 'Color',[0,0.7,0.9]);
plot(-targetProfile/pi*180, 'r','LineWidth',2);

xlabel('Gait Cycle (%)', 'fontsize',12);
ylabel('Knee Profile (Degree)', 'fontsize',12);

x1 = 2;
y1 = 65;
txt1 = 'Phase 1';
text(x1,y1,txt1)
x1 = 75;
txt1 = 'Phase 2';
text(x1,y1,txt1)
x1 = 160;
txt1 = 'Phase 3';
text(x1,y1,txt1)
x1 = 240;
txt1 = 'Phase 4';
text(x1,y1,txt1)
hleg1 = legend('Target Profile','Trial')
set(hleg1,'Location','Southeast');

% phase transition lines
x = [38 140 210];
for idx = 1 : numel(x)
    plot([x(idx) x(idx)], [-10 70], 'k');
end

ylabels = {[], 0, [], 20, [], 40, [], 60, []};
set(gca,'YTickLabel',ylabels)
ylim([-10 70])

xlabels = {0, [], [], [], [], 100};
set(gca,'XTick',0:60:300,'XTickLabel',xlabels)
xlim([0 300])

set(gca,'FontSize',12)
title('ADP Tuned Knee Profile and Target')

if savePrintFigure
    print('after_2017','-dpng','-r300');
end

%% initial plot
array = [88,89,92,93,94,96,99,81,82,83,84,85];

figure
load('InitImpedanceSet.mat');
plot(-targetProfile/pi*180, 'r','LineWidth',2);
hold on
for i = 1:size(array,2);
    plot(-InitialMotion(:,array(i))/pi*180, 'Color',[0,0.7,0.9]);
    hold on
end
plot(-targetProfile/pi*180, 'r','LineWidth',2);

% phase transition lines
x = [38 140 210];
for idx = 1 : numel(x)
    plot([x(idx) x(idx)], [-10 70], 'k');
end

xlabel('Gait Cycle (%)', 'fontsize',12);
ylabel('Knee Profile (Degree)', 'fontsize',12);

x1 = 2;
y1 = 65;
txt1 = 'Phase 1';
text(x1,y1,txt1)
x1 = 75;
txt1 = 'Phase 2';
text(x1,y1,txt1)
x1 = 160;
txt1 = 'Phase 3';
text(x1,y1,txt1)
x1 = 240;
txt1 = 'Phase 4';
text(x1,y1,txt1)
hleg1 = legend('Target Profile','Trial')
set(hleg1,'Location','Southeast');

ylabels = {[], 0, [], 20, [], 40, [], 60, []};
set(gca,'YTickLabel',ylabels)
ylim([-10 70])

labels = {0, [], [], [], [], 100};
set(gca,'XTick',0:60:300,'XTickLabel',labels)
xlim([0 300])

set(gca,'FontSize',12)
title('Initial Knee Profile and Target')

if savePrintFigure
    print('before_2017','-dpng','-r300');
end

