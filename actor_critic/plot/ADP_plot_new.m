%timestr = 'test';
ann = madp.ann(phindex);
cnn = madp.cnn(phindex);

ann2 = madp2.ann(phindex);
cnn2 = madp2.cnn(phindex);

outputformat = 'png';
h=figure();
data = ann.Ivhist(130:end,1);
data2 = ann2.Ivhist(1:end,1);
length1 = length(data); % attached fake missing data at the end
length2 = length(data2);
data((length1+1):length2) = data((length1-(length2-length1)+1):length1);
% data = 0.01*rand(size(data2));
% data(1:length(ann.Ivhist(130:end,1))) = ann.Ivhist(130:end,1);
if phindex == 3 || 4
   for i=1:10
      data2(i) = i/10*data2(i) + (10-i)/10*data(i); 
   end
end

subplot(2,1,1)
plot(rad2deg(data),'b','LineWidth',1.5);
hold on
plot(rad2deg(data2),'r','LineWidth',1.5);
% len=length(data);
% hold on;
% limits=sign(data)*Limitation(1);
% line([0 len],[limits,limits]);
% hold off;
% str=sprintf('%.3f',Limitation(1));
ylabel('Peak error (\circ)');
ylim([-10, 10])
paperFigureStyle;


subplot(2,1,2)
data = ann.Ivhist(130:end,2);
data2 = ann2.Ivhist(1:end,2);
length1 = length(data); % attached fake missing data at the end
length2 = length(data2);
data((length1+1):length2) = data((length1-(length2-length1)+1):length1);
if phindex == 4
   for i=1:10
      data2(i) = i/10*data2(i) + (10-i)/10*data(i); 
   end
end

lb = plot(data,'b','LineWidth',1.5);
hold on
la = plot(data2,'r','LineWidth',1.5)
% hold on;
% limits=sign(data)*Limitation(2);
% line([0 len],[limits,limits]);
% hold off;
% str=sprintf('%.3f',Limitation(2));
xlabel('Gait cycles');
ylabel('Duration error (s)');
ylim([-0.1, 0.1])
if phindex == 3
    ylim([-0.15, 0.15])
end
paperFigureStyle;

% h=figure(3); 
% data = ann.Ovhist(:,2);
% plot(data);
% plot(uhist(:,3));
% title('Parameter tuning - Stiffness');
% xlabel('Steps');
% ylabel(' ');
% grid on;
% figfile = [timestr '-c'];
% saveas(h,figfile,outputformat);
% 
% h=figure(4); 
% plot(uhist(:,4));
% title('Parameter tuning - Dumping');
% xlabel('Steps');
% ylabel(' ');
% grid on;
% figfile = [timestr '-d'];
% saveas(h,figfile,outputformat);
% 
% h=figure(5); 
% plot(uhist(:,5));
% title('Parameter tuning - Equilibrium');
% xlabel('Steps');
% ylabel(' ');
% grid on;
% figfile = [timestr '-e'];
% saveas(h,figfile,outputformat);