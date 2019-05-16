%% show success rate figure
clear;clc;


joint_num = 21;
suc_rate = 0.00:0.001:0.08;
suc_rate_thred = 1000*suc_rate;

color = [
204,236,230
153,216,201
102,194,164
44,162,95
0,109,44
254,217,118
254,178,76
253,141,60
240,59,32
189,0,38
    ]/255.0;

color = color - 0.2;
color(color<0) = 0;


% Load the results
modes =   {'1_1';                   
           };

for f = 1 : length(modes)
    mode = char(modes(f));
    load(['results_v2\Error_Seen_UnSeen_Ego_', mode,'.mat']);
%     load(['results_v2\Error_Seen_UnSeen_3rd_', mode,'.mat']);
    numberSeen = size(SeenError,1);
    numberUnseen = size(UnseenError,1);
%     for e = 1 : length(suc_rate_thred)
%         CurveError_Seen(e, f) = sum(sum(SeenError<suc_rate_thred(e)))/numberSeen/21;
%         CurveError_Unseen(e, f) = sum(sum(UnseenError<suc_rate_thred(e)))/numberUnseen/21;
%     end 
    for e = 1 : length(suc_rate_thred)
        CurveError_Seen(e) = sum(max(SeenError,[],2)<suc_rate_thred(e))/numberSeen;
        CurveError_Unseen(e) = sum(max(UnseenError,[],2)<suc_rate_thred(e))/numberUnseen;
    end 
end




hFig = figure(1);
set(hFig, 'Position', [0 0 400 300]);

% set(gcf, 'position', [300 300 420 280]);
set(gcf, 'position', [300 300 420*1.7 280*1.9]);
set(gca,'Position',[0.25 0.25 0.45 0.45]);

set(gca,'YTick', 0:10:100);
set(gca,'XTick', 0:10:80);
grid on;
box on;
set(gca,'gridlinestyle',':');
% set(gca,'XGrid','off');
xlabel('error threshold  \epsilon (mm)', 'fontsize', 16);
ylabel('proportion of frames with error < \epsilon', 'fontsize', 14);
set(gca,'yticklabel', 0:10:100);
labels=get(gca,'yticklabel');
for i=1:size(labels,1)
   labels_(i,:)=[labels(i,:) '%'];
end
set(gca,'yticklabel',labels_);
hold on;

plot(suc_rate*1000, 100*CurveError_Seen, '-', 'Color', color(5,:), 'LineWidth', 2); hold on;
plot(suc_rate*1000, 100*CurveError_Unseen, '-', 'Color', color(9,:), 'LineWidth', 2); hold on;

h_legend=legend('Seen','Unseen',...
                'Location','southeast');
% print('results_v2/Challenge_V2_Ego_All_Mean_vs_Portion_vs_1_1','-depsc','-tiff')            
% print('results_v2/Challenge_V2_3rd_All_Mean_vs_Portion_vs_1_1','-depsc','-tiff')            
% print('results_v2/Challenge_V2_3rd_All_Mean_vs_Portion_vs_1_1_max','-depsc','-tiff')  
print('results_v2/Challenge_V2_Ego_All_Mean_vs_Portion_vs_1_1_max','-depsc','-tiff')  
