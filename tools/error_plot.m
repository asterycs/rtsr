clear all;
close all;

D = csvread('error.csv',1);

PC_IDX = 1;
LEVELS = 3;
lgdstr = cell(1,LEVELS+1);

for l=0:3
    hold on;
    R = D(D(:,2) == l & D(:,1) == PC_IDX,:);
    plot(R(:,4));
    grid on;
    lgdstr{l+1} = sprintf('Residual level %d',l);
end

lgd = legend(lgdstr);
lgd.FontSize = 14;
axis tight;

