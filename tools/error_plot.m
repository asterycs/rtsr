clear all;


error_files = { 'error_w_0_5_seq_cpu.csv' ...
                'error_w_1_0_seq_cpu.csv' ...
                'error_w_1_5_seq_cpu.csv' ...
                'error_w_0_5_par_cpu.csv' ...
                'error_w_1_0_par_cpu.csv' ...
                'error_w_1_5_par_cpu.csv'};
            
titles =      { 'Sequential CPU solver with w = 0.5' ...
                'Sequential CPU solver with w = 1.0' ...
                'Sequential CPU solver with w = 1.5' ...
                'Parallel CPU solver with w = 0.5' ...
                'Parallel CPU solver with w = 1.0' ...
                'Parallel CPU solver with w = 1.5'};
            
image_filenames = { 'res_seq_w_0_5' ...
                    'res_seq_w_1_0' ...
                    'res_seq_w_1_5' ...
                    'res_par_w_0_5' ...
                    'res_par_w_1_0' ...
                    'res_par_w_1_5'};
                
                
for f=1:length(error_files)
    color_handles = [];
    D = csvread(error_files{f},1);

    PC_IDX = 0;
    LEVELS = 6;
    lgdstr = cell(1,LEVELS);

    fig = figure;
    for l=0:LEVELS-1
        hold on;
        R = D(D(:,2) == l & D(:,1) == PC_IDX,:);
        h = plot(R(:,4));
        set(h,'LineWidth',2)
        plot(R(:,4),'xk');
        
        color_handles = [color_handles h];
        
        lgdstr{l+1} = sprintf('Mesh level %d',l);
    end

    lgd = legend(color_handles,lgdstr);
    lgd.FontSize = 20;
    axis tight;
    grid on;
    xlabel('Iteration (1)','FontSize',20);
    ylabel('Residual value (1)','FontSize',20);
    %title(titles{f});
    saveas(fig, image_filenames{f},'epsc');
end
