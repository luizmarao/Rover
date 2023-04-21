
close all

% what happened
figure(1)
subplot(length(acker_experiment), 1, 1)
% error
figure(2)
subplot(length(acker_experiment), 1, 1)

acker_experiment_idx = [13 14];

steer_and_drive = true

for i = 1:length(acker_experiment_idx)
    
    h=1;
    h1=1;
    g=1;
    g1=1;
    
    %% what happened
    figure(1)
    
    subplot(length(acker_experiment_idx), 1, i)
    
    h(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(acker_experiment(acker_experiment_idx(i)).ghost_steer_hinge));
    hold on

    h(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(acker_experiment(acker_experiment_idx(i)).f_l_steer_hinge),'r','LineWidth', 2);
    h(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(polyval(pleft, acker_experiment(acker_experiment_idx(i)).ghost_steer_hinge)),'r');

    h(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(acker_experiment(acker_experiment_idx(i)).f_r_steer_hinge),'g','LineWidth', 2);
    h(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(polyval(pright, acker_experiment(acker_experiment_idx(i)).ghost_steer_hinge)),'g');
    
    if steer_and_drive
    
        for j = 1:length(acker_experiment(acker_experiment_idx(i)).about.tested_steer)
            plot([  (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period, ...
                    (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period   ],...
                [-50 50],'Color', [0.1 0.1 0.1], 'LineStyle', ':')

            yyaxis right
            h1(end+1)=plot([ (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period, ...
                            (j)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period   ],...
                            [ rad2deg(acker_experiment(acker_experiment_idx(i)).about.tested_steer(j)), ...
                            rad2deg(acker_experiment(acker_experiment_idx(i)).about.tested_steer(j))], '-ob');
            h1(end+1)=plot([ (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period, ...
                            (j)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period   ],...
                            [ acker_experiment(acker_experiment_idx(i)).about.tested_drive(j), ...
                            acker_experiment(acker_experiment_idx(i)).about.tested_drive(j)], '-or');
            ylim([-20 20])
        end

        yyaxis left
    end
    
    legend(h(2:end),'ghost', 'left', 'pleft', 'right', 'pright')
    xlabel('time in seconds')
    ylabel('angle in degrees')
    title(sprintf('comparison between what should be and what it is | imp [ %7.4f %7.4f %7.4f ] | ref [ %7.4f %7.4f ]', acker_experiment(acker_experiment_idx(i)).about.solimp, acker_experiment(acker_experiment_idx(i)).about.solref))
    grid

    %% error
    figure(2)
    
    subplot(length(acker_experiment_idx), 1, i)

    g(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(acker_experiment(acker_experiment_idx(i)).error_left),'r');
    hold on
    g(end+1)=plot(acker_experiment(acker_experiment_idx(i)).time, rad2deg(acker_experiment(acker_experiment_idx(i)).error_right),'g');

    if steer_and_drive
    
        for j = 1:length(acker_experiment(acker_experiment_idx(i)).about.tested_steer)
            plot([(j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period],...
                [-50 50],'Color', [0.1 0.1 0.1], 'LineStyle', ':');

            yyaxis right
            g(end+1)=plot([ (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period, ...
                            (j)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period   ],...
                            [ acker_experiment(acker_experiment_idx(i)).about.tested_steer(j), ...
                            acker_experiment(acker_experiment_idx(i)).about.tested_steer(j)], '-ob');
            g(end+1)=plot([ (j-1)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period, ...
                            (j)*0.02*acker_experiment(acker_experiment_idx(i)).about.each_period   ],...
                            [ acker_experiment(acker_experiment_idx(i)).about.tested_drive(j), ...
                            acker_experiment(acker_experiment_idx(i)).about.tested_drive(j)], '-or');
            ylim([-3 3])
        end

        yyaxis left
    
    end
    legend(g(2:end),'left', 'right')
    xlabel('time in seconds')
    ylabel('angle error in degrees')
    title(sprintf('error from what happened to polynomials | imp [ %7.4f %7.4f %7.4f ] | ref [ %7.4f %7.4f ]', acker_experiment(acker_experiment_idx(i)).about.solimp, acker_experiment(acker_experiment_idx(i)).about.solref))
    grid
end
