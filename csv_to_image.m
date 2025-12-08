% Replicates the graphs from https://arxiv.org/pdf/2507.11217 as images

function plot_losses(csvFiles, legendNames, outFile, plotTitle, colours)
% plot_losses(csvFiles, legendNames, outFile, plotTitle)
%
% csvFiles    = cell array of paths to CSV files
% legendNames = cell array of labels for each curve
% outFile     = filename to save the figure
% plotTitle   = title of the plot

    if length(csvFiles) ~= length(legendNames)
        error('csvFiles and legendNames must be the same length.');
    end

    figure('Position', [200 200 600 450]); hold on;

    colors = lines(length(csvFiles));   % distinct MATLAB colors

    for i = 1:length(csvFiles)
        data = readmatrix(csvFiles{i});
        loss = data(:, end);
        epochs = 1:length(loss);

        plot(epochs, loss, 'LineWidth', 2.5, 'Color', 0.9*colours{i});
    end

    xlabel('Training Period', 'FontSize', 14);
    ylabel('Loss', 'FontSize', 14);
    title(plotTitle, 'FontSize', 24, 'FontWeight', 'bold');

    legend(legendNames, 'FontSize', 14, 'Location', 'northeast');

    grid on;
    box on;

    % Tight layout and save
    % set(gca, 'FontSize', 12);
    saveas(gcf, outFile);

    hold off;
end

plot_losses({'run-QAE_CiFAR_1_layer-tag-Loss_train_avg_epoch.csv', ...
    'run-QAE_CiFAR_2_Layer-tag-Loss_train_avg_epoch.csv', ...
    'run-QAE_CiFAR_3_Layer-tag-Loss_train_avg_epoch.csv', ...
    'run-SEN_CiFAR-tag-Loss_train_avg_epoch.csv'...
    }, {'QAE-Net-1layer', 'QAE-Net-2layer', 'QAE-Net-3layer', 'SENet'}, ...
    'layersLoss.png', 'Varitational Layers on CIFAR-10', ...
    {[0,0,0],[0,1,0],[1,0,0],[0,0,1]});

plot_losses({'run-QAE_MNIST-tag-Loss_train_avg_epoch.csv', ...
    'run-SEN_MNIST-tag-Loss_train_avg_epoch.csv', ...
    }, {'QAE-Net', 'SENet'}, ...
    'MNISTLoss.png', 'MNIST', ...
    {[1,0,0],[0,0,1]});

plot_losses({'run-QAE_F_MNIST-tag-Loss_train_avg_epoch.csv', ...
    'run-SEN_F_MNIST-tag-Loss_train_avg_epoch.csv', ...
    }, {'QAE-Net', 'SENet'}, ...
    'FMNISTLoss.png', 'FashionMNIST', ...
    {[1,0,0],[0,0,1]});

plot_losses({'run-QAE_CiFAR_1_layer-tag-Loss_train_avg_epoch.csv', ...
    'run-SEN_CiFAR-tag-Loss_train_avg_epoch.csv', ...
    }, {'QAE-Net', 'SENet'}, ...
    'CIFARLoss.png', 'CIFAR-10', ...
    {[1,0,0],[0,0,1]});