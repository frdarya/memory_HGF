function validation = validate_similarity_effects(output_results, data, model_name)
% Validate similarity effects for a single model
% Your exact plotting approach with additional diagnostics

uniqueSubs = unique(data.subj);
n_subs = length(uniqueSubs);

similarity_levels = [1.0, 0.75, 0.5, 0.25];
level_names = {'Target', 'F1', 'F2', 'F3'};

% Preallocate
model_p_old = NaN(n_subs, 4);
actual_p_old = NaN(n_subs, 4);

fprintf('\n=== %s Model Similarity Validation ===\n', model_name);

for s = 1:n_subs
    subID = uniqueSubs(s);
    subj_data = data(data.subj == subID, :);
    
    % Get model predictions
    if s <= length(output_results) && ~isempty(output_results(s).optim)
        model_pred = output_results(s).optim.yhat;
        model_pred_bin = output_results(s).optim.yhat >= 0.5;
        
        % Extract by similarity level
        target_trials = subj_data.Similarity == 1.0;
        f1_trials = subj_data.Similarity == 0.75;
        f2_trials = subj_data.Similarity == 0.5;
        f3_trials = subj_data.Similarity == 0.25;
        
        if sum(target_trials) > 0
            model_p_old(s, 1) = mean(model_pred(target_trials));
            actual_p_old(s, 1) = mean(subj_data.Response(target_trials));
        end
        if sum(f1_trials) > 0
            model_p_old(s, 2) = mean(model_pred(f1_trials));
            actual_p_old(s, 2) = mean(subj_data.Response(f1_trials));
        end
        if sum(f2_trials) > 0
            model_p_old(s, 3) = mean(model_pred(f2_trials));
            actual_p_old(s, 3) = mean(subj_data.Response(f2_trials));
        end
        if sum(f3_trials) > 0
            model_p_old(s, 4) = mean(model_pred(f3_trials));
            actual_p_old(s, 4) = mean(subj_data.Response(f3_trials));
        end
    end
end

%% Your exact visualization
figure('Name', sprintf('%s Similarity Effects', model_name), 'Position', [100, 100, 1000, 400]);

errorbar(1:4, nanmean(model_p_old, 1), nanstd(model_p_old, 1)/sqrt(n_subs), 'bo-', 'LineWidth', 2);
hold on;
errorbar(1:4, nanmean(actual_p_old, 1), nanstd(actual_p_old, 1)/sqrt(n_subs), 'ro-', 'LineWidth', 2);
xlabel('Similarity Level');
ylabel('P("old")');
title('Group-Level Similarity Effects');
legend('Model', 'Actual', 'Location', 'northeast');
set(gca, 'XTick', 1:4, 'XTickLabel', level_names);
grid on;
ylim([0, 1]);


%% Additional diagnostics
fprintf('\nSimilarity Level Analysis:\n');
fprintf('Level\tSim\tModel P("old")\tActual P("old")\tDifference\tCorrelation\n');
fprintf('-----\t----\t-----------\t------------\t----------\t-----------\n');

level_correlations = NaN(4,1);
for level = 1:4
    model_mean = nanmean(model_p_old(:, level));
    actual_mean = nanmean(actual_p_old(:, level));
    difference = model_mean - actual_mean;
    
    % Level-specific correlation
    valid_level = ~isnan(model_p_old(:, level)) & ~isnan(actual_p_old(:, level));
    if sum(valid_level) > 5
        level_correlations(level) = corr(model_p_old(valid_level, level), ...
                                        actual_p_old(valid_level, level));
    end
    
    fprintf('%s\t%.2f\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
        level_names{level}, similarity_levels(level), ...
        model_mean, actual_mean, difference, level_correlations(level));
end

% Test monotonic decrease
model_gradient_group = nanmean(model_p_old, 1);
actual_gradient_group = nanmean(actual_p_old, 1);

model_monotonic = all(diff(model_gradient_group) < 0);
actual_monotonic = all(diff(actual_gradient_group) < 0);

% Save plot
% saveas(gcf, sprintf('%s_similarity_validation.png', model_name));
% fprintf('Plot saved as: %s_similarity_validation.png\n', model_name);

% Return validation metrics
validation.model_p_old = model_p_old;
validation.actual_p_old = actual_p_old;
validation.level_correlations = level_correlations;

end