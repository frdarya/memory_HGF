function analyze_exposure_by_similarity(output_exposure, data)
% Analyze exposure effects by similarity level (reconstructing weighted exposure)


fprintf('=== RECONSTRUCTING WEIGHTED EXPOSURE ===\n');

uniqueSubs = unique(data.subj);
n_subjects = length(uniqueSubs);

fprintf('Found %d unique subjects: %s\n', n_subjects, mat2str(uniqueSubs));
fprintf('Model outputs available: %d\n', length(output_exposure));

if n_subjects ~= length(output_exposure)
    error('Mismatch: %d subjects in data, %d model outputs', n_subjects, length(output_exposure));
end

all_results = [];

for s = 1:numel(uniqueSubs)
    subID = uniqueSubs(s);  % Get actual subject ID
    fprintf('Processing subject %d (index %d)...', subID, s);
    
    % Get subject data using actual subject ID  
    subj_data = data(data.subj == subID, :);
    % Reconstruct weighted exposure for this subject
    weighted_exposure = compute_exposure_weights_single_subject(subj_data);
    
    % Get model predictions
    predictions = output_exposure(s).optim.yhat;
    
    % Check lengths match
    fprintf(' Data=%d, Predictions=%d', height(subj_data), length(predictions));
    
    if length(predictions) ~= height(subj_data)
        fprintf(' - LENGTH MISMATCH! Skipping subject %d\n', s);
        continue;
    end
    
    % Store results with matching lengths
    n_trials = height(subj_data);
    subj_results = table();
    subj_results.subject = repmat(s, n_trials, 1);
    subj_results.similarity = subj_data.Similarity;
    subj_results.response = subj_data.Response;
    subj_results.prediction = predictions(:); % Ensure column vector
    subj_results.exposure = weighted_exposure;
    subj_results.set_id = subj_data.SetID;
    
    all_results = [all_results; subj_results];
    fprintf(' - OK\n');
end

fprintf('Successfully processed %d subjects\n', length(unique(all_results.subject)));

%% ANALYSIS BY SIMILARITY LEVEL
fprintf('\n=== EXPOSURE EFFECTS BY SIMILARITY LEVEL ===\n');

% Group by similarity levels
target_idx = all_results.similarity == 1.0;
f1_idx = all_results.similarity == 0.75;
f2_idx = all_results.similarity == 0.5;
f3_idx = all_results.similarity == 0.25;

similarity_groups = {target_idx, f1_idx, f2_idx, f3_idx};
sim_names = {'Targets', 'F1', 'F2', 'F3'};
sim_values = [1.0, 0.75, 0.5, 0.25];

% Analyze each similarity level
for i = 1:4
    idx = similarity_groups{i};
    if sum(idx) == 0
        continue;
    end
    
    % Get data for this similarity level
    preds = all_results.prediction(idx);
    expos = all_results.exposure(idx);
    resps = all_results.response(idx);
    
    % Calculate correlation between exposure and predictions
    [r_pred, p_pred] = corr(expos, preds, 'rows', 'complete');
    [r_resp, p_resp] = corr(expos, resps, 'rows', 'complete');
    
    % Divide into low/high exposure groups for clearer comparison
    if length(expos) > 10 % Need sufficient data
        median_exposure = median(expos(expos > 0)); % Exclude zero exposures
        low_exp_idx = expos <= median_exposure;
        high_exp_idx = expos > median_exposure;
        
        % Model predictions
        pred_low_exp = mean(preds(low_exp_idx));
        pred_high_exp = mean(preds(high_exp_idx));
        exp_effect_model = pred_high_exp - pred_low_exp;
        
        % Human responses
        resp_low_exp = mean(resps(low_exp_idx));
        resp_high_exp = mean(resps(high_exp_idx));
        exp_effect_human = resp_high_exp - resp_low_exp;
        
        fprintf('%s (%.2f similarity): n=%d trials\n', sim_names{i}, sim_values(i), sum(idx));
        fprintf('  Exposure-Prediction correlation: r=%.3f, p=%.4f\n', r_pred, p_pred);
        fprintf('  Exposure-Response correlation:   r=%.3f, p=%.4f\n', r_resp, p_resp);
        fprintf('  Low exposure (≤%.1f):  Model=%.3f, Human=%.3f (n=%d)\n', ...
            median_exposure, pred_low_exp, resp_low_exp, sum(low_exp_idx));
        fprintf('  High exposure (>%.1f): Model=%.3f, Human=%.3f (n=%d)\n', ...
            median_exposure, pred_high_exp, resp_high_exp, sum(high_exp_idx));
        fprintf('  Exposure effect:        Model=%.3f, Human=%.3f\n\n', exp_effect_model, exp_effect_human);
    end
end

%% COMPARE HIGH VS LOW SIMILARITY
fprintf('=== HIGH vs LOW SIMILARITY COMPARISON ===\n');

% High similarity: Targets + F1
high_sim_idx = target_idx | f1_idx;
% Low similarity: F2 + F3  
low_sim_idx = f2_idx | f3_idx;

if sum(high_sim_idx) > 0 && sum(low_sim_idx) > 0
    % Correlations
    [r_high, p_high] = corr(all_results.exposure(high_sim_idx), all_results.prediction(high_sim_idx));
    [r_low, p_low] = corr(all_results.exposure(low_sim_idx), all_results.prediction(low_sim_idx));
    
    fprintf('Exposure-Prediction Correlations:\n');
    fprintf('  High similarity (Targets+F1): r=%.3f, p=%.4f (n=%d)\n', r_high, p_high, sum(high_sim_idx));
    fprintf('  Low similarity (F2+F3):       r=%.3f, p=%.4f (n=%d)\n', r_low, p_low, sum(low_sim_idx));
    
    % Test difference in correlations using Fisher's z-transform
    if ~isnan(r_high) && ~isnan(r_low)
        z_high = atanh(r_high);
        z_low = atanh(r_low);
        se_diff = sqrt(1/(sum(high_sim_idx)-3) + 1/(sum(low_sim_idx)-3));
        z_diff = (z_high - z_low) / se_diff;
        p_diff = 2 * (1 - normcdf(abs(z_diff)));
        
        fprintf('  Correlation difference test: z=%.2f, p=%.4f\n', z_diff, p_diff);
        
        if p_diff < 0.05
            if abs(r_high) > abs(r_low)
                fprintf('  → High similarity shows STRONGER exposure effects\n');
            else
                fprintf('  → Low similarity shows STRONGER exposure effects\n');
            end
        else
            fprintf('  → No significant difference in exposure effects\n');
        end
    end
end

%% VISUALIZATION
fprintf('\n=== GENERATING PLOTS ===\n');

figure('Position', [100, 100, 1400, 1000]);

% Plot 1: Exposure distribution by similarity
subplot(2,3,1);
group_labels = [];
group_data = [];
for i = 1:4
    idx = similarity_groups{i};
    if sum(idx) > 0
        group_data = [group_data; all_results.exposure(idx)];
        group_labels = [group_labels; i * ones(sum(idx), 1)];
    end
end
boxplot(group_data, group_labels, 'Labels', sim_names);
title('Exposure Distribution by Similarity');
ylabel('Weighted Exposure');

% Plot 2: Predictions vs Exposure (all data)
subplot(2,3,2);
colors = {'r', 'b', 'g', 'm'};
for i = 1:4
    idx = similarity_groups{i};
    if sum(idx) > 0
        scatter(all_results.exposure(idx), all_results.prediction(idx), ...
                15, colors{i}, 'filled', 'DisplayName', sim_names{i});
        hold on;
    end
end
xlabel('Weighted Exposure');
ylabel('Model Predictions');
title('Predictions vs Exposure');
legend('Location', 'best');

% Plot 3-6: Individual similarity levels with trend lines
for i = 1:4
    subplot(2,3,i+2);
    idx = similarity_groups{i};
    if sum(idx) > 20 % Need sufficient data for trend
        x_data = all_results.exposure(idx);
        y_data = all_results.prediction(idx);
        
        scatter(x_data, y_data, 20, colors{i}, 'filled');
        hold on;
        
        % Add trend line
        valid_idx = ~isnan(x_data) & ~isnan(y_data);
        if sum(valid_idx) > 10
            p = polyfit(x_data(valid_idx), y_data(valid_idx), 1);
            x_trend = linspace(min(x_data), max(x_data), 100);
            y_trend = polyval(p, x_trend);
            plot(x_trend, y_trend, colors{i}, 'LineWidth', 2);
            
            [r, p_val] = corr(x_data, y_data, 'rows', 'complete');
            title(sprintf('%s: r=%.3f, p=%.3f', sim_names{i}, r, p_val));
        else
            title(sim_names{i});
        end
        xlabel('Weighted Exposure');
        ylabel('Model Predictions');
    end
end

end

function weighted_exposure = compute_exposure_weights_single_subject(subj_data)
% Reconstruct weighted exposure for a single subject
% Same logic as before but for one subject

n_trials = height(subj_data);
weighted_exposure = zeros(n_trials, 1);

unique_sets = unique(subj_data.SetID);

for set_idx = 1:length(unique_sets)
    current_set = unique_sets(set_idx);
    
    % Find all trials for this set
    set_trials = find(subj_data.SetID == current_set);
    
    % Sort by trial number
    [~, sort_order] = sort(subj_data.Trial_Number(set_trials));
    set_trials = set_trials(sort_order);
    
    % Calculate cumulative exposure
    for i = 1:length(set_trials)
        trial_idx = set_trials(i);
        
        if i == 1
            weighted_exposure(trial_idx) = 0;
        else
            previous_trials = set_trials(1:i-1);
            previous_similarities = subj_data.Similarity(previous_trials);
            weighted_exposure(trial_idx) = sum(previous_similarities);
        end
    end
end
end