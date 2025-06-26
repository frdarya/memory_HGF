function weighted_exposure = compute_exposure_weights(data)
% Pre-compute similarity-weighted exposure for each trial

n_trials = height(data);
weighted_exposure = zeros(n_trials, 1);

% Get unique sets
unique_sets = unique(data.SetID);

for set_idx = 1:length(unique_sets)
    current_set = unique_sets(set_idx);
    
    % Find all trials for this set
    set_trials = find(data.SetID == current_set);
    
    % Sort by trial number to get correct order
    [~, sort_order] = sort(data.Trial_Number(set_trials));
    set_trials = set_trials(sort_order);
    
    % Calculate cumulative exposure for each trial in this set
    for i = 1:length(set_trials)
        trial_idx = set_trials(i);
        
        if i == 1
            % First trial from this set - no previous exposure
            weighted_exposure(trial_idx) = 0;
        else
            % Sum similarities from all previous trials in this set
            previous_trials = set_trials(1:i-1);
            previous_similarities = data.Similarity(previous_trials);
            weighted_exposure(trial_idx) = sum(previous_similarities);
        end
    end
end
end