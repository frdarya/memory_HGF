function results = analyze_subsequent_trial_parameters_comprehensive(output_exposure, data)
% Comprehensive version with ALL comparisons to control for confounds

fprintf('=== COMPREHENSIVE PARAMETER ANALYSIS (ALL COMPARISONS) ===\n');

% Get unique subjects
uniqueSubs = unique(data.subj);
n_subjects = length(uniqueSubs);

% Initialize comprehensive subject-level storage
subject_results = struct();

highlight_trials = struct('subj',[],'trial_idx',[],'PE2',[],'type',{{}});

fprintf('Processing %d subjects...\n', n_subjects);

%% EXTRACT PARAMETERS FOR EACH SUBJECT
for s = 1:n_subjects
    subID = uniqueSubs(s);
    subj_data = data(data.subj == subID, :);

    % Check if we have HGF output for this subject
    if s > length(output_exposure) || isempty(output_exposure(s).traj)
        fprintf('Warning: No HGF output for subject %d\n', subID);
        continue;
    end

    % Extract HGF trajectories
    mu1 = output_exposure(s).traj.mu(:,1);
    mu2 = output_exposure(s).traj.mu(:,2);
    mu3 = output_exposure(s).traj.mu(:,3);

    sigma1 = sqrt(output_exposure(s).traj.sa(:,1));
    sigma2 = sqrt(output_exposure(s).traj.sa(:,2));
    sigma3 = sqrt(output_exposure(s).traj.sa(:,3));

    PE1 = output_exposure(s).traj.da(:,1);
    PE2 = output_exposure(s).traj.da(:,2);
    PE3 = output_exposure(s).traj.da(:,3);

    precision1 = 1 ./ output_exposure(s).traj.sa(:,1);
    precision2 = 1 ./ output_exposure(s).traj.sa(:,2);
    precision3 = 1 ./ output_exposure(s).traj.sa(:,3);

    weighted_PE1 = output_exposure(s).traj.epsi(:,1);
    weighted_PE2 = output_exposure(s).traj.epsi(:,2);
    weighted_PE3 = output_exposure(s).traj.epsi(:,3);

    % Store PE2 for this subject for plotting
    subject_trial_idx = (1:length(PE2))';

    % Check trajectory length
    if length(mu1) ~= height(subj_data)
        fprintf('Warning: Trajectory length mismatch for subject %d\n', subID);
        continue;
    end

    % Storage for this subject's trials
    conditions = {'miss_to_f1_cr', 'miss_to_f1_fa', 'hit_to_f1_cr', 'hit_to_f1_fa', ...
        'fa_to_target_hit', 'fa_to_target_miss', 'cr_to_target_hit', 'cr_to_target_miss'};
    trial_data = struct();
    for c = 1:length(conditions)
        cond = conditions{c};
        trial_data.(cond) = struct(...
            'mu2', [], 'mu1', [], 'sigma2', [], 'performance', [], ...
            'PE1', [], 'PE2', [], 'PE3', [], ...
            'precision1', [], 'precision2', [], 'precision3', [], ...
            'weighted_PE1', [], 'weighted_PE2', [], 'weighted_PE3', []);
    end

    %% 1. TARGET MISS → F1 ANALYSIS (SPLIT BY F1 OUTCOME)
    target_miss_trials = find(strcmp(subj_data.Item_Type, 'Target') & ...
        subj_data.Response == 0 & ...
        subj_data.Correctness == 0);

    for i = 1:length(target_miss_trials)
        miss_trial = target_miss_trials(i);
        set_id = subj_data.SetID(miss_trial);

        subsequent_f1 = find(subj_data.SetID == set_id & ...
            strcmp(subj_data.Item_Type, 'F1') & ...
            subj_data.Trial_Number > subj_data.Trial_Number(miss_trial));

        if ~isempty(subsequent_f1)
            next_f1 = subsequent_f1(1);
            f1_outcome = subj_data.Correctness(next_f1); % 1=CR, 0=FA

            if f1_outcome == 1  % F1 CR
                cond = 'miss_to_f1_cr';
                % === NEW: Highlight Miss→F1_CR ===
                highlight_trials.subj(end+1) = subID;
                highlight_trials.trial_idx(end+1) = subject_trial_idx(next_f1);
                highlight_trials.PE2(end+1) = PE2(next_f1);
                highlight_trials.type{end+1} = 'Miss_F1_CR';

            else
                cond = 'miss_to_f1_fa';
            end

            % Store parameters
            trial_data.(cond).mu2(end+1) = mu2(next_f1);
            trial_data.(cond).mu1(end+1) = mu1(next_f1);
            trial_data.(cond).sigma2(end+1) = sigma2(next_f1);
            trial_data.(cond).performance(end+1) = f1_outcome;
            trial_data.(cond).PE1(end+1) = PE1(next_f1);
            trial_data.(cond).PE2(end+1) = PE2(next_f1);
            trial_data.(cond).PE3(end+1) = PE3(next_f1);
            trial_data.(cond).precision1(end+1) = precision1(next_f1);
            trial_data.(cond).precision2(end+1) = precision2(next_f1);
            trial_data.(cond).precision3(end+1) = precision3(next_f1);
            trial_data.(cond).weighted_PE1(end+1) = weighted_PE1(next_f1);
            trial_data.(cond).weighted_PE2(end+1) = weighted_PE2(next_f1);
            trial_data.(cond).weighted_PE3(end+1) = weighted_PE3(next_f1);
        end
    end

    %% 2. TARGET HIT → F1 ANALYSIS (SPLIT BY F1 OUTCOME)
    target_hit_trials = find(strcmp(subj_data.Item_Type, 'Target') & ...
        subj_data.Response == 1 & ...
        subj_data.Correctness == 1);

    for i = 1:length(target_hit_trials)
        hit_trial = target_hit_trials(i);
        set_id = subj_data.SetID(hit_trial);

        subsequent_f1 = find(subj_data.SetID == set_id & ...
            strcmp(subj_data.Item_Type, 'F1') & ...
            subj_data.Trial_Number > subj_data.Trial_Number(hit_trial));

        if ~isempty(subsequent_f1)
            next_f1 = subsequent_f1(1);
            f1_outcome = subj_data.Correctness(next_f1);

            % Determine condition
            if f1_outcome == 1  % F1 CR
                cond = 'hit_to_f1_cr';
            else  % F1 FA
                cond = 'hit_to_f1_fa';
            end

            % Store parameters
            trial_data.(cond).mu2(end+1) = mu2(next_f1);
            trial_data.(cond).mu1(end+1) = mu1(next_f1);
            trial_data.(cond).sigma2(end+1) = sigma2(next_f1);
            trial_data.(cond).performance(end+1) = f1_outcome;
            trial_data.(cond).PE1(end+1) = PE1(next_f1);
            trial_data.(cond).PE2(end+1) = PE2(next_f1);
            trial_data.(cond).PE3(end+1) = PE3(next_f1);
            trial_data.(cond).precision1(end+1) = precision1(next_f1);
            trial_data.(cond).precision2(end+1) = precision2(next_f1);
            trial_data.(cond).precision3(end+1) = precision3(next_f1);
            trial_data.(cond).weighted_PE1(end+1) = weighted_PE1(next_f1);
            trial_data.(cond).weighted_PE2(end+1) = weighted_PE2(next_f1);
            trial_data.(cond).weighted_PE3(end+1) = weighted_PE3(next_f1);
        end
    end

    %% 3. F1 FALSE ALARM → TARGET ANALYSIS (SPLIT BY TARGET OUTCOME)
    f1_fa_trials = find(strcmp(subj_data.Item_Type, 'F1') & ...
        subj_data.Response == 1 & ...
        subj_data.Correctness == 0);

    for i = 1:length(f1_fa_trials)
        fa_trial = f1_fa_trials(i);
        set_id = subj_data.SetID(fa_trial);

        subsequent_target = find(subj_data.SetID == set_id & ...
            strcmp(subj_data.Item_Type, 'Target') & ...
            subj_data.Trial_Number > subj_data.Trial_Number(fa_trial));

        if ~isempty(subsequent_target)
            next_target = subsequent_target(1);
            target_outcome = subj_data.Correctness(next_target); % 1=hit, 0=miss

            if target_outcome == 1
                cond = 'fa_to_target_hit';
            else
                cond = 'fa_to_target_miss';
            end

            % Store parameters
            trial_data.(cond).mu2(end+1) = mu2(next_target);
            trial_data.(cond).mu1(end+1) = mu1(next_target);
            trial_data.(cond).sigma2(end+1) = sigma2(next_target);
            trial_data.(cond).performance(end+1) = target_outcome;
            trial_data.(cond).PE1(end+1) = PE1(next_target);
            trial_data.(cond).PE2(end+1) = PE2(next_target);
            trial_data.(cond).PE3(end+1) = PE3(next_target);
            trial_data.(cond).precision1(end+1) = precision1(next_target);
            trial_data.(cond).precision2(end+1) = precision2(next_target);
            trial_data.(cond).precision3(end+1) = precision3(next_target);
            trial_data.(cond).weighted_PE1(end+1) = weighted_PE1(next_target);
            trial_data.(cond).weighted_PE2(end+1) = weighted_PE2(next_target);
            trial_data.(cond).weighted_PE3(end+1) = weighted_PE3(next_target);

            % === NEW: Highlight FA→Target ===
            % highlight_trials.subj(end+1) = subID;
            % highlight_trials.trial_idx(end+1) = subject_trial_idx(next_target);
            % highlight_trials.PE2(end+1) = PE2(next_target);
            % highlight_trials.type{end+1} = 'FA_Target';

        end
    end



    %% 4. F1 CORRECT REJECTION → TARGET ANALYSIS (SPLIT BY TARGET OUTCOME)
    f1_cr_trials = find(strcmp(subj_data.Item_Type, 'F1') & ...
        subj_data.Response == 0 & ...
        subj_data.Correctness == 1);

    for i = 1:length(f1_cr_trials)
        cr_trial = f1_cr_trials(i);
        set_id = subj_data.SetID(cr_trial);

        subsequent_target = find(subj_data.SetID == set_id & ...
            strcmp(subj_data.Item_Type, 'Target') & ...
            subj_data.Trial_Number > subj_data.Trial_Number(cr_trial));

        if ~isempty(subsequent_target)
            next_target = subsequent_target(1);
            target_outcome = subj_data.Correctness(next_target);

            % Determine condition
            if target_outcome == 1
                cond = 'cr_to_target_hit';
            else
                cond = 'cr_to_target_miss';
            end

            % Store parameters
            trial_data.(cond).mu2(end+1) = mu2(next_target);
            trial_data.(cond).mu1(end+1) = mu1(next_target);
            trial_data.(cond).sigma2(end+1) = sigma2(next_target);
            trial_data.(cond).performance(end+1) = target_outcome;
            trial_data.(cond).PE1(end+1) = PE1(next_target);
            trial_data.(cond).PE2(end+1) = PE2(next_target);
            trial_data.(cond).PE3(end+1) = PE3(next_target);
            trial_data.(cond).precision1(end+1) = precision1(next_target);
            trial_data.(cond).precision2(end+1) = precision2(next_target);
            trial_data.(cond).precision3(end+1) = precision3(next_target);
            trial_data.(cond).weighted_PE1(end+1) = weighted_PE1(next_target);
            trial_data.(cond).weighted_PE2(end+1) = weighted_PE2(next_target);
            trial_data.(cond).weighted_PE3(end+1) = weighted_PE3(next_target);
        end
    end

    %% CALCULATE SUBJECT-LEVEL AVERAGES
    conditions = {'miss_to_f1_cr', 'miss_to_f1_fa', 'hit_to_f1_cr', 'hit_to_f1_fa', ...
        'fa_to_target_hit', 'fa_to_target_miss', 'cr_to_target_hit', 'cr_to_target_miss'};

    param_names = {'mu2', 'mu1', 'sigma2', 'performance', ...
        'PE1', 'PE2', 'PE3', ...
        'precision1', 'precision2', 'precision3', ...
        'weighted_PE1', 'weighted_PE2', 'weighted_PE3'};

    for c = 1:length(conditions)
        cond = conditions{c};

        if ~isempty(trial_data.(cond).mu2)
            for p = 1:length(param_names)
                param = param_names{p};
                if ~isempty(trial_data.(cond).(param))
                    subject_results(s).(cond).(param) = mean(trial_data.(cond).(param));
                else
                    subject_results(s).(cond).(param) = NaN;
                end
            end
            subject_results(s).(cond).n_trials = length(trial_data.(cond).mu2);
        else
            for p = 1:length(param_names)
                param = param_names{p};
                subject_results(s).(cond).(param) = NaN;
            end
            subject_results(s).(cond).n_trials = 0;
        end
    end

    fprintf('Subject %d: Miss→F1[CR=%d,FA=%d], Hit→F1[CR=%d,FA=%d], FA→Target[Hit=%d,Miss=%d], CR→Target[Hit=%d,Miss=%d]\n', ...
        subID, ...
        subject_results(s).miss_to_f1_cr.n_trials, subject_results(s).miss_to_f1_fa.n_trials, ...
        subject_results(s).hit_to_f1_cr.n_trials, subject_results(s).hit_to_f1_fa.n_trials, ...
        subject_results(s).fa_to_target_hit.n_trials, subject_results(s).fa_to_target_miss.n_trials, ...
        subject_results(s).cr_to_target_hit.n_trials, subject_results(s).cr_to_target_miss.n_trials);

    % Plot per subject
    % figure; hold on;
    % plot(subject_trial_idx, PE2, 'k-');
    %
    % fa_idx = strcmp(highlight_trials.type, 'FA_Target') & highlight_trials.subj == subID;
    % miss_idx = strcmp(highlight_trials.type, 'Miss_F1_CR') & highlight_trials.subj == subID;
    %
    % plot(highlight_trials.trial_idx(fa_idx), ...
    %      highlight_trials.PE2(fa_idx), 'ro', 'MarkerFaceColor', 'r');
    %
    % plot(highlight_trials.trial_idx(miss_idx), ...
    %      highlight_trials.PE2(miss_idx), 'bs', 'MarkerFaceColor', 'b');
    %
    % legend({'PE₂ trajectory','FA→Target','Miss→F1_CR'});
    % xlabel('Trial');
    % ylabel('PE₂');
    % title(sprintf('Subject %d', subID));
    % box off;

end

%% ROBUST T-TEST FUNCTION
    function [h, p_val, ci, stats, cohen_d, success] = robust_ttest(data1, data2, param_name)
        success = false;
        h = NaN; p_val = NaN; ci = [NaN, NaN];
        stats = struct('tstat', NaN, 'df', NaN);
        cohen_d = NaN;

        if length(data1) < 3 || length(data2) < 3
            return;
        end

        % Clean data
        valid_idx = ~isnan(data1) & ~isnan(data2) & ~isinf(data1) & ~isinf(data2);
        data1_clean = data1(valid_idx);
        data2_clean = data2(valid_idx);

        if length(data1_clean) < 3 || std(data1_clean - data2_clean) < 1e-10
            return;
        end

        try
            [h, p_val, ci, stats] = ttest(data1_clean, data2_clean);
            cohen_d = mean(data1_clean - data2_clean) / std(data1_clean - data2_clean);
            success = true;
        catch ME
            fprintf('    WARNING: t-test failed for %s: %s\n', param_name, ME.message);
        end
    end

%% COMPREHENSIVE STATISTICAL COMPARISONS
fprintf('\n=== COMPREHENSIVE STATISTICAL COMPARISONS ===\n');

test_params = {'mu2', 'PE1', 'PE2', 'PE3', 'precision1', 'precision2', 'precision3', ...
    'weighted_PE1', 'weighted_PE2', 'weighted_PE3'};
param_labels = {'μ₂ (volatility)', 'PE₁ (stimulus)', 'PE₂ (volatility)', 'PE₃ (meta-vol)', ...
    'Precision₁', 'Precision₂', 'Precision₃', ...
    'Weighted PE₁', 'Weighted PE₂', 'Weighted PE₃'};

%% 1. MAIN EFFECTS
fprintf('\n--- A. MAIN EFFECTS  ---\n');

% Miss→F1 vs Hit→F1 (pooled across outcomes)
fprintf('\n1. TARGET ERROR EFFECT (Miss→F1 vs Hit→F1 ):\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    % Pool across outcomes
    miss_pooled = [];
    hit_pooled = [];

    for s = 1:n_subjects
        % Miss→F1 (both CRs and FAs)
        miss_cr_val = subject_results(s).miss_to_f1_cr.(param);
        miss_fa_val = subject_results(s).miss_to_f1_fa.(param);
        hit_cr_val = subject_results(s).hit_to_f1_cr.(param);
        hit_fa_val = subject_results(s).hit_to_f1_fa.(param);

        % Pool by weighting by number of trials
        n_miss_cr = subject_results(s).miss_to_f1_cr.n_trials;
        n_miss_fa = subject_results(s).miss_to_f1_fa.n_trials;
        n_hit_cr = subject_results(s).hit_to_f1_cr.n_trials;
        n_hit_fa = subject_results(s).hit_to_f1_fa.n_trials;

        if (n_miss_cr + n_miss_fa) > 0 && (n_hit_cr + n_hit_fa) > 0
            % Weighted average for Miss→F1
            if ~isnan(miss_cr_val) && ~isnan(miss_fa_val)
                miss_pooled(end+1) = (miss_cr_val*n_miss_cr + miss_fa_val*n_miss_fa) / (n_miss_cr + n_miss_fa);
            elseif ~isnan(miss_cr_val)
                miss_pooled(end+1) = miss_cr_val;
            elseif ~isnan(miss_fa_val)
                miss_pooled(end+1) = miss_fa_val;
            end

            % Weighted average for Hit→F1
            if ~isnan(hit_cr_val) && ~isnan(hit_fa_val)
                hit_pooled(end+1) = (hit_cr_val*n_hit_cr + hit_fa_val*n_hit_fa) / (n_hit_cr + n_hit_fa);
            elseif ~isnan(hit_cr_val)
                hit_pooled(end+1) = hit_cr_val;
            elseif ~isnan(hit_fa_val)
                hit_pooled(end+1) = hit_fa_val;
            end
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(miss_pooled, hit_pooled, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(miss_pooled) > mean(hit_pooled)
                fprintf(' (Miss→F1 higher)');
            else
                fprintf(' (Hit→F1 higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(miss_pooled));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(miss_pooled));
    end
end

% FA→Target vs CR→Target
fprintf('\n2. F1 ERROR EFFECT (FA→Target vs CR→Target):\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    % Pool across outcomes
    fa_pooled = [];
    cr_pooled = [];

    for s = 1:n_subjects
        % Get values
        fa_hit_val = subject_results(s).fa_to_target_hit.(param);
        fa_miss_val = subject_results(s).fa_to_target_miss.(param);
        cr_hit_val = subject_results(s).cr_to_target_hit.(param);
        cr_miss_val = subject_results(s).cr_to_target_miss.(param);

        % Get trial counts
        n_fa_hit = subject_results(s).fa_to_target_hit.n_trials;
        n_fa_miss = subject_results(s).fa_to_target_miss.n_trials;
        n_cr_hit = subject_results(s).cr_to_target_hit.n_trials;
        n_cr_miss = subject_results(s).cr_to_target_miss.n_trials;

        if (n_fa_hit + n_fa_miss) > 0 && (n_cr_hit + n_cr_miss) > 0
            % Weighted average for FA
            if ~isnan(fa_hit_val) && ~isnan(fa_miss_val)
                fa_pooled(end+1) = (fa_hit_val*n_fa_hit + fa_miss_val*n_fa_miss) / (n_fa_hit + n_fa_miss);
            elseif ~isnan(fa_hit_val)
                fa_pooled(end+1) = fa_hit_val;
            elseif ~isnan(fa_miss_val)
                fa_pooled(end+1) = fa_miss_val;
            end

            % Weighted average for CR
            if ~isnan(cr_hit_val) && ~isnan(cr_miss_val)
                cr_pooled(end+1) = (cr_hit_val*n_cr_hit + cr_miss_val*n_cr_miss) / (n_cr_hit + n_cr_miss);
            elseif ~isnan(cr_hit_val)
                cr_pooled(end+1) = cr_hit_val;
            elseif ~isnan(cr_miss_val)
                cr_pooled(end+1) = cr_miss_val;
            end
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(fa_pooled, cr_pooled, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(fa_pooled) > mean(cr_pooled)
                fprintf(' (FA→Target higher)');
            else
                fprintf(' (CR→Target higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(fa_pooled));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(fa_pooled));
    end
end

%% 2. OUTCOME-CONTROLLED COMPARISONS
fprintf('\n--- B. OUTCOME-CONTROLLED COMPARISONS ---\n');

% Compare HITS only: FA→Target_Hit vs CR→Target_Hit
fprintf('\n3. HITS ONLY: FA→Target_Hit vs CR→Target_Hit:\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    fa_hit_data = [];
    cr_hit_data = [];

    for s = 1:n_subjects
        if ~isnan(subject_results(s).fa_to_target_hit.(param)) && ...
                ~isnan(subject_results(s).cr_to_target_hit.(param)) && ...
                subject_results(s).fa_to_target_hit.n_trials > 0 && ...
                subject_results(s).cr_to_target_hit.n_trials > 0

            fa_hit_data(end+1) = subject_results(s).fa_to_target_hit.(param);
            cr_hit_data(end+1) = subject_results(s).cr_to_target_hit.(param);
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(fa_hit_data, cr_hit_data, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(fa_hit_data) > mean(cr_hit_data)
                fprintf(' (FA→Hit higher)');
            else
                fprintf(' (CR→Hit higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(fa_hit_data));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(fa_hit_data));
    end
end

% Compare MISSES only: FA→Target_Miss vs CR→Target_Miss
fprintf('\n4. MISSES ONLY: FA→Target_Miss vs CR→Target_Miss:\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    fa_miss_data = [];
    cr_miss_data = [];

    for s = 1:n_subjects
        if ~isnan(subject_results(s).fa_to_target_miss.(param)) && ...
                ~isnan(subject_results(s).cr_to_target_miss.(param)) && ...
                subject_results(s).fa_to_target_miss.n_trials > 0 && ...
                subject_results(s).cr_to_target_miss.n_trials > 0

            fa_miss_data(end+1) = subject_results(s).fa_to_target_miss.(param);
            cr_miss_data(end+1) = subject_results(s).cr_to_target_miss.(param);
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(fa_miss_data, cr_miss_data, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(fa_miss_data) > mean(cr_miss_data)
                fprintf(' (FA→Miss higher)');
            else
                fprintf(' (CR→Miss higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(fa_miss_data));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(fa_miss_data));
    end
end

% Compare F1 CRs only: Miss→F1_CR vs Hit→F1_CR
fprintf('\n5. F1 CRs ONLY: Miss→F1_CR vs Hit→F1_CR:\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    miss_f1_cr_data = [];
    hit_f1_cr_data = [];

    for s = 1:n_subjects
        if ~isnan(subject_results(s).miss_to_f1_cr.(param)) && ...
                ~isnan(subject_results(s).hit_to_f1_cr.(param)) && ...
                subject_results(s).miss_to_f1_cr.n_trials > 0 && ...
                subject_results(s).hit_to_f1_cr.n_trials > 0

            miss_f1_cr_data(end+1) = subject_results(s).miss_to_f1_cr.(param);
            hit_f1_cr_data(end+1) = subject_results(s).hit_to_f1_cr.(param);
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(miss_f1_cr_data, hit_f1_cr_data, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(miss_f1_cr_data) > mean(hit_f1_cr_data)
                fprintf(' (Miss→F1_CR higher)');
            else
                fprintf(' (Hit→F1_CR higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(miss_f1_cr_data));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(miss_f1_cr_data));
    end
end

% Compare F1 FAs only: Miss→F1_FA vs Hit→F1_FA
fprintf('\n6. F1 FAs ONLY: Miss→F1_FA vs Hit→F1_FA:\n');

for p = 1:length(test_params)
    param = test_params{p};
    label = param_labels{p};

    miss_f1_fa_data = [];
    hit_f1_fa_data = [];

    for s = 1:n_subjects
        if ~isnan(subject_results(s).miss_to_f1_fa.(param)) && ...
                ~isnan(subject_results(s).hit_to_f1_fa.(param)) && ...
                subject_results(s).miss_to_f1_fa.n_trials > 0 && ...
                subject_results(s).hit_to_f1_fa.n_trials > 0

            miss_f1_fa_data(end+1) = subject_results(s).miss_to_f1_fa.(param);
            hit_f1_fa_data(end+1) = subject_results(s).hit_to_f1_fa.(param);
        end
    end

    [h, p_val, ci, stats, cohen_d, success] = robust_ttest(miss_f1_fa_data, hit_f1_fa_data, label);

    if success
        fprintf('  %s: t(%d)=%.3f, p=%.4f, d=%.3f', ...
            label, stats.df, stats.tstat, p_val, cohen_d);

        if p_val < 0.05
            fprintf(' *');
            if mean(miss_f1_fa_data) > mean(hit_f1_fa_data)
                fprintf(' (Miss→F1_FA higher)');
            else
                fprintf(' (Hit→F1_FA higher)');
            end
        end
        fprintf(' (n=%d pairs)\n', length(miss_f1_fa_data));
    else
        fprintf('  %s: Test failed or insufficient data (n=%d pairs)\n', label, length(miss_f1_fa_data));
    end
end

%% 3. SAMPLE SIZE SUMMARY
fprintf('\n--- C. SAMPLE SIZE SUMMARY ---\n');

all_conditions = {'miss_to_f1_cr', 'miss_to_f1_fa', 'hit_to_f1_cr', 'hit_to_f1_fa', ...
    'fa_to_target_hit', 'fa_to_target_miss', 'cr_to_target_hit', 'cr_to_target_miss'};
condition_labels = {'Miss→F1_CR', 'Miss→F1_FA', 'Hit→F1_CR', 'Hit→F1_FA', ...
    'FA→Target_Hit', 'FA→Target_Miss', 'CR→Target_Hit', 'CR→Target_Miss'};

fprintf('Condition\t\tSubjects\tTotal Trials\tMean Trials/Subj\n');
fprintf('--------\t\t--------\t-----------\t----------------\n');

for c = 1:length(all_conditions)
    cond = all_conditions{c};

    n_subj_with_data = 0;
    total_trials = 0;

    for s = 1:n_subjects
        if subject_results(s).(cond).n_trials > 0
            n_subj_with_data = n_subj_with_data + 1;
            total_trials = total_trials + subject_results(s).(cond).n_trials;
        end
    end

    mean_trials = iff(n_subj_with_data > 0, total_trials / n_subj_with_data, 0);

    fprintf('%-15s\t%d\t\t%d\t\t%.1f\n', ...
        condition_labels{c}, n_subj_with_data, total_trials, mean_trials);
end

% Store comprehensive results
results = struct();
results.subject_results = subject_results;
results.all_conditions = all_conditions;
results.condition_labels = condition_labels;
results.highlight_trials = highlight_trials;

% Store comprehensive results
results = struct();
results.subject_results = subject_results;
results.highlight_trials = highlight_trials;


fprintf('\n=== ANALYSIS COMPLETE ===\n');

end

function out = iff(condition, true_val, false_val)
% Inline if function
if condition
    out = true_val;
else
    out = false_val;
end
end