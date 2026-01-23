
clearvars
addpath(genpath('tapas-6.1.0'))
%% 1) Load your long‐format data
longT = readtable('Exp1_and_Exp2_long_hgf.csv');
%% 2) Build the per‐set summary table
summaryAll = table();
uniqueSubs  = unique(longT.subj);
hgf_all_subjs = struct();
for s = 1:numel(uniqueSubs)
    subID  = uniqueSubs(s);
    subRaw = longT(longT.subj == subID, :);
    y = subRaw.Response;

u = [subRaw.Old_new,subRaw.Similarity, subRaw.SetID]; %set needs to be dummy-coded
raw_ids = u(:,3);
[set_levels, ~, set_idx] = unique(raw_ids, 'stable');

if numel(set_levels) ~= 78
    error('Expected 78 unique sets, found %d', numel(set_levels));
end

u(:,3) = set_idx;  % now 1..78

c_prc = tapas_hgf_binary_config();
c_obs=tapas_memory_recall_linear_obs_config();
output_memory_hgf(s) = tapas_fitModel(y, u, c_prc, c_obs);
tapas_hgf_binary_condhalluc_plotTraj(output_memory_hgf(s))


% Group by similarity level
target_trials = subRaw.Old_new == 1 & subRaw.Similarity == 1.0;
f1_trials = subRaw.Old_new == 0 & subRaw.Similarity == 0.75;
f2_trials = subRaw.Old_new == 0 & subRaw.Similarity == 0.5;
f3_trials = subRaw.Old_new == 0 & subRaw.Similarity == 0.25;

% Compare to actual behavior
fprintf('\nActual Behavior (P("old")):\n');
fprintf('Targets (1.0):  %.3f\n', mean(subRaw.Response(target_trials)));
fprintf('F1 (0.75):      %.3f\n', mean(subRaw.Response(f1_trials)));
fprintf('F2 (0.50):      %.3f\n', mean(subRaw.Response(f2_trials)));
fprintf('F3 (0.25):      %.3f\n', mean(subRaw.Response(f3_trials)));

end

% look at gamma and alpha values across subs 
for i = 1:numel(uniqueSubs);gamma(i)=output_memory_hgf(i).p_obs.gamma(end);end
for i = 1:numel(uniqueSubs);alpha(i)=output_memory_hgf(i).p_obs.alpha(end);end
[h_alpha,p_alpha,ci_alpha,s_alpha]=ttest(alpha);
[h_gamma,p_gamma,ci_gamma,s_gamma]=ttest(gamma);

save('hgf_all_subjs_Exp1_Exp2.mat','output_memory_hgf')



%% compare outputs
validation = validate_similarity_effects(output_memory_hgf, longT, 'output_memory_hgf');

results_with_set = analyze_subsequent_trial_parameters(output_memory_hgf, longT);
