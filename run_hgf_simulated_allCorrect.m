%% run hgf with simulated data with no mistakes
% "null" model of gamma


clearvars
addpath(genpath('tapas-6.1.0'))
%% 1) Load your long‐format data
longT = readtable('Exp1_and_Exp2_long_hgf.csv');
%% 2) Build the per‐set summary table
summaryAll = table();
uniqueSubs  = unique(longT.subj);
hgf_all_subjs = struct();
sim_longT = longT;
sim_longT.Response = sim_longT.Old_new; % converts response to all be correct;
sim_longT.Correctness = ones(height(sim_longT),1);
for s = 1:numel(uniqueSubs)
    subID  = uniqueSubs(s);
    subRaw = sim_longT(sim_longT.subj == subID, :);
    y = subRaw.Response;
    oldVals = [1, 0.75, 0.5, 0.25];
    newVals = [0.8, 0.5, 0.3, 0.1];

    % Copy original column
    newSimilarity = subRaw.Similarity;

    % Replace values one by one
    for k = 1:numel(oldVals)
        newSimilarity(subRaw.Similarity == oldVals(k)) = newVals(k);
    end
    u = [subRaw.Old_new,newSimilarity, subRaw.SetID]; %set needs to be dummy-coded
    %u = [subRaw.Old_new,subRaw.Similarity, subRaw.SetID]; %set needs to be dummy-coded

    raw_ids = u(:,3);
    [set_levels, ~, set_idx] = unique(raw_ids, 'stable');

    if numel(set_levels) ~= 78
        error('Expected 78 unique sets, found %d', numel(set_levels));
    end

    u(:,3) = set_idx;  % now 1..78

    c_prc = tapas_hgf_binary_config();
    c_obs=tapas_memory_recall_linear_obs_config();
    output_memory_hgf(s) = tapas_fitModel(y, u, c_prc, c_obs);
    %tapas_hgf_binary_condhalluc_plotTraj(output_memory_hgf(s))

end
for i = 1:numel(uniqueSubs);gamma(i)=output_memory_hgf(i).p_obs.gamma(end);end
for i = 1:numel(uniqueSubs);alpha(i)=output_memory_hgf(i).p_obs.alpha(end);end

%save('hgf_all_subjs_Exp1_Exp2_simulated_allCorrect.mat','output_memory_hgf')
