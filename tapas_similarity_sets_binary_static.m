function [logp, yhat, res] = tapas_similarity_sets_binary_static(r, infStates, ptrans)
% Baseline: Static set multipliers (no learning)
% Expects input u with 3 columns: [old_new, similarity, set_id]

% Use predictions or posteriors
pop = 1;
if isfield(r.c_obs, 'predorpost') && r.c_obs.predorpost == 2
    pop = 3;
end

% Transform parameters
ze = exp(ptrans(1));

% Initialize outputs
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 3
    error('Inputs must have 3 columns: [old_new, similarity, set_id]');
end

% Get data and remove irregular trials
mu1 = infStates(:,1,pop);
mu1(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];
similarity = r.u(:,2);
similarity(r.irr) = [];
set_id = r.u(:,3);
set_id(r.irr) = [];

% Static set multipliers (all equal for baseline)
max_set_id = max(set_id);
set_multipliers = ones(max_set_id, 1); % All sets equal

% Get multipliers for each trial
current_multipliers = set_multipliers(set_id);

% Evidence scaling: similarity^3 + set structure (but no learning yet)
scaled_evidence = mu1 .* (similarity .^ 3) .* current_multipliers;

% Response probabilities
p_old = 1 ./ (1 + exp(-ze .* scaled_evidence));

% Calculate outputs
reg = ~ismember(1:n, r.irr);
logp(reg) = y .* log(p_old) + (1-y) .* log(1-p_old);
yhat(reg) = p_old;
res(reg) = y - p_old;

return;