function [logp, yhat, res] = tapas_similarity_exposure_precomputed(r, infStates, ptrans)
% Observation model with pre-computed exposure weights
% Expects input u with 4 columns: [old_new, similarity, set_id, weighted_exposure]

% Use predictions or posteriors
pop = 1;
if isfield(r.c_obs, 'predorpost') && r.c_obs.predorpost == 2
    pop = 3;
end

% Transform parameters
ze = exp(ptrans(1));
beta_exposure = ptrans(2);

% Initialize outputs
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 4
    error('Inputs must have 4 columns: [old_new, similarity, set_id, weighted_exposure]');
end

% Get data and remove irregular trials
mu1 = infStates(:,1,pop);
mu1(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];
similarity = r.u(:,2);
similarity(r.irr) = [];
weighted_exposure = r.u(:,4);  % Pre-computed exposure weights
weighted_exposure(r.irr) = [];

% Vectorized evidence scaling with exposure weighting
exposure_effect = beta_exposure * weighted_exposure;
scaled_evidence = mu1 .* (similarity .^ 3) .* (1 + exposure_effect);

% Response probabilities (vectorized)
p_old = 1 ./ (1 + exp(-ze .* scaled_evidence));

% Calculate outputs
reg = ~ismember(1:n, r.irr);
logp(reg) = y .* log(p_old) + (1-y) .* log(1-p_old);
yhat(reg) = p_old;
res(reg) = y - p_old;

return;