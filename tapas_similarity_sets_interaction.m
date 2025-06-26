function [logp, yhat, res] = tapas_similarity_sets_interaction(r, infStates, ptrans)
% Observation model with similarity × set_id interaction
% Evidence = mu1 * similarity^3 * (1 + beta * set_id)

% Use predictions or posteriors
pop = 1;
if isfield(r.c_obs, 'predorpost') && r.c_obs.predorpost == 2
    pop = 3;
end

% Transform parameters
ze = exp(ptrans(1));           % Decision noise
beta_interaction = ptrans(2);   % Interaction parameter (not exp transformed)

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

% Normalize set_id to reasonable range (optional but recommended)
normalized_set_id = (set_id - mean(set_id)) / std(set_id);

% Core mechanism: similarity^3 modulated by set interaction
% Evidence = mu1 * similarity^3 * (1 + beta * normalized_set_id)
similarity_effect = similarity .^ 3;
set_modulation = 1 + beta_interaction * normalized_set_id;
scaled_evidence = mu1 .* similarity_effect .* set_modulation;

% Response probability
p_old = 1 ./ (1 + exp(-ze .* scaled_evidence));

% Calculate outputs
reg = ~ismember(1:n, r.irr);
logp(reg) = y .* log(p_old) + (1-y) .* log(1-p_old);
yhat(reg) = p_old;
res(reg) = y - p_old;

return;