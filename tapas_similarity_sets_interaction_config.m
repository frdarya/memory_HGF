function c = tapas_similarity_sets_interaction_config
% Configuration for similarity × set interaction model

% Config structure
c = struct;

% Use predictions
c.predorpost = 1;

% Model name
c.model = 'tapas_similarity_sets_interaction';

% Prior for decision noise (ze)
c.logze_mu = log(48);
c.logze_sa = 1;

% Prior for interaction parameter (beta) - centered at 0
c.beta_interaction_mu = 0;    % No interaction by default
c.beta_interaction_sa = 1;    % Allow moderate deviations

% Gather priors
c.priormus = [
    c.logze_mu;
    c.beta_interaction_mu
];

c.priorsas = [
    c.logze_sa;
    c.beta_interaction_sa
];

% Function handles
c.obs_fun = @tapas_similarity_sets_interaction;
c.transp_obs_fun = @tapas_similarity_sets_interaction_transp;

return;