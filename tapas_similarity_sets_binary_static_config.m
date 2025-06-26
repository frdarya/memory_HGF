function c = tapas_similarity_sets_binary_static_config
% Configuration for similarity + static set structure (baseline)

% Config structure
c = struct;

% Use predictions
c.predorpost = 1;

% Model name
c.model = 'tapas_similarity_sets_binary_static';

% Prior for decision noise parameter (ze) - only parameter
c.logze_mu = log(48);
c.logze_sa = 1;

% Gather prior settings (only one parameter for static model)
c.priormus = [
    c.logze_mu
];

c.priorsas = [
    c.logze_sa
];

% Model function handle
c.obs_fun = @tapas_similarity_sets_binary_static;

% Transform function handle
c.transp_obs_fun = @tapas_similarity_sets_binary_static_transp;

return;