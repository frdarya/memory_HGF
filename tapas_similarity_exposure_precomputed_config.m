function c = tapas_similarity_exposure_precomputed_config
% Configuration for pre-computed exposure model

c = struct;
c.predorpost = 1;
c.model = 'tapas_similarity_exposure_precomputed';

% Priors
c.logze_mu = log(48);
c.logze_sa = 1;
c.beta_exposure_mu = 0;
c.beta_exposure_sa = 1;

c.priormus = [c.logze_mu; c.beta_exposure_mu];
c.priorsas = [c.logze_sa; c.beta_exposure_sa];

c.obs_fun = @tapas_similarity_exposure_precomputed;
c.transp_obs_fun = @tapas_similarity_exposure_precomputed_transp;

return;