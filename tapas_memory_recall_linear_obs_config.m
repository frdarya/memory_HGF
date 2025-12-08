function c = tapas_memory_recall_linear_obs_config
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Contains the configuration for the response model used to analyze data
% from the memory recall experiment with item-specific error correction.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The rationale for this model is as follows:
% (ALSO SEE COMMENTS IN THE OBSERVATION MODEL FUNCTION)
%
% 1. Participants combine prior expectations with similarity evidence in a Bayesian manner.
% 2. When items are repeated, participants use similarity from previous presentations
%    to infer whether they made an error.
% 3. The model tests the hypothesis that participants correct perceived errors on
%    subsequent presentations of the same item.
%
% Model parameters:
% 1. be: Inverse decision temperature (response determinism)
% 2. nu: Prior weighting (expectations vs. evidence)
% 3. gamma: Error correction learning rate 
% 4. alpha: Item memory update rate
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <https://urldefense.com/v3/__http://www.gnu.org/licenses/__;!!PDiH4ENfjr2_Jw!CM7g5E9fCCRXw29h2rs1__4UWel5Euln3Cc6farw1RO1ZVEF-Yu28SdZRq3J8K3Dkkk6tf_GFZtzecVUUVt0y1GUgdGZuBgIggqEygk$ [gnu[.]org]>.

% Config structure
c = struct;

% Model name
c.model = 'tapas_memory_recall_linear_obs';

% Sufficient statistics of Gaussian parameter priors

% ---------------------------------------------------------
% 1. Beta (be) - Inverse decision temperature
% ---------------------------------------------------------
% Interpretation: How deterministically responses follow beliefs
% High be = low decision noise, deterministic responses
% Low be = high decision noise, more random responses
% Prior: Centered at moderate determinism with some variability
c.logbemu = log(2);        % exp(0.69) = 2.0 (moderate determinism)
% In the hallucination paper they had: c.logbemu = log(48);
c.logbesa = 1;

% ---------------------------------------------------------
% 2. Nu (nu) - Prior weighting parameter  
% ---------------------------------------------------------
% Interpretation: Balance between expectations and evidence
% High nu = priors dominate (expectation-driven)
% Low nu = evidence dominates (detail-oriented)
% Prior: Centered at balanced weighting (nu = 1)
c.lognumu = log(1);        % exp(0) = 1.0 (balanced)
c.lognusa = 0.5;           % Moderate variability

% In the hallucination paper they had:
% c.lognumu = log(1/4);  % Changed 12/6/16 to tighten fit.
% c.lognusa = 1/4;

% ---------------------------------------------------------
% 3. Gamma (gamma) - Error correction learning rate
% ---------------------------------------------------------
% Interpretation: How much participants adjust based on perceived errors
% High gamma = strong error correction (hypothesis predicts this!)
% Low gamma = minimal error correction  
% Prior: Transformed via sigmoid, so prior on native space is normal
% We set prior mean to 0 (sigmoid(0) = 0.5) = moderate learning
c.gammamu = 0;             % sigmoid(0) = 0.5
c.gammasa = 1;             % Reasonable variability

% ---------------------------------------------------------
% 4. Alpha (alpha) - Item memory update rate
% ---------------------------------------------------------
% Interpretation: How quickly item-specific knowledge updates
% High alpha = rapid item learning
% Low alpha = slow item learning
% Prior: Transformed via sigmoid, centered at moderate updating
c.alphamu = 0;             % sigmoid(0) = 0.5
c.alphasa = 1;             % Reasonable variability

% ---------------------------------------------------------
% Gather prior settings in vectors
% Order corresponds to: [be, nu, gamma, alpha]
% ---------------------------------------------------------
c.priormus = [
    c.logbemu, ...    % be (log space)
    c.lognumu, ...    % nu (log space)  
    c.gammamu, ...    % gamma (native space for sigmoid)
    c.alphamu, ...    % alpha (native space for sigmoid)
];

c.priorsas = [
    c.logbesa, ...    % be
    c.lognusa, ...    % nu
    c.gammasa, ...    % gamma
    c.alphasa, ...    % alpha
];

% Model filehandle
c.obs_fun = @tapas_memory_recall_linear_obs;

% Handle to function that transforms observation parameters to their native space
% from the space they are estimated in
c.transp_obs_fun = @tapas_memory_recall_linear_obs_transp;

return;

