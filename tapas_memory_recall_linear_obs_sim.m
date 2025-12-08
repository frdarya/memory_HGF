function y = tapas_memory_recall_linear_obs_sim(r, infStates, p)

% Simulates responses according to the condhalluc_obs model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <https://urldefense.com/v3/__http://www.gnu.org/licenses/__;!!PDiH4ENfjr2_Jw!CM7g5E9fCCRXw29h2rs1__4UWel5Euln3Cc6farw1RO1ZVEF-Yu28SdZRq3J8K3Dkkk6tf_GFZtzecVUUVt0y1GUgdGZuBgIggqEygk$ [gnu[.]org]>.

% Get parameters
be = p(1);
nu = p(2);
gamma = p(3);
alpha = p(4);

% Check input format
if size(r.u,2) < 3
    error('Inputs incompatible: expected trial_type, similarity, item_id in r.u')
end

% Prediction trajectory from perceptual model
mu1hat = infStates(:,1,1);

% Extract experimental variables
similarity = r.u(:,2);    % Similarity strength (0.25, 0.50, 0.75, 1.00)
item_id = r.u(:,3);       % Unique identifier for each item


% Initialise container to track item-specific learning
item_history = containers.Map('KeyType', 'double', 'ValueType', 'any');
% Stores: [previous_response, previous_similarity, correction_bias, presentation_count]

% Initialize arrays
n_trials = length(mu1hat);
x = zeros(n_trials, 1);  % Beliefs
prob = zeros(n_trials, 1); % Response probabilities
y_sim = zeros(n_trials, 1); % Simulated responses

% Initialize random number generator
rng('shuffle');

% Simulate trial by trial
for t = 1:n_trials
    current_item = item_id(t);
    current_similarity = similarity(t);
    current_mu1hat = mu1hat(t);
    
    if isKey(item_history, current_item)
        % ---------------------------------------------------------
        % ITEM HAS BEEN SEEN BEFORE - APPLY ERROR CORRECTION
        % ---------------------------------------------------------
        [prev_response, prev_similarity, correction_bias, presentation_count] = item_history(current_item);
        
        % Calculate error confidence (linear approach)
        if prev_response == 0 
            % Previously said "NEW" - error confidence increases with similarity
            error_confidence = prev_similarity;
        else 
            % Previously said "OLD" - error confidence increases as similarity decreases
            error_confidence = 1 - prev_similarity;
        end
        error_confidence = max(0.05, min(0.95, error_confidence));
        
        % PROPER PROBABILITY COMBINATION IN LOG-ODDS SPACE
        % Convert prior to log-odds
        log_odds_prior = log(current_mu1hat / (1 - current_mu1hat));
        
        % Calculate correction strength
        correction_strength = correction_bias * error_confidence;
        correction_strength = max(0.01, min(0.99, correction_strength));
        log_odds_correction = log(correction_strength / (1 - correction_strength));
        
        % Apply correction with learning rate gamma
        if prev_response == 0 
            % Increase tendency to say "OLD"
            log_odds_adjusted = log_odds_prior + gamma * log_odds_correction;
        else 
            % Decrease tendency to say "OLD"
            log_odds_adjusted = log_odds_prior - gamma * log_odds_correction;
        end
        
        % Convert back to probability space
        adjusted_prior = exp(log_odds_adjusted) / (1 + exp(log_odds_adjusted));
        adjusted_prior = max(0.01, min(0.99, adjusted_prior));
        
        % Standard Bayesian belief update with adjusted prior
        x(t) = adjusted_prior + 1/(1 + nu) * (current_similarity - adjusted_prior);
        
        % Store simulated response for this trial (needed for updating history)
        prob(t) = tapas_sgm(be * (2*x(t) - 1), 1);
        current_response = binornd(1, prob(t));
        y_sim(t) = current_response;
        
        % UPDATE CORRECTION BIAS using exponential moving average
        prediction_error = error_confidence - correction_bias;
        new_correction_bias = correction_bias + alpha * prediction_error;
        new_correction_bias = max(0.01, min(0.99, new_correction_bias));
        
        % Store updated history
        item_history(current_item) = [current_response, current_similarity, new_correction_bias, presentation_count + 1];
        
    else
        % ---------------------------------------------------------
        % FIRST PRESENTATION OF ITEM - NO ERROR CORRECTION
        % ---------------------------------------------------------
        % Standard belief update
        x(t) = current_mu1hat + 1/(1 + nu) * (current_similarity - current_mu1hat);
        
        % Apply logistic sigmoid to get response probability
        prob(t) = tapas_sgm(be * (2*x(t) - 1), 1);
        
        % Simulate response
        current_response = binornd(1, prob(t));
        y_sim(t) = current_response;
        
        % Initialize item history
        item_history(current_item) = [current_response, current_similarity, 0.5, 1];
    end
end

y = y_sim;

return;
