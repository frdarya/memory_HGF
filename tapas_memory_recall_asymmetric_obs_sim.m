function y = tapas_memory_recall_asymmetric_obs_sim(r, infStates, p)

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
kappa = p(5);

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
bias = NaN(n_trials,8);

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
        %         [prev_response, prev_similarity, correction_bias, presentation_count] = item_history(current_item);
        d = item_history(current_item);

        prev_response=d(1); prev_similarity=d(2); correction_bias=d(3); presentation_count=d(4);

        % Calculate error confidence (Power law model)
        if prev_response == 0 % new --
            % target (sim=1)
            % f1 (0.5)
            % f2 (sim = 0.3)
            % f3 (0.1)
            error_confidence = prev_similarity.^(1./kappa);
        else
            error_confidence = (1 - prev_similarity).^kappa;
            % f3 = 0.9.^kappa --> high error 

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
        if prev_response == 0 %new
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
        prior_shift = x(t) - current_mu1hat;
        % positive -> old; negative -> new?
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
        bias(t,:) = [r.u(t,3),presentation_count,current_similarity,...
            new_correction_bias,prev_response,prev_similarity,prior_shift, correction_strength];

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
        bias(t,:) = NaN(1,8);

        % Initialize item history
        item_history(current_item) = [current_response, current_similarity, 0.5, 1];
    end
end

%%%%% NEED TO ALSO SAVE Y AND X(T) TO PLOT DATA LIKE BEHAVIOURAL RESULTS

bias_clean = bias(~isnan(bias(:,1)),:);
model_resps = [x, y_sim, prob, r.u];
% Write bias_clean to CSV (append mode)
output_file_bias = fullfile(pwd, 'bias_clean_output_targ8.csv');
output_file_resps = fullfile(pwd, 'simulated_states_resp_targ8.csv');

% Add headers only if file doesn't exist yet
if ~isfile(output_file_bias)
    fid = fopen(output_file_bias, 'w');
    fprintf(fid, 'setID,presentation_count,sim,bias,prev_response,prev_sim,prior_shift,correction_strength\n');
    fclose(fid);
end
if ~isfile(output_file_resps)
    fid = fopen(output_file_resps, 'w');
    fprintf(fid, 'x,y_sim,prob,old_new,similarity,setID\n');
    fclose(fid);
end

% Append data
writematrix(bias_clean, output_file_bias, 'WriteMode', 'append');
writematrix(model_resps, output_file_resps, 'WriteMode', 'append');

y = y_sim;

return;
