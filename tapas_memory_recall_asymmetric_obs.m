function [logp, yhat, res] = tapas_memory_recall_asymmetric_obs(r, infStates, ptrans)

% Memory recall model with item-specific ERROR CORRECTION. The model uses a
% simple linear error detection. This is the most parsimonious approach,
% direct linear mapping from similarity to error confidence
%
% Hypothesis: subjects correct perceived mistakes on subsequent
% presentations
%
% Calculates the log-probability of response y=1 under the unit-square
% sigmoid model, where the decision variable is a function of the prior
% beliefs (muhat) which is obtained as a Bayesian update using a conjugate
% beta model (as in Power et al. paper).
%
% To model the effect of a previous error (either a miss or a false alarm),
% the model assumes that prior beliefs (the muhat) about the next response
% is affected by the type of error made previously. This is implemented by
% introducing a "correction" (bias) to the prior beliefs that is a function
% of the level of similarity. That is, since there is no feedback in the
% experiment, we assume that the subjects use "similarity" as a proxy for
% correctness (i.e. as evidence for having made an error before),
% effectively becoming an internal learning signal. That is,
%
%
% The function can be easily adapted to use a different proxy (e.g.
% "confidence ratings" (CR) or "response time" (RT)), by simply adding the measured
% CR/RT as an input column and modifying the loop in the code to
% look at CR/RT rather than similarity as the "learning signal"
%
% Why bias the prior belief rather than the decision variable directly?
%
% 1- Theoretical distinction: Perception vs Decision:
%
%   In the Bayesian (Conjugate Beta model) framework we are using:
%    - mu1hat (prior): Represents the perceptual/cognitive expectation
%    before seeing evidence
%    - x (posterior): The final belief after combining prior with evidence
%    - Sigmoid response function: Maps belief x to actual decision
%   By biasing the prior (mu1hat), we're saying: "The experience of making an error
%   changes your baseline expectations about what you're likely to
%   encounter."
%
% 2- Psychological Interpretation
%
%   Biasing the prior means:
%   "Based on my previous experience with this item, I expect it to be
%   old". This affects how the current sensory evidence is interpreted. It
%   represents a genuine change in perception/expectation
%
%   Biasing the decision variable directly would mean:
%   "I see the evidence the same way, but I'll adjust my decision
%   threshold". This is more like response strategy or decision bias.
%
% 3- Our hypothesis favours prior biasing
%
%   Hypothesis: "more likely to respond 'old' for a true old object if they
%   previously falsely identified it as 'new'"
%
%   This suggests a genuine change in perception rather than just a
%   decision strategy. That is, memory is reconstructive, so previous
%   errors might actually change how we encode/retrieve the item. Also,
%   after an error, you might pay more attention to diagnostic features
%   (attention mechanism). Finnally, from a Bayesian brain hypothesis point
%   of view, priors are updated based on experience to improve future
%   inferences. This means that in the experiment, the first error makes
%   you more sensitive to the item's features, this can affect your prior
%   beliefs for future trials so that you actually perceive it differently
%   on the second presentation. Not just "I'll say old no matter what".
%
%
%%%%% DF: this assumes the error effect happens before seeing current item.
%%%% but perhaps worth considering an alternative model whereby the error
%%%% affects the decision itself rather than the expectation?
%%%% would that arbitrate between the two accounts?
%
% Formatting of the input variable:
%
%   r.u should have 3 columns:
%   Column 1: Trial type / condition identifier
%   Column 2: Similarity strength (0.25, 0.50, 0.75, 1.00)
%   Column 3: Item ID (unique identifier for each item)
%
%   Example structure:
%   r.u = [
%      1, 0.75, 101;  % Trial 1: Condition 1, similarity 75%, Item 101
%      1, 0.25, 102;  % Trial 2: Condition 1, similarity 25%, Item 102
%      0, 1.00, 101;  % Trial 3: Condition 2, similarity 100%, Item 101 (repetition!)
%      1, 0.50, 103;  % Trial 4: Condition 2, similarity 50%, Item 103
%      % ... etc.
%   ];
%
% Interpretation of parameters:
%
%   be - Inverse decision temperature (response noise)
%        How deterministically do you follow internal beliefs when making
%        decisions
%        be>1 - Deterministic/consistent responder: Responses follow
%               internal beliefs
%        be<1 - Noisy/impulsive responder: Responses are more random
%               relative to beliefs
%        Clinical relevance:
%           - Low be: Impulsivity, attention problems, response uncertainty
%           - High be: Rigid, overconfident responding
%        Mathematical role: Controls slope of sigmoid mapping from
%        (posterior) belief (x) to response (y)
%               P("OLD") = 1/(1+exp(-be(2x-1)(2y-1))), or in log scale
%               logP("OLD") = logp(reg) = -log(1 + exp(-be(2x-1)(2y-1)))
%
%   nu - Prior weighting parameter
%        How much do you rely on your expectations vs the current evidence
%        nu>1 - Expectation-driven perceiver. Strong priors dominate
%               perception. Less influenced by current similarity evidence.
%               More false memories to similar lures. Analogous to
%               halllucinators in original Powers et al. paper
%        nu<1 - Evidence-driven perceiver. Stron reliance on current
%               similarity. Priors have little influence. Better
%               discrimination, but possibly more misses. More accurate but
%               less efficient.
%        Clinical relevance:
%           - High nu: Susceptibility to mempory illusions, confirmation
%             bias
%           - Low nu: Detail-oriented, potentially overly skeptical
%           Mathematical role: Controls learning rate in belief update
%               x = mu_prior + 1/(1+nu)(similarity-mu_prior)
%
%   alpha - Item memory update rate (or "forgetting rate")
%
%           How quickly do you update what yu have learned about specific
%           items
%           - High alpha (~1): Rapid learner. Quickly incorporates new
%             information about each item. In your context this means short
%             item-specific memory. Adapts rapidly, but may be unstable.
%           - Low alpha (~0): Slow item learner: Slowly builds item
%             knowledge across repetitions. In your context, this means
%             long item-specific memory.
%           Mathematical role: Learning rate in the exponential moving
%           average that combines old bias with new evidence:
%               new_bias = (1-alpha)*old_bias + alpha*current_evidence
%           or equvalently in the for of error-based learning:
%               new_bias = old_bias + alpha*(current_evidence - old-bias)
%
%           Rationale:
%             People don't completely overhaul their beliefs after each
%             trial. Prevents wild oscillations in beliefs. Allows old
%             evidence to gradually fade.
%
%   gamma - Error correction learning rate
%           How much do you adjust your expectations after thinking you
%           made an error
%           High gamma (~1) - Quick self-corrector. Strongly adjusts
%               expectations after perceived error. High meta-cognitive
%               awareness. Flexible, adaptive learning.
%           Low gamma (~0) - Stubborn/perseverative. Ignoores potential
%               errors. Sticks with initial responses. Low meta-cognitive
%               sensitivity. Inflexible, but consistent.
%           Zero gamma (0) - No error correction. Back to baseline model,
%               No effect of previous errors on current decisions
%           Mathematical role: Scales the error correction term for the
%           prior
%               Total_correction = gamma * correction_bias*error_confidence
%
%           Testing your hypothesis:
%               If the hypothesis is correct, you should find:
%               - Significant gamma > 0 across p[articipants
%               - Higher gamma predicts better error correction on repeated
%                 items
%               - Model with gamma is better (higher negative free energy)
%                 than model without.
%   kappa - Asymmetric error detection parameter
%           How asymmetrically do you detect misses vs. false alarms?
%           Modulates the mapping from similarity to error confidence
%
%           Mathematical form:
%               For misses (previously said "NEW" to an item):
%                   error_confidence = similarity^(1/kappa)
%               For false alarms (previously said "OLD" to an item):
%                   error_confidence = (1 - similarity)^kappa
%
%           kappa = 1: Symmetric error detection
%               error_confidence = similarity for misses
%               error_confidence = 1 - similarity for false alarms
%               Equal ability to detect both types of errors
%
%           kappa > 1: Misses more detectable than false alarms
%               Misses: High similarity items produce STRONG error signals
%                   Example: kappa = 2, similarity = 0.8
%                       miss confidence = 0.8^(1/2) = 0.894 (amplified)
%               False alarms: Low similarity items produce WEAK error signals
%                   Example: kappa = 2, similarity = 0.2
%                       false alarm confidence = (1-0.2)^2 = 0.8^2 = 0.64 (attenuated)
%               Psychological interpretation: "I'm better at realizing when
%                   I've forgotten something than when I've falsely remembered it"
%
%           kappa < 1: False alarms more detectable than misses
%               Misses: High similarity items produce WEAK error signals
%                   Example: kappa = 0.5, similarity = 0.8
%                       miss confidence = 0.8^(1/0.5) = 0.8^2 = 0.64 (attenuated)
%               False alarms: Low similarity items produce STRONG error signals
%                   Example: kappa = 0.5, similarity = 0.2
%                       false alarm confidence = (1-0.2)^0.5 = 0.8^0.5 = 0.894 (amplified)
%               Psychological interpretation: "I'm better at realizing when
%                   I've imagined something than when I've missed something real"
%
%           Mathematical role: Shapes the error confidence signal before it
%           is combined with correction bias and scaled by gamma
%               error_confidence = f(similarity, kappa, response_type)
%               Total_correction = gamma * correction_bias * error_confidence
%
%           Connection to your observed correlations:
%               You found: gamma correlates positively with hit rate,
%               negatively with correct rejection rate.
%
%               If kappa > 1 (misses more detectable):
%                   - High-gamma individuals strongly correct misses
%                   - This improves hit rates on repeated items
%                   - False alarms are weakly detected, so not corrected
%                   - Correct rejection rates may even decrease due to
%                     overall liberalization from miss correction
%
%               This provides a parsimonious explanation: a single learning
%               rate (gamma) combined with asymmetric error detection (kappa > 1)
%               produces the pattern you observe.
%
%           Testing hypotheses:
%
%               - If estimated kappa > 1 across participants: Supports
%                 asymmetric error detection interpretation. People are
%                 better at detecting misses, so high gamma improves hits
%                 but not correct rejections.
%
%               - If kappa ≈ 1: Suggests symmetric error detection. Your
%                 correlation pattern would then reflect response bias
%                 shift rather than asymmetric detection.
%
%               - Individual differences: Does kappa correlate with
%                 metacognitive ability or behavioural measures?
%
%               - If kappa correlates with gamma: Suggests meta-cognitive
%                 awareness and asymmetry are related
%
%               - Model comparison (if you decide to go for it): Does model
%                 with free kappa fit better than model with kappa fixed to
%                 1 (symmetric)? This tests whether asymmetry is necessary
%                 to explain your data. As I said, this might not be
%                 necessary, but thinking again about it, when reframed as
%                 a model comparisson (change in structure), rather than a
%                 hypothesis test (null: kappa = 1), you are actually
%                 evaluating whether the added complexity of a non-linear
%                 model compensates for the gain in explanatory power.
%
%           Notes:
%
%           kappa works in concert with gamma. Gamma determines HOW MUCH
%           you correct; kappa determines WHAT you correct (misses, false
%           alarms, or both equally). Your finding that gamma correlates
%           positively with hits but negatively with correct rejections
%           suggests kappa > 1 is the appropriate interpretation.
%
%           Relationship to your correlation pattern: The positive
%           correlation with hit rates suggests high-gamma individuals
%           effectively correct misses (because misses are detectable when
%           kappa > 1). The negative correlation with correct rejections
%           suggests high-gamma individuals do NOT effectively correct
%           false alarms (because false alarms are less detectable when
%           kappa > 1). Thus, kappa > 1 provides a coherent explanation for
%           the observed pattern without requiring separate gamma
%           parameters for each error type.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <https://urldefense.com/v3/__http://www.gnu.org/licenses/__;!!PDiH4ENfjr2_Jw!Gb4QkgA7OV_3cH8EpcTYbzZ60fDr6aCVm4gquuwFee8FzeoCaQdtU0u41o_CaOl0nLgrtjoXd4jrisoKBnu6ClbBLW66M748wCVCggA$ [gnu[.]org]>.

% Transform all parameters to their native spaces
be = exp(ptrans(1));                % Inverse decision parameter
nu = exp(ptrans(2));                % Prior weighting (precision) parameter
gamma = tapas_sgm(ptrans(3), 1);    % Error correction learning rate (0-1)
alpha = tapas_sgm(ptrans(4), 1);    % Item memory update learning rate
kappa = exp(ptrans(5));
% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 3
    error('Inputs incompatible: expected trial_type, similarity, item_id in r.u')
end

% Extract experimental variables
similarity = r.u(:,2);
item_id = r.u(:,3);

% Weed irregular trials out
mu1hat = infStates(:,1,1);
mu1hat(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];
similarity(r.irr) = [];
item_id(r.irr) = [];
old_new_status = r.u(:,1);  % 1=Target (old), 0=Foil (new)
old_new_status(r.irr) = [];
% Initialise container
item_history = containers.Map('KeyType', 'double', 'ValueType', 'any');
% Stores: [previous_response, previous_similarity, correction_bias, presentation_count, old_new_status]

x = zeros(size(mu1hat));
bias = NaN(1,9);

for t = 1:length(mu1hat)

    current_item = item_id(t);
    current_similarity = similarity(t);
    current_mu1hat = mu1hat(t);
    curret_old_new = old_new_status(t);
    current_resp = y(t);

    if isKey(item_history, current_item)
        d = item_history(current_item);
        prev_response=d(1); prev_similarity=d(2); correction_bias=d(3); presentation_count=d(4); prev_old_new=d(5);
        %[prev_response, prev_similarity, correction_bias, presentation_count, prev_old_new] = item_history(current_item);

        % Calculate error confidence (Power law model)
        if prev_response == 0 % new
            error_confidence = prev_similarity.^(1./kappa); % CHECK - should we use current similarity too?
        else
            error_confidence = (1 - prev_similarity).^kappa;

        end
        error_confidence = max(0.05, min(0.95, error_confidence));

        % PROPER PROBABILITY COMBINATION IN LOG-ODDS SPACE
        % Convert prior to log-odds
        log_odds_prior = log(current_mu1hat / (1 - current_mu1hat));

        % Calculate correction in log-odds space
        % correction_strength combines bias and confidence
        correction_strength = correction_bias * error_confidence; 
        % cs is usually < 0.5  (because correction bias init 0.3 multiplied by 0.05-0.95)
        % --> neg values of correction strength 
        % this accidentaly fits the behaviour new ->new, old-> old pattern
        % meaning error confidence plays a part in determining the
        % 'old/new' direction
        correction_strength = max(0.01, min(0.99, correction_strength));
        log_odds_correction = log(correction_strength / (1 - correction_strength));
        % this is "how strongly should I correct" : 
        % When cs = 0.5 → here we multiply by 0
        % When cs = 0.9 → here we multiply by 2.2
        % When cs = 0.1 → here we multiply by −2.2

        % Apply correction with learning rate gamma
        if prev_response == 0 % new
            log_odds_adjusted = log_odds_prior + gamma * log_odds_correction;%DF: the plus here indicates "toward
        else
            log_odds_adjusted = log_odds_prior - gamma * log_odds_correction;
        end
        

%         %%%%% DF start: or separate the two?
%         % magnitude only: bounded positive effect
%         correction_mag = gamma * correction_bias * error_confidence; i.e.
%         correction_mag = gamma * correction_strength
%         if prev_response == 0
%             % previous NEW response
%             % reinforce NEW / reduce OLD tendency on related next item
%             direction = -1;
% 
%         elseif prev_response == 1
%             % previous OLD response
%             % reinforce OLD / increase OLD tendency on related next item
%             direction = +1;
%         end
%         % and then correct by direct
%         log_odds_adjusted = log_odds_prior + direction * correction_mag;
%         %%%%% DF over

        % Convert back to probability
        adjusted_prior = exp(log_odds_adjusted) / (1 + exp(log_odds_adjusted));
        adjusted_prior = max(0.01, min(0.99, adjusted_prior));


        % Standard belief update
        x(t) = adjusted_prior + 1/(1 + nu) * (current_similarity - adjusted_prior);
        
        prior_shift = x(t) - current_mu1hat;
        % positive -> old; negative -> new
        
        % UPDATE BIAS using proper learning rate
        new_correction_bias = (1-alpha) * correction_bias + alpha * error_confidence;
        new_correction_bias = max(0.01, min(0.99, new_correction_bias));

        % we update the correct bias everytime we encounter an item from
        % the same set, and the bias is adjusted based on the similarity
        % level (error confidence)
        item_history(current_item) = [y(t), current_similarity,...
            new_correction_bias, presentation_count + 1, curret_old_new];

        %% extract bias to compare sets with Target - F3- F2- F1 vs Target - F1 - F3 -F2
        bias(t,:) = [r.u(t,3),presentation_count, current_resp,current_similarity,...
            new_correction_bias,prev_response,prev_similarity,prior_shift, correction_strength];

    else

        % FIRST PRESENTATION - no correction yet
        x(t) = current_mu1hat + 1/(1 + nu) * (current_similarity - current_mu1hat);
        bias(t,:) = NaN(1,9);
        % Initialize with current response and medium correction bias
        item_history(current_item) = [y(t), current_similarity, 0.3, 1, curret_old_new];


    end

end
bias_clean = bias(~isnan(bias(:,1)),:);
% Calculate log-probabilities
reg = ~ismember(1:n, r.irr);
logp(reg) = -log(1 + exp(-be .* (2.*x - 1) .* (2.*y - 1)));
yhat(reg) = x;
res(reg) = (y - x) ./ sqrt(x .* (1 - x));

return
