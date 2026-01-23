function [logp, yhat, res] = tapas_condhalluc_obs2(r, infStates, ptrans)
% Calculates the log-probability of response y=1 under the unit-square sigmoid model
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

% Transform alpha and beta to their native spaces
be = exp(ptrans(1));
nu = exp(ptrans(2));

% Initialize returned log-probabilities as NaNs so that NaN is
% returned for all irregualar trials
n = size(infStates,1);
logp = NaN(n,1);
yhat = NaN(n,1);
res  = NaN(n,1);

% Check input format
if size(r.u,2) ~= 2
    error('tapas:hgf:CondHalluc:InputsIncompatible', 'Inputs incompatible with condhalluc_obs observation model. See tapas_condhalluc_obs_config.m.')
end

% Get true-positive rate corresponding to stimuli
tp = r.u(:,2);
old_new = r.u(:,1);

% Weed irregular trials out
mu1hat = infStates(:,1,1);
mu1hat(r.irr) = [];
y = r.y(:,1);
y(r.irr) = [];
tp(r.irr) = [];

% Update belief using precision-weighted prediction error
% with nu the generalized precision
% powers (original):
 x = mu1hat + 1/(1 + nu)*(tp - mu1hat);
 x_seen = 
 % add a "seen before"

% inverse nu scaling from powers
% x = mu1hat + nu/(1 + nu)*(tp - mu1hat);

% % scale belief by old_new and sim by (scaled) nu  
%this is to be used if want to simulate responses (where the truth is
%known)
% x = mu1hat .* old_new + 1/(1 + nu) * tp;

 % Calculate log-probabilities for non-irregular trials
reg = ~ismember(1:n,r.irr);
logp(reg) = -log(1+exp(-be.*(2.*x-1).*(2.*y-1))); %this looks like sfotmax?

yhat(reg) = x;
res(reg) = (y-x)./sqrt(x.*(1-x));

return;
