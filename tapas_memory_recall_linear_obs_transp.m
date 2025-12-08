function [pvec, pstruct] = tapas_memory_recall_linear_obs_transp(r, ptrans)

% Transforms observation parameters from their native space to the space they are estimated in
%
% This function transforms parameters between:
% 1. Estimation space (unconstrained, suitable for optimization)
% 2. Native space (constrained, psychologically meaningful)
%
% Parameters:
% be: (0, +∞) in native space ↔ (-∞, +∞) in estimation space via log
% nu: (0, +∞) in native space ↔ (-∞, +∞) in estimation space via log  
% gamma: (0, 1) in native space ↔ (-∞, +∞) in estimation space via logit
% alpha: (0, 1) in native space ↔ (-∞, +∞) in estimation space via logit
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <https://urldefense.com/v3/__http://www.gnu.org/licenses/__;!!PDiH4ENfjr2_Jw!CM7g5E9fCCRXw29h2rs1__4UWel5Euln3Cc6farw1RO1ZVEF-Yu28SdZRq3J8K3Dkkk6tf_GFZtzecVUUVt0y1GUgdGZuBgIggqEygk$ [gnu[.]org]>.

% Check if we're transforming TO native space or FROM native space
% Based on typical HGF toolbox convention: 
% - Input ptrans is in estimation space
% - We transform TO native space for the observation function

% Initialize output vectors
pvec    = NaN(1, length(ptrans));
pstruct = struct;

% ---------------------------------------------------------
% 1. Transform be (inverse decision temperature)
% ---------------------------------------------------------
% Native space: (0, +∞) - must be positive
% Estimation space: (-∞, +∞) - unconstrained via log
pvec(1) = exp(ptrans(1));
pstruct.be = pvec(1);

% ---------------------------------------------------------
% 2. Transform nu (prior weighting parameter)
% ---------------------------------------------------------
% Native space: (0, +∞) - must be positive
% Estimation space: (-∞, +∞) - unconstrained via log
pvec(2) = exp(ptrans(2));
pstruct.nu = pvec(2);

% ---------------------------------------------------------
% 3. Transform gamma (error correction learning rate)
% ---------------------------------------------------------
% Native space: (0, 1) - probability/bounded rate
% Estimation space: (-∞, +∞) - unconstrained via logit
% logit(p) = log(p/(1-p))
% Inverse: p = 1/(1+exp(-x))
pvec(3)    = tapas_sgm(ptrans(3), 1);   % gamma
pstruct.gamma = pvec(3);

% ---------------------------------------------------------
% 4. Transform alpha (item memory update rate)
% ---------------------------------------------------------
% Native space: (0, 1) - probability/bounded rate
% Estimation space: (-∞, +∞) - unconstrained via logit
pvec(4)    = tapas_sgm(ptrans(4), 1);   % alpha
pstruct.alpha = pvec(4);

return;
