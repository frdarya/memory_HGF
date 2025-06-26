function [pvec, pstruct] = tapas_similarity_sets_interaction_transp(r, ptrans)
% Transform parameters

pvec = NaN(1,length(ptrans));
pvec(1) = exp(ptrans(1));  % ze (decision noise)
pvec(2) = ptrans(2);       % beta_interaction (keep as is)

% Parameter structure
pstruct = struct();
pstruct.ze = pvec(1);
pstruct.beta_interaction = pvec(2);

return;