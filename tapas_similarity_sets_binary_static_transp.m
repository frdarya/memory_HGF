function [pvec, pstruct] = tapas_similarity_sets_binary_static_transp(r, ptrans)
% Transform parameters (only one parameter for static model)

% Transform from log space
pvec = NaN(1,length(ptrans));
pvec(1) = exp(ptrans(1)); % ze (decision noise)

% Create parameter structure
pstruct = struct();
pstruct.ze = pvec(1);

return;