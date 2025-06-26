function [pvec, pstruct] = tapas_similarity_exposure_precomputed_transp(r, ptrans)
pvec = NaN(1,length(ptrans));
pvec(1) = exp(ptrans(1));  % ze
pvec(2) = ptrans(2);       % beta_exposure

pstruct = struct();
pstruct.ze = pvec(1);
pstruct.beta_exposure = pvec(2);
return;