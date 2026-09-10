function pstruct = tapas_memory_recall_asymmetric_obs_namep(pvec)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2016 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <https://urldefense.com/v3/__http://www.gnu.org/licenses/__;!!PDiH4ENfjr2_Jw!CM7g5E9fCCRXw29h2rs1__4UWel5Euln3Cc6farw1RO1ZVEF-Yu28SdZRq3J8K3Dkkk6tf_GFZtzecVUUVt0y1GUgdGZuBgIggqEygk$ [gnu[.]org]>.

pstruct = struct;

pstruct.be = pvec(1);
pstruct.nu = pvec(2);
pstruct.gamma = pvec(3);
pstruct.alpha = pvec(4);
pstruct.kappa = pvec(5);
return;
