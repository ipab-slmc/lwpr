function [num, numstd] = lwpr_pls_stats(model)
% function [num, numstd] = lwpr_pls_stats(model)
%
% For each output dimension, this function returns the average
% number of PLS regression axes in "num", and its standard 
% deviation in "numstd".


% LWPR: A MATLAB library for incremental online learning
% Copyright (C) 2007  Stefan Klanke, Sethu Vijayakumar, Stefan Schaal
% Contact: sethu.vijayakumar@ed.ac.uk
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Library General Public
% License along with this library; if not, write to the Free
% Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.


nOut = length(model.sub);

num = zeros(nOut,1);
numstd = zeros(nOut,1);

for k=1:nOut
   nk = length(model.sub(k).rfs);
   pls = zeros(nk,1);
   corr = zeros(nk,1);
   for i=1:nk
      U = model.sub(k).rfs(i).U;
      P = model.sub(k).rfs(i).P;      
      pls(i) = size(U,2);
      corr(i) = abs(U(:,1)'*U(:,2));
      corr(i) = 1 - abs(P(:,1)'*U(:,1));
   end
   fprintf(1,'%f +/- %f\n',mean(corr),std(corr));
   num(k) = mean(pls);
   numstd(k) = std(pls);
end
