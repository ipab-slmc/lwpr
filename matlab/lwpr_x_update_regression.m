function [rf,yp,e_cv,e] = lwpr_x_update_regression(rf,x,y,w)
% [rf, yp, e_cv, e] = lwpr_x_update_regression(rf, x, y, w)
%
% Update the PLS (partial least squares) regression parameters of
% a receptive field. Also returns the current prediction, as well
% as the cross-validated and usual prediction error.
%
% INPUT
%  rf    Receptive field structure
%  x     Input vector  (zero mean)
%  y     Output sample (zero mean, specific to one dimension)
%  w     Activation of the receptive field
%  
% OUTPUT
%  rf    Updated receptive field
%  yp    Current prediction (including offset beta0)
%  e_cv  Cross-validated error (prediction error BEFORE update)
%  e     Prediction error (y-yp)


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

[n_in, n_reg] = size(rf.U);

% compute the projection
[rf.s,xres] = lwpr_x_compute_projection_r(x, rf.U, rf.P);

% compute all residual errors and targets at all projection stages
yres  = rf.beta .* rf.s;
for i=2:n_reg
   yres(i) = yres(i) + yres(i-1);
end

yres  = repmat(y,n_reg,1) - yres;
e_cv  = yres;

ytarget = [y;yres(1:n_reg-1,1)];

% update the projections
%lambda_slow = 1-(1-rf.lambda)/10;
lambda_slow = 0.9 + 0.1*rf.lambda;
rf.SXresYres = rf.SXresYres .* repmat(lambda_slow',n_in,1) + w * repmat(ytarget',n_in,1).*xres;
Unorm = sqrt(sum(rf.SXresYres.^2,1))';

% additional numerical safety heuristics
if all(Unorm>1e-12)  
   rf.U = rf.SXresYres .* repmat((Unorm.^(-1))',n_in,1);
end

% update sufficient statistics for regressions
rf.SSs2 = rf.lambda.*rf.SSs2 + rf.s.^2 * w;
rf.SSYres = rf.lambda.*rf.SSYres + w*ytarget.*rf.s;
rf.SSXres = repmat(rf.lambda',n_in,1).*rf.SSXres + w * repmat(rf.s',n_in,1).* xres;

% update the regression and input reduction parameters

inv_SSs2 = rf.SSs2.^(-1);

rf.beta = rf.SSYres .* inv_SSs2;
rf.P = rf.SSXres .* repmat(inv_SSs2',n_in,1);

% update sufficient statistics for confidence bounds
rf.SSp = rf.lambda(end)*rf.SSp + w^2*sum((rf.s.^2).*inv_SSs2);

% the new predicted output after updating
rf.s = lwpr_x_compute_projection(x, rf.U, rf.P);


%%%%%%%%% ADDITIONAL SAFETY HEURISTIC: 
%%%%%%%%% DO NOT INCLUDE NEW PLS DIMENSIONS
if rf.n_data(end) <= n_in*2
   rf.s(end)=0;  % -> no contribution to prediction
end

yp = rf.beta' * rf.s;
e  = y - yp;
yp = yp + rf.beta0;


%if rf.n_data(1) > 0.1./(1.-rf.lambda(1))
if rf.n_data(1)*(1-rf.lambda(1)) > 0.1
   rf.sum_e_cv2 = rf.sum_e_cv2.*rf.lambda + w*e_cv.^2;
   % TODO: check whether "end" or "1" is right here 
   rf.sum_e2    = rf.sum_e2*rf.lambda(end) + w*e^2;
end
e_cv = e_cv(end);

% is the RF trustworthy: a simple data count
if (rf.n_data > n_in*2)
  rf.trustworthy = 1;
end

