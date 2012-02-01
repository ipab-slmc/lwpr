function [rf,transient_multiplier] = lwpr_x_update_distance_metric(model,rf,w,dwdq,ddwdqdq,e_cv,e,xn)
% [rf,transient_multiplier] = lwpr_x_update_distance_metric(model,rf,
%                                   w,dwdq,ddwdqdq,e_cv,e,xn)
%
% Performs an update to a receptive fields distance metric given the current
% (cross-validated and usual) prediction error.
%
% INPUT
%  model    LWPR model structure
%  rf       Receptive field structure
%  w        Activation of the receptive field
%  dwdq     Derivative of w with respect to squared distance q
%  ddwdqdq  Second derivative of w with respect to q
%  e_cv     Cross-validated prediction error
%  e        Usual prediction error
%  xn       Normalised input vector
%
% OUTPUT
%  rf       The updated receptive field
%  transient_multiplier   Damping factor for preventing too large updates


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


% update the distance metric
penalty   = model.penalty/length(xn); % normalize penality w.r.t. number of inputs
meta      = model.meta;
meta_rate = model.meta_rate;
diag_only = model.diag_only;

% an indicator vector in how far individual projections are trustworthy
% based on how much data the projection has been trained on
derivative_ok = rf.n_data > 0.1./(1.-rf.lambda);
if ~derivative_ok(1),
   transient_multiplier = 0;
   return;
end

% useful pre-computations: they need to come before the updates
s                    = rf.s;
e2                   = e^2;
e_cv2                = e_cv^2;
h                    = w*sum(s.^2./rf.SSs2.*derivative_ok);
W                    = rf.sum_w(1)*rf.lambda(1) + w;
E                    = rf.sum_e_cv2(end);
transient_multiplier = min(1,(rf.sum_e2/(E+1.e-10))^4); % this is a numerical safety heuristic

% the derivative dJ1/dw
Ps    = s./rf.SSs2.*derivative_ok;  % zero the terms with insufficient data support
Pse   = Ps*e;
dJ1dw = -E/W^2 + 1/W*(e_cv2 - sum(sum((2*Pse).*rf.H)) - sum((2*Ps.^2).*rf.r));

% the derivatives dw/dM and dJ2/dM
[dwdM,dJ2dM,dwwdMdM,dJ2J2dMdM] = lwpr_x_dist_derivatives(w,dwdq,ddwdqdq,rf,xn-rf.c,diag_only,penalty,meta);

% the final derivative becomes (note this is upper triangular)
dJdM = dwdM*dJ1dw + w/W*dJ2dM;

% the second derivative if meta learning is required, and meta learning update
if (meta)   

   % second derivatives
   dJ1J1dwdw = -e_cv2/W^2 - 2/W*sum(sum((-Pse/W -2*Ps*(s'*Pse)).*rf.H)) + 2/W*e2*h/w - ...
     1/W^2*(e_cv2-2*sum(sum(Pse.*rf.H))) + E/W^3;
          
   dJJdMdM = (dwwdMdM*dJ1dw + dwdM.^2*dJ1J1dwdw) + w/W*dJ2J2dMdM;

   % update the learning rates
   aux = meta_rate * transient_multiplier * (dJdM.*rf.h);

   % limit the update rate
   ind = find(abs(aux) > 0.1);
   if (~isempty(ind)),
     aux(ind) = 0.1*sign(aux(ind));
   end
   rf.b = rf.b - aux;

   % prevent numerical overflow
   ind = find(abs(rf.b) > 10);
   if (~isempty(ind)),
     rf.b(ind) = 10*sign(rf.b(ind));
   end

   rf.alpha = exp(rf.b);

   aux = 1 - (rf.alpha.*dJJdMdM) * transient_multiplier ;
   ind = find(aux < 0);
   if (~isempty(ind)),
     aux(ind) = 0;
   end

   rf.h = rf.h.*aux - (rf.alpha.*dJdM) * transient_multiplier;

end

% update the distance metric, use some caution for too large gradients
maxM = max(max(abs(rf.M)));
delta_M = rf.alpha.*dJdM*transient_multiplier;
ind = find(delta_M > 0.1*maxM);
if (~isempty(ind)),
   rf.alpha(ind) = rf.alpha(ind)/2;
   delta_M(ind) = 0;
   % disp('Reduced learning rate');
end
%rf.M = rf.M - rf.alpha.*dJdM*transient_multiplier;
rf.M = rf.M - delta_M;
rf.D = rf.M'*rf.M;

% update sufficient statistics: note this must come after the updates and
% is conditioned on that sufficient samples contributed to the derivative
H = rf.lambda.*rf.H + (w/(1-h))*s*e_cv'*transient_multiplier;
r = rf.lambda.*rf.r + (w^2*e_cv2/(1-h))*(s.^2)*transient_multiplier;
rf.H = derivative_ok.*H + (1-derivative_ok).*rf.H;
rf.r = derivative_ok.*r + (1-derivative_ok).*rf.r;
