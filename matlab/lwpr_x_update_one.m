function [sub,yp, max_w] = lwpr_x_update_one(model, dim, xn, yn)
% [sub, yp, max_w] = lwpr_x_update_one(model, dim, xn, yn)
%
% Updates the LWPR model using an input/output sample (x,y)
% along a specific output dimension (dim).
% Optionally returns the current prediction for x and the
% maximum RF activation.
%
% Input parameters
%
%     model    Valid LWPR model structure
%     dim      Output dimension to be handled
%     xn       Input vector (nIn x 1 matrix)
%     yn       Output vector (nOut x 1 matrix)
%
% Return values
%
%     sub      Updated LWPR submodel structure (=> model.sub[dim])
%     yp       Current prediction of y given x, useful for
%              tracking the training error
%     max_w    Maximum RF activation


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

wv = zeros(3,1);
iv = zeros(3,1);
yp = 0;
sum_w = 0;

sub = model.sub(dim);
numrfs = length(sub.rfs);

if numrfs==0
   sub.rfs = lwpr_x_init_rf(model,[],xn,yn);
   max_w = 0;
   return;
end

for i=1:numrfs
   RF = sub.rfs(i);
       
   % check all RFs for updating
   % wv is a vector of 3 weights, ordered [w; sec_w; max_w]
   % iv is the corresponding vector containing the RF indices
   
   xc = xn - RF.c;
   [w, dwdq, ddwdqdq] = lwpr_x_kernel(model.kernel,xc'*RF.D*xc);
   
   RF.w = w;
   wv(1) = w;
   iv(1) = i;
   [wv, ind]=sort(wv);
   iv = iv(ind);
         
   % only update if activation is high enough
   if (w > 0.001)

      % update weighted mean for xn and y, and create mean-zero variables
      [RF, xmz, ymz] = lwpr_x_update_means(RF, xn, yn, w);

      % update the regression
      [RF, yp_i, e_cv, e] = lwpr_x_update_regression(RF, xmz, ymz, w);
            
      if (RF.trustworthy)
         yp = w*yp_i + yp;
         sum_w = sum_w + w;
      end
      
      if model.update_D 
         [RF, tm] = lwpr_x_update_distance_metric(model, RF, w, dwdq, ddwdqdq, e_cv, e, xn);
      end

      % check whether a projection needs to be added

      RF = lwpr_x_check_add_projection(model, RF);

      % update simple statistical variables
      RF.sum_w  = RF.sum_w.*RF.lambda + w;
      RF.n_data = RF.n_data.*RF.lambda + 1;
      RF.lambda = model.tau_lambda * RF.lambda + model.final_lambda*(1.-model.tau_lambda);

      % incorporate updates
      sub.rfs(i) = RF;
   else 
      sub.rfs(i).w = 0;
   end % if (w > 0.001)
end
        
% do we need to add a new RF?
if (wv(3) <= model.w_gen)
   if (wv(3) > 0.1*model.w_gen & sub.rfs(iv(3)).trustworthy)
      sub.rfs(numrfs+1) = lwpr_x_init_rf(model, sub.rfs(iv(3)), xn, yn);
   else
      sub.rfs(numrfs+1) = lwpr_x_init_rf(model, [], xn, yn);
   end
end

% do we need to prune a RF? Prune the "smaller" one ("larger" D)
if (wv(2) > model.w_prune),
%   if (sum(sum(sub.rfs(iv(2)).D)) > sum(sum(sub.rfs(iv(3)).D)))
   if trace(sub.rfs(iv(2)).D) > trace(sub.rfs(iv(3)).D)
      sub.rfs(iv(2)) = [];
      disp(sprintf('Output dim. %d: Pruned #RF=%d',dim,iv(2)));
   else
      sub.rfs(iv(3)) = [];
      disp(sprintf('Output dim. %d: Pruned #RF=%d',dim,iv(3)));
   end
   sub.n_pruned = sub.n_pruned + 1;
end

% the final prediction
if (sum_w > 0),
   yp = yp/sum_w;
else
   yp = 0;
end

max_w = wv(3);

