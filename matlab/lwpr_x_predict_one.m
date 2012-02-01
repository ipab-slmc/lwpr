function [yp, conf, max_w] = lwpr_x_predict_one(model,dim,x,cutoff)
% [yp, conf, max_w] = lwpr_x_predict_one(model,dim,x,cutoff)
%
% Predict the output of an LWPR model along a specific output dimension,
% and also return a confidence bound and the maximum activation
%
% INPUT
%  model    LWPR model structure
%  dim      Output dimension ( 1 <= dim <= model.nOut )
%  x        Input vector (already normalised)
%  cutoff   Threshold for ignoring receptive fields with low activation
%
% OUTPUT
%  yp       Predicted output
%  conf     Confidence bound for yp
%  max_w    Maximum activation over all receptive fields 




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




% maintain the maximal activation
yp = 0;
sum_w = 0;
max_w = 0;

if nargout > 1

   conf = 0;
   sum_wyy = 0;

   for i=1:length(model.sub(dim).rfs),
      RFS = model.sub(dim).rfs(i);

      % compute the weight

      % subtract the center
      xc = x - RFS.c;
      w = lwpr_x_kernel(model.kernel,xc'*RFS.D*xc);

      max_w = max([max_w,w]);

      % model.sub(dim).rfs(i).w = w; % composite-control extension ???

      % only predict if activation is high enough
      if (w > cutoff & RFS.trustworthy)

         % the mean zero input
         xmz = x - RFS.mean_x;

         % compute the projected inputs
         s = lwpr_x_compute_projection(xmz,RFS.U,RFS.P);
         
         %%%%%%%%% ADDITIONAL SAFETY HEURISTIC: 
         %%%%%%%%% DO NOT INCLUDE NEW PLS DIMENSIONS
         if RFS.n_data(end) <= model.nIn*2
            s(end)=0;  % -> no contribution to prediction
         end

         sigma2 = RFS.sum_e_cv2(end)/(RFS.sum_w(end) - RFS.SSp)*(1+w*sum(s.^2./RFS.SSs2));

         yp_i = (RFS.beta'*s + RFS.beta0);

         conf = conf + w*sigma2;

         sum_wyy = sum_wyy + w*yp_i^2;

         % prediction of i-th model
         yp = yp + w*yp_i;
         sum_w = sum_w + w;
      end % if (w > cutoff)
   end

   % the final prediction
   if (sum_w > 0)
      yp = yp/sum_w;
      conf = sqrt(abs(conf + sum_wyy - sum_w*yp*yp))/sum_w;   
   else
      yp = 0;
      conf = Inf;
   end 
   
else
   % Same as above, but without confidence bounds and max_w calculations
   for i=1:length(model.sub(dim).rfs),
      RFS = model.sub(dim).rfs(i);

      % compute the weight

      % subtract the center
      xc = x - RFS.c;
      w = lwpr_x_kernel(model.kernel,xc'*RFS.D*xc);

      max_w = max([max_w,w]);

      % model.sub(dim).rfs(i).w = w; % composite-control extension ???

      % only predict if activation is high enough
      if (w > cutoff & RFS.trustworthy)

         % the mean zero input
         xmz = x - RFS.mean_x;

         % compute the projected inputs
         s = lwpr_x_compute_projection(xmz,RFS.U,RFS.P);
         
         %%%%%%%%% ADDITIONAL SAFETY HEURISTIC: 
         %%%%%%%%% DO NOT INCLUDE NEW PLS DIMENSIONS
         if RFS.n_data(end) <= model.nIn*2
            s(end)=0;  % -> no contribution to prediction
         end

         yp_i = (RFS.beta'*s + RFS.beta0);

         % prediction of i-th model
         yp = yp + w*yp_i;
         sum_w = sum_w + w;
      end % if (w > cutoff)
   end

   % the final prediction
   if (sum_w > 0)
      yp = yp/sum_w;
   else
      yp = 0;
   end 
end
