function [yp, dypdx, conf, dcdx ] = lwpr_x_predict_one_JcJ(model,dim,x,cutoff)
% [yp, dypdx, conf, dcdx] = lwpr_x_predict_one_JcJ(model,dim,x,cutoff)
%
% Predict the output of an LWPR model along a specific output dimension,
% the confidence interval, and also the derivatives of both quantities
% with respect to the input vector
%
% INPUT
%  model    LWPR model structure
%  dim      Output dimension ( 1 <= dim <= model.nOut )
%  x        Input vector (already normalised)
%  cutoff   Threshold for ignoring receptive fields with low activation
%
% OUTPUT
%  yp       Predicted output
%  dypdx    Derivatives of yp with respect to x
%  conf     Confidence interval
%  dcdx     Derivatives of conf with respect to x



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
sum_R = 0;

sum_dwdx = zeros(model.nIn,1);
sum_ydwdx = zeros(model.nIn,1);
sum_wdydx = zeros(model.nIn,1);

sum_dRdx = zeros(model.nIn,1);

for i=1:length(model.sub(dim).rfs),
   RFS = model.sub(dim).rfs(i);

   % compute the weight and its derivate w.r.t. x
   xc = x - RFS.c;
   Dx = RFS.D*xc;
   [w,dwdq] = lwpr_x_kernel(model.kernel,xc'*Dx);
   dwdx = (2*dwdq)*Dx;
   
   % only predict if activation is high enough
   if (w > cutoff & RFS.trustworthy)

      % the mean zero input
      xmz = x - RFS.mean_x;

      % compute the projected inputs
      [s,dsdx] = lwpr_x_compute_projection_d(xmz,RFS.U,RFS.P);
      
      %%%%%%%%% ADDITIONAL SAFETY HEURISTIC: 
      %%%%%%%%% DO NOT INCLUDE NEW PLS DIMENSIONS
      if RFS.n_data(end) <= model.nIn*2
         s(end)=0;          % -> no contribution to prediction
         dsdx(:,end)=0;
      end
      
      % the prediction
      dydx = dsdx * RFS.beta;

      % prediction of i-th model
      yi = RFS.beta'*s + RFS.beta0;

      yp = yp + yi*w;
      sum_w = sum_w + w;
      
      sum_dwdx = sum_dwdx + dwdx;
      sum_ydwdx = sum_ydwdx + yi*dwdx;
      sum_wdydx = sum_wdydx + w*dydx;
      
      % confidence intervals and derivatives
      Gamma = RFS.sum_e_cv2(end)/(RFS.sum_w(end) - RFS.SSp);
            
      sum_s2S = sum(s.^2./RFS.SSs2);
      dsum_s2S_dx = 2*dsdx*(s./RFS.SSs2);

      sigma2 = Gamma*(1+w*sum_s2S);
      dsigma2_dx = Gamma*(dwdx*sum_s2S + w*dsum_s2S_dx);
      
      sum_R    = sum_R    + w*(sigma2 + yi^2);
      sum_dRdx = sum_dRdx + w*(dsigma2_dx + 2*yi*dydx) + dwdx*(sigma2 + yi^2);

   end % if (w > cutoff)
end

% the final prediction
if (sum_w > 0),
   yp = yp/sum_w;
   dypdx = (sum_ydwdx + sum_wdydx)/sum_w - (yp/sum_w)*sum_dwdx;
   
   R    = sum_R    - sum_w   *yp^2;  
   dRdx = sum_dRdx - sum_dwdx*yp^2 - sum_w*2*yp*dypdx;
   
   conf = sqrt(R)/sum_w;   
   dcdx = 0.5/(sum_w * sqrt(R))*dRdx - (conf/sum_w)*sum_dwdx;
else
   yp = 0;
   dypdx = zeros(model.nIn,1);
   conf = Inf;
   dcdx = Inf*ones(model.nIn,1);
end 
