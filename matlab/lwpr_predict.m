function [yp, conf, max_w] = lwpr_predict(model,x,cutoff)
% [yp, conf, max_w] = lwpr_predict(model, x, [cutoff])
%
% Calculates the predicted output for a given model and input
% data x. Optionally also returns confidence bounds and maximum
% activations.
%
% Input parameters
%
%     model    Valid LWPR model structure
%     x        N input vectors (as nIn x N matrix), i.e. a
%              single input sample is passed as a column vector
%     cutoff   Optional threshold parameter for ignoring RFs with
%              low activation. Can be used to speed up predictions.
%
% Return values
%
%     yp       Predicted outputs (nOut x N matrix). If only a single
%              input sample was given, the output will be a single
%              column vector.
%     conf     Confidence bounds per output dimension.
%     max_w    Maximum RF activation per output dimension.





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

if nargin<3
   cutoff = 0;
end

[d,N] = size(x);

if N>1
   xn = x.*repmat(model.norm_in.^(-1),1,N);
else
   xn = x./model.norm_in;
end

yp = zeros(model.nOut,N);

switch nargout
   case 1
      for n=1:N
         for i=1:model.nOut
            yp(i,n) = lwpr_x_predict_one(model,i,xn(:,n),cutoff);
         end
      end
      
   case 2
      conf = zeros(model.nOut,N);

      for n=1:N
         for i=1:model.nOut
            [yp(i,n), conf(i,n)] = lwpr_x_predict_one(model,i,xn(:,n),cutoff);
         end
      end
      
   case 3
      max_w = zeros(model.nOut,N);
      conf = zeros(model.nOut,N);
      
      for n=1:N
         for i=1:model.nOut
            [yp(i,n), conf(i,n), max_w(i,n)] = lwpr_x_predict_one(model,i,xn(:,n),cutoff);
         end
         conf(:,n) = conf(:,n).*model.norm_out;
      end
end

if N>1
   yp = yp.*repmat(model.norm_out, 1, N);
else
   yp = yp.*model.norm_out;
end
