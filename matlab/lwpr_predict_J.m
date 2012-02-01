function [yp, dypdx] = lwpr_predict_J(model,x,cutoff)
% [yp, dypdx] = lwpr_predict_J(model, x, [cutoff])
%
% Calculates the predicted output for a given model and input
% data x, and also the Jacobian, i.e. the derivatives of the 
% predictions with respect to the inputs.
%
% Input parameters
%
%     model    Valid LWPR model structure
%     x        Input vector (nIn x 1 matrix)
%     cutoff   Optional threshold parameter for ignoring RFs with
%              low activation. Can be used to speed up predictions.
%
% Return values
%
%     yp       Predicted output (nOut x 1 matrix)
%     dypdx    Jacobian matrix (nOut x nIn)






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
   error('Jacobian can only be calculated for one input vector.');
end

xn = x./model.norm_in;
yp = zeros(model.nOut,1);
dypdx = zeros(model.nOut,model.nIn);

for i=1:model.nOut
   [yp(i), dypidx] = lwpr_x_predict_one_J(model,i,xn,cutoff);
   dypdx(i,:)=dypidx';
end

yp = yp.*model.norm_out;
dypdx = dypdx.*(model.norm_out*(model.norm_in.^(-1))'); 
