function [model, yp, max_w] = lwpr_update(model,x,y)
% [model, yp, max_w] = lwpr_update(model, x, y)
%
% Updates the LWPR model using an input/output sample (x,y)
% Optionally returns the current prediction for x and the
% maximum RF activation.
%
% Input parameters
%
%     model    Valid LWPR model structure
%     x        Input vector (nIn x 1 matrix)
%     y        Output vector (nOut x 1 matrix)
%
% Return values
%
%     model    Updated LWPR model structure
%     yp       Current prediction of y given x, useful for
%              tracking the training error
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

    
% update the global mean and variance of the training data for
% information purposes
model.mean_x = (model.mean_x * model.n_data + x)/(model.n_data+1);
model.var_x  = (model.var_x * model.n_data + (x - model.mean_x).^2)/(model.n_data+1);
model.n_data = model.n_data + 1;
    
% normalize the inputs
xn = x./model.norm_in;
    
% normalize the outputs
yn = y./model.norm_out;

yp = zeros(model.nOut,1);
max_w = zeros(model.nOut,1);

for i=1:model.nOut
   [model.sub(i), yp(i), max_w(i)] = lwpr_x_update_one(model,i,xn,yn(i));
end

yp = yp.*model.norm_out;
