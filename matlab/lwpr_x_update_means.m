function [rf,xmz,ymz] = lwpr_x_update_means(rf,x,y,w)
% [rf,xmz,ymz] = lwpr_x_update_means(rf,x,y,w)
%
% Update receptive field mean and variance statistics, 
% and also compute mean zero variables
%
% INPUT
%  rf    Receptive field structure
%  x     Input vector sample
%  y     Output sample (specific to the sub-model dimension)
%  w     Activation of the receptive field
%
% OUTPUT
%  rf    Updated receptive field (mean_x, var_x, beta0 = mean_y)
%  xmz   Input vector minus the new mean_x
%  ymz   Output sample minus the new beta0



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

sum_w_lambda = rf.sum_w(1)*rf.lambda(1);

rf.mean_x = (sum_w_lambda*rf.mean_x + w*x)/(sum_w_lambda + w);
rf.var_x  = (sum_w_lambda*rf.var_x + w*(x-rf.mean_x).^2)/(sum_w_lambda + w);
rf.beta0  = (sum_w_lambda*rf.beta0 + w*y)/(sum_w_lambda + w);
xmz = x - rf.mean_x;
ymz = y - rf.beta0;


