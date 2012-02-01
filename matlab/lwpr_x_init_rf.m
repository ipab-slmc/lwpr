function rf = lwpr_x_init_rf(model,template_rf,c,y)
% rf = lwpr_x_init_rf(model,template_rf,c,y)
%
% Initialise a receptive field (local linear model)
%
% INPUT
%  model       LWPR model structure
%  template_rf Template receptive field for copying distance metrics, pass
%              [] if you wish to use the model-global init_D etc.
%  c           Centre of the new receptive field
%  y           Initial offset (or bias) beta0 of the receptive field
%
% OUTPUT
%  rf          The new receptive field




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


if ~isempty(template_rf),
  rf = template_rf;
else
  rf.D     = model.init_D;
  rf.M     = model.init_M;
  rf.alpha = model.init_alpha;
  rf.beta0 = y;                             % the weighted mean of output
end

% if more than univariate input, start with two projections such that
% we can compare the reduction of residual error between two projections
n_in = model.nIn;

if (n_in > 1) 
  n_reg = 2;
else
  n_reg = 1;
end

rf.beta        = zeros(n_reg,1);             % the regression parameters
rf.c           = c;                          % the center of the RF
rf.SXresYres   = zeros(n_in,n_reg);          % needed to compute projections
rf.SSs2        = ones(n_reg,1)*model.init_S2; % variance per projection
rf.SSYres      = zeros(n_reg,1);             % needed to compute linear model
rf.SSXres      = zeros(n_in,n_reg);          % needed to compute input reduction
rf.U           = eye(n_in,n_reg);            % matrix of projections vectors
rf.P           = eye(n_in,n_reg);            % reduction of input space
rf.H           = zeros(n_reg,1);             % trace matrix
rf.r           = zeros(n_reg,1);             % trace vector
rf.h           = zeros(size(rf.alpha));      % a memory term for 2nd order gradients
rf.b           = triu(log(rf.alpha+1.e-10)); % a memory term for 2nd order gradients
rf.sum_w       = ones(n_reg,1)*1.e-10;       % the sum of weights
rf.sum_e_cv2   = zeros(n_reg,1);             % weighted sum of cross.valid. err. per dim
rf.sum_e2      = 0;                          % same as above, but without CV
rf.n_data      = ones(n_reg,1)*1.e-10;       % discounted amount of data in RF
rf.trustworthy = 0;                          % indicates statistical confidence
rf.lambda      = ones(n_reg,1)*model.init_lambda; % forgetting rate
rf.mean_x      = zeros(n_in,1);              % the weighted mean of inputs
rf.var_x       = zeros(n_in,1);              % the weighted variance of inputs
rf.w           = 0;                          % store the last computed weight
rf.s           = zeros(n_reg,1);             % store the projection of inputs
rf.SSp         = 0;                          % sufficient statistics for confidence bounds
