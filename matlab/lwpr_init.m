function model = lwpr_init(nIn,nOut,varargin)
% model = lwpr_init(nIn, nOut, ... )
%
% This function creates an LWPR model structure suitable for a
% learning task with nIn inputs and nOut outputs.
%
% Input parameters
%
%    nIn   dimensionality of input data
%    nOut  dimensionality of output data
%
% Result
%
%    An initial LWPR model structure.
%    
% Further (optional) parameters are passed to lwpr_set, e.g.
%
%    model = lwpr_init(6,3,'diag_only',1,'meta',0)
%
% has the same effect as
%
%    model = lwpr_init(6,3);
%    model = lwpr_set(model,'diag_only',1);
%    model = lwpr_set(model,'meta',0);
%
% For a discussion of those optional parameters, please see lwpr_set


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



if nargin==0 | mod(nargin,2)==1
  error('Incorrect call to lwpr_init');
end

model = struct;

if length(nIn(:))~=1 | nIn<1 | nIn~=round(nIn)
   error('First parameter (input dimension) must be a natural number (1,2,..)');
end

if length(nOut(:))~=1 | nOut<1 | nOut~=round(nOut)
   error('Second parameter (output dimension) must be a natural number (1,2,..)');
end

model.nIn = nIn;
model.nOut = nOut;
model.n_data = 0;
model.mean_x = zeros(nIn,1);
model.var_x  = zeros(nIn,1);

if nargin>2
   model=lwpr_set(model,varargin{:});
end


%% Set default values

if ~isfield(model,'diag_only')
   model.diag_only=1;
end

if ~isfield(model,'update_D')
   model.update_D=1;
end

if ~isfield(model,'meta')
   model.meta = 0;
end

if ~isfield(model,'meta_rate')
   model.meta_rate = 250;
end

if ~isfield(model,'penalty')
   model.penalty = 1e-6;
end

if ~isfield(model,'init_alpha')
   model.init_alpha = 50 * ones(nIn,nIn);
end

if ~isfield(model,'norm_in')
   model.norm_in = ones(nIn,1);
end

if ~isfield(model,'norm_out')
   model.norm_out = ones(nOut,1);
end

if ~isfield(model,'init_D')
   model.init_D = 25*eye(nIn);
   model.init_M = chol(model.init_D);
end

if ~isfield(model,'name')
   model.name = '';
end

if ~isfield(model,'w_gen')
   model.w_gen = 0.1;
end

if ~isfield(model,'w_prune')
   model.w_prune = 1.0;
end

if ~isfield(model,'init_lambda')
   model.init_lambda  = 0.999;
end

if ~isfield(model,'final_lambda')
   model.final_lambda = 0.99999;
end

if ~isfield(model,'tau_lambda')
   model.tau_lambda   = 0.9999;
end

if ~isfield(model,'init_S2')
   model.init_S2 = 1e-10;
end

if ~isfield(model,'add_threshold')
   model.add_threshold = 0.5;
end

if ~isfield(model,'kernel')
   model.kernel = 'Gaussian';
end

subInit=struct;
subInit.rfs = [];
subInit.n_pruned = 0;

model.sub = repmat(subInit,nOut,1);
