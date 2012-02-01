function model = lwpr_set(model,varargin)
% model = lwpr_set(model, param_name, param_value, ... )
%
% This function should be used to modify parameters of an LWPR 
% model structure.
%
% Input parameters
%
%     model        LWPR model structure 
%     param_name   Name of parameter that should be modified
%     param_value  New value of that parameter
%
% Result
%
%     Modified LWPR model structure.
%
% You can pass as many name/value pairs as you like. 
%
%     model = lwpr_set(model,'diag_only',1,'meta',0)
%
% has the same effect as
%
%     model = lwpr_set(model,'diag_only',1);
%     model = lwpr_set(model,'meta',0);
%
% The following list enumerates the changeable parameters 
% and their possible values.
%
%     'diag_only' (flag)
%        1 --> use only diagonal distance matrices
%        0 --> use full matrices
%
%     'meta'      (flag)  
%        1 --> use meta learning, i.e. perform second order
%              updates to the distances metrics
%        0 --> do not use meta learning
%                   
%     'meta_rate' (scalar)
%        Learning rate for 2nd order distance metric updates 
%
%     'penalty'   (scalar)
%        Pre-factor of smoothness penalty term
%
%     'init_alpha'   (scalar or nIn x nIn matrix)
%        Per-element learning rate for meta updates
%
%     'norm_in'   (scalar or nIn x 1 matrix)
%        Input normalisation factors. Set this to the expected
%        scale (std. deviation) of your input data
%
%     'norm_out'  (scalar or nOut x 1 matrix)
%        Output normalisation factors. Set this to the expected
%        scale (std. deviation) of your output data
%
%     'name'      (string)
%        A descriptive name for your LWPR model.
%
%     'init_D'    (scalar, nIn x 1 vector (diagonal), or nIn x nIn matrix)
%        Initial distance metric for new receptive fields (RF)
%
%     'update_D'  (flag)
%        1 --> perform distance metric updates
%        0 --> keep all distance matrices constant
%
%     'w_gen'     (scalar)
%        Threshold that determines the minimum activation before 
%        a new RF is created. 
%
%     'w_prune'   (scalar)
%        Threshold that determines above which (second highest) 
%        activation a RF is pruned. Setting a value >= 1.0 effectively
%        disables pruning.
%
%     'init_lambda'  (scalar)
%        Initial forgetting factor
%
%     'final_lambda' (scalar)
%     	Final forgetting factor
%
%     'tau_lambda'   (scalar)
%      	This parameter describes the annealing schedule of the 
%        forgetting factor
%
%     'init_S2'    (scalar)
%      	Initial value for sufficient statistics SSs2.
%
%     'add_threshold'   (scalar)
%      	Threshold that determines when a new PLS regression axis is 
%        added.
%
%     'kernel'    (string)
%        Either 'Gaussian' or 'BiSquare', determines which basis
%        function to use for calculating activations from distances.





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



if mod(nargin,2)~=1
  error('Incorrect call to lwpr_set');
end

nIn = model.nIn;
nOut = model.nOut;

if ~isscalar(nIn) | nIn<1 | nIn~=round(nIn) | ~isscalar(nOut) | nOut<1 | nOut~=round(nOut)
   error('Invalid model: Check <nIn> and <nOut>');
end

ii=1;
while ii<nargin-1
   param = varargin{ii};
   value = varargin{ii+1};
   
   switch param
      case 'diag_only'
         if ~(isequal(value,0) | isequal(value,1)) 
            error('<diag_only> must be either 0 or 1');
         end
         model.diag_only = value;
         
      case 'meta'
         if ~(isequal(value,0) | isequal(value,1)) 
            error('<meta> must be either 0 or 1');
         end
         model.meta = value;
         
      case 'meta_rate'
         if ~isscalar(value) | value<0
            error('<meta_rate> must be a positive scalar');
         end
         model.meta_rate = value;
         
      case 'penalty'
         if ~isscalar(value) | value<0
            error('<penalty> must be a positive scalar');
         end
         model.penalty = value;
         
      case 'init_alpha'
         if any(value<=0)
            error('<init_alpha> must contain only positive elements');
         end
         if isequal(size(value),[nIn nIn]) 
            model.init_alpha = value;
         elseif isscalar(value)
            model.init_alpha = value * ones(nIn,nIn);
         else
            error('<init_alpha> must be a positive scalar or (nIn x nIn) matrix');
         end
         
      case 'norm_in'
         if isscalar(value)
            model.norm_in = value*ones(nIn,1);
         elseif isequal(size(value),[nIn 1])
            model.norm_in = value;            
         else
            error('<norm_in> must be either scalar or column vector (nIn x 1)');
         end
         
      case 'norm_out'
         if isscalar(value)
            model.norm_out = value*ones(nOut,1);
         elseif isequal(size(value),[nOut 1])
            model.norm = value;            
         else
            error('<norm_out> must be either scalar or column vector (nOut x 1)');
         end
         
      case 'name'
         model.name = value;
         
      case 'init_D'
         if isscalar(value)
            model.init_D = value*eye(nIn);
            model.init_M = sqrt(value)*eye(nIn);
         elseif isequal(size(value),[nIn 1])
            model.init_D = diag(value);
            model.init_M = diag(value.^0.5);
         elseif isequal(size(value),[nIn nIn])
            model.init_D = value;
            model.init_M = chol(value);
         else 
            error('init_D must be a scalar, a vector (nIn x 1) or a square matrix (nIn x nIn)');
         end
         
      case 'update_D'
         if ~(isequal(value,0) | isequal(value,1)) 
            error('<update_D> must be either 0 or 1');
         end
         model.update_D = value;
         
      case 'w_gen'
         if ~isscalar(value) | value<0 
            error('<w_gen> must be a positive scalar');
         end
         model.w_gen = value; 
         
      case 'w_prune'
         if ~isscalar(value) | value<0 
            error('<w_prune> must be a positive scalar');
         end
         model.w_prune = value; 
         
      case 'init_lambda'
         if ~isscalar(value) | value<0 
            error('<init_lambda> must be a positive scalar');
         end
         model.init_lambda = value; 
         
      case 'final_lambda'
         if ~isscalar(value) | value<0 
            error('<final_lambda> must be a positive scalar');
         end
         model.final_lambda = value; 

      case 'tau_lambda'
         if ~isscalar(value) | value<0 
            error('<tau_lambda> must be a positive scalar');
         end
         model.tau_lambda = value; 
         
      case 'init_S2'
         if ~isscalar(value) | value<0 
            error('<init_S2> must be a positive scalar');
         end
         model.init_S2 = value; 
         
      case 'add_threshold'
         if ~isscalar(value) | value<0 
            error('<add_threshold> must be a positive scalar');
         end
         model.add_threshold = value; 
         
      case 'kernel'
         if isequal(value,'Gaussian')
            model.kernel = 'Gaussian';
         elseif isequal(value,'BiSquare')
            model.kernel = 'BiSquare';
         else
            error('<kernel> must either be "Gaussian" or "BiSquare"');
         end
         
      otherwise
         disp(param);
         error('Unrecognised parameter!');
   end
   ii=ii+2;
end



function yesno = isscalar(value)

yesno = isa(value,'double') & prod(size(value)) == 1;
