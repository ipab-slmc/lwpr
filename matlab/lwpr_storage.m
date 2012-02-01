function model = lwpr_storage(command,model)
% model = lwpr_storage(command,model)
%
% Transfers MATLAB structures of LWPR models into MEX-file
% internal storage and vice versa. Only useful if the 
% MEX-implementation is used.
%
% Input parameters
%     
%     model    Valid LWPR model structure  or  storage-ID
%     command  Describes which action to perform
%
% Return value
%
%     model    LWPR model structure  or  storage-ID
%
% Examples:
%
% > model_ID = lwpr_storage('Store', model)
% Transfers the MATLAB struct 'model' into MEX-storage
% and returns an ID (uInt32 or uInt64), which is effectively
% a pointer to an LWPR_Model structure in the C implementation.
%
% This ID can be passed to lwpr_predict, lwpr_predict_J and lwpr_update
% instead of a normal MATLAB struct, which yields dramatic speedups,
% since the overhead of converting between MATLAB and C is removed.
% Note that you cannot access the internal variables of the model 
% in this mode.
%
% > model = lwpr_storage('Get', model_ID)
% Retrieves a MATLAB struct from MEX-internal storage, pointed to
% by the input parameter 'model'. Useful for inspecting internal
% variables of the model after it has been trained.
%
% > lwpr_storage('Free', model_ID)
% Disposes the internally stored model pointed to by model_ID
%
% > model = lwpr_storage('GetFree', model_ID)
% Combines the calls with 'Get' and 'Free' commands
%
% > lwpr_storage('FreeAll')
% Disposes all internally stored models. This call is usually not
% necessary, since all memory is free'd as soon as the MEX-library
% is unloaded.









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


warning 'MATLAB implementation of lwpr_storage is just a dummy. Use lwpr_buildmex';

switch(command)
   case 'Store'
      return
   case 'GetFree'
      return
   case 'Get'
      return
   case 'Free'
      model = [];
   case 'FreeAll'
      model = [];
   otherwise
      error('Bad 1st argument (command)');
end
      
      
