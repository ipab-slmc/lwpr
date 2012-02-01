function ok = lwpr_write_binary(model, filename)
%function ok = lwpr_write_binary(model, filename)
%
% Writes an LWPR model (given as a MATLAB structure) into
% a binary file.
%
% Input parameters
%     
%     model       Valid LWPR model structure
%     filename    Name of the destination file 
%
% Return value
%
%     ok       1 in case of success,   0 on failure
%


error 'Binary export of LWPR models if only available through MEX-files (see lwpr_buildmex)'



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


