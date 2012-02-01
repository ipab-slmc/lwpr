function [w, dwdq, ddwdqdq] = lwpr_x_kernel(kernel,q)
% [w, dwdq, ddwdqdq] = lwpr_x_kernel(kernel,q)
%
% Computes activation and first and second outer derivatives
% of a receptive field given the quadratic distance q of
% an input vector from the receptive fields centre
%
% INPUT
%  kernel   String describing the kernel function. In the current
%           implementation, only the first letter is looked at:
%           'B' for Bisquare kernel, 'G' for Gaussian kernel
%  q        Squared distance between input and RF centre
%
% OUTPUT
%  w        Activiation / weight
%  dwdq     First derivative with respect to q
%  ddwdqdq  2nd derivative with respect to q


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

switch kernel(1)
   case 'B'
      e = 1-0.25*q;
      if e<0
         w = 0;
         dwdq = 0;
         ddwdqdq = 0;
      else
         w = e^2;
         dwdq = -0.5*e;
         ddwdqqd = 0.125;
      end
      
   case 'G'
      w = exp(-0.5*q);
      dwdq = -0.5 * w;
      ddwdqdq = 0.25 * w;

   otherwise
      error('Unknown kernel function');
end
