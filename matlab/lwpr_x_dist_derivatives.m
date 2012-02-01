function [dwdM,dJ2dM,ddwdMdM,ddJ2dMdM] = lwpr_x_dist_derivatives(w,dwdq,ddwdqdq,rf,dx,diag_only,penalty,meta)
% [dwdM,dJ2dM,ddwdMdM,ddJ2dMdM] = lwpr_x_dist_derivatives(w,dwdq,ddwdqdq, ...
%                             rf,dx,diag_only,penalty,meta)
%
% Compute first and second derivatives with respect to the distance metrics,
% or rather to the upper triangular Cholesky factor M
%
% INPUT
%  w               Receptive field activation (output of kernel function)
%  dwdq, ddwdqdq   First and 2nd outer derivative of kernel function
%  rf              Receptive field structure
%  dx              Difference between input vector and receptive field centre
%  diag_only       Flag determining whether the distance metric is treated as diagonal
%  penalty         Pre-factor of smoothness penalty term 
%  meta            Flag determining whether 2nd derivatives should be computed
%
% OUTPUT
%  dwdM        Derivative of activation w with respect to Cholesky factor M
%  dJ2dM       Derivative of penalty term with respect to M
%  ddwdMdM     2nd derivative of w with respect to M
%  ddJ2dMdM    2nd derivative of J2 with respect to M



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

n_in      = length(dx);
dwdM      = zeros(n_in);    
ddwdMdM   = zeros(n_in);
dJ2dM     = zeros(n_in);
ddJ2dMdM  = zeros(n_in);

for n=1:n_in
   for m=n:n_in

      sum_aux    = 0;
      sum_aux1   = 0;

      % take the derivative of D with respect to nm_th element of M */

      if (diag_only & n==m),

         aux = 2*rf.M(n,n);
         dwdM(n,n) = dx(n)^2 * aux;
         sum_aux = rf.D(n,n)*aux;
         if (meta) 
            sum_aux1 = sum_aux1 + aux^2;
         end	

      elseif (~diag_only),

         for i=n:n_in,                                                  

            % aux corresponds to the in_th (= ni_th) element of dDdm_nm  
            % this is directly processed for dwdM and dJ2dM              

            if (i == m)                                                  
               aux = 2*rf.M(n,i);                                         
               dwdM(n,m) = dwdM(n,m) + dx(i) * dx(m) * aux;               
               sum_aux = sum_aux + rf.D(i,m)*aux;                         
               if (meta)                                                  
                  sum_aux1 = sum_aux1 + aux^2;                             
               end                                                        
            else                                                         
               aux = rf.M(n,i);                                           
               dwdM(n,m) = dwdM(n,m) + 2. * dx(i) * dx(m) * aux;          
               sum_aux = sum_aux + 2.*rf.D(i,m)*aux;                      
               if (meta)                                                  
                  sum_aux1 = sum_aux1 + 2*aux^2;                           
               end                                                        
            end                                                          

         end                                                            
      end	  
      
      dwdM_nm = dwdM(n,m);
      dwdM(n,m)  = dwdq*dwdM(n,m);

      dJ2dM(n,m)  = 2.*penalty*sum_aux;

      if (meta)
         ddJ2dMdM(n,m) = 2.*penalty*(2*rf.D(m,m) + sum_aux1);
         ddwdMdM(n,m) = ddwdqdq * dwdM_nm^2 + 2*dwdq*dx(m)^2;
      end
   end
end
