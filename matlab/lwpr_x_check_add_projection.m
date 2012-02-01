function rf = lwpr_x_check_add_projection(model,rf)
% rf = lwpr_x_check_add_projection(model,rf)
%
% Checks whether a new projection needs to be added to the
% receptive field, and if yes, modifys the relevant variables.
%
% INPUT 
%  model     An LWPR model structure
%  rf        A receptive field [i.e. model.sub(outdim).rfs(index) ]
%
% OUTPUT
%  rf        The modified or unchanged receptive field




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



[n_in,n_reg] = size(rf.U);
% n_out = model.nOut;  only in org. multi-output implementation
if (n_reg >= n_in)
   return;
end

% here, the mean squared error of the current regression dimension
% is compared against the previous one. Only if there is a signficant
% improvement in MSE, another dimension gets added. Some additional
% heuristics had to be added to ensure that the MSE decision is 
% based on sufficient data 

mse_n_reg   = rf.sum_e_cv2(n_reg)  / rf.sum_w(n_reg) + 1.e-10;
mse_n_reg_1 = rf.sum_e_cv2(n_reg-1)/ rf.sum_w(n_reg-1) + 1.e-10;

if (mse_n_reg/mse_n_reg_1 < model.add_threshold & ...
    rf.n_data(n_reg)/rf.n_data(1) > 0.99 & ...
    rf.n_data(n_reg)*(1.-rf.lambda(n_reg)) > 0.5),

   fprintf(1,'add a dimension');

   rf.beta      = [rf.beta; 0];

   rf.SXresYres = [rf.SXresYres zeros(n_in,1)];
   rf.SSs2      = [rf.SSs2;model.init_S2];
   rf.SSYres    = [rf.SSYres; 0];  

   rf.SSXres    = [rf.SSXres zeros(n_in,1)];
   rf.U         = [rf.U zeros(n_in,1)];
   rf.U(n_reg+1,n_reg+1) = 1;
   rf.P         = [rf.P zeros(n_in,1)];
   rf.P(n_reg+1,n_reg+1) = 1;
   rf.H         = [rf.H; 0];

   rf.r         = [rf.r; 0];
   rf.sum_w     = [rf.sum_w; 1.e-10];
   rf.sum_e_cv2 = [rf.sum_e_cv2; 0];
   rf.n_data    = [rf.n_data; 0];
   rf.lambda    = [rf.lambda; model.init_lambda];
   rf.s         = [rf.s; 0];
   rf.SSp       = 0;

end

