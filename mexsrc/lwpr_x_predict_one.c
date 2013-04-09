/*********************************************************************
LWPR: A library for incremental online learning
Copyright (C) 2007  Stefan Klanke, Sethu Vijayakumar
Contact: sethu.vijayakumar@ed.ac.uk

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either 
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free
Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*********************************************************************/
#include <lwpr_matlab.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[]) {
   LWPR_Model model;
   LWPR_ThreadData TD;    
      
   if (nrhs<4) mexErrMsgTxt("Too few arguments.");

   create_model_from_matlab(&model,prhs[0]);

   TD.model = &model;
   TD.ws = &model.ws[0];
   
   TD.dim = (int) mxGetScalar(prhs[1])-1;
   if (TD.dim<0 || TD.dim>=model.nOut) mexErrMsgTxt("2nd parameter (dim) exceeds model size.\n");
   
   TD.xn = mxGetPr(prhs[2]);
   if (mxGetM(prhs[2])!=model.nIn || mxGetN(prhs[2])!=1) {
      lwpr_free_model(&model);
      mexErrMsgTxt("3rd parameter (x) does not match model dimensions.\n");
   }
   TD.cutoff = mxGetScalar(prhs[3]);
   
   lwpr_aux_predict_conf_one_T(&TD);
   
   plhs[0] = mxCreateDoubleScalar(TD.yn);
   if (nlhs>1) plhs[1] = mxCreateDoubleScalar(TD.w_sec);
   if (nlhs>2) plhs[2] = mxCreateDoubleScalar(TD.w_max);
   lwpr_free_model(&model);
}

