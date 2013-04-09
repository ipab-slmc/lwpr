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
   int dim;
   LWPR_Model model;
   
   const double *xn;
   double yn;
   double yp;
   double max_w;
   
   if (nrhs<4) mexErrMsgTxt("Too few arguments.");
   
   create_model_from_matlab(&model,prhs[0]);
   
   dim = (int) mxGetScalar(prhs[1])-1;
   if (dim<0 || dim>=model.nOut) mexErrMsgTxt("2nd parameter (dim) exceeds model size.\n");
   
   xn = mxGetPr(prhs[2]);
   if (mxGetM(prhs[2])!=model.nIn || mxGetN(prhs[2])!=1) {
      lwpr_free_model(&model);
      mexErrMsgTxt("3rd parameter (center) does not match model dimensions.\n");
   }
   yn = mxGetScalar(prhs[3]);

   lwpr_aux_update_one(&model, dim, xn, yn, &yp, &max_w);   
   
   plhs[0] = mxCreateStructMatrix(1,1, SUB_FIELDS, SUB_FIELD_NAMES);
   
   fill_matlab_from_sub(&model.sub[dim], plhs[0], 0);
   
   plhs[1] = mxCreateDoubleScalar(yp);
   plhs[2] = mxCreateDoubleScalar(max_w);
   lwpr_free_model(&model);
}

