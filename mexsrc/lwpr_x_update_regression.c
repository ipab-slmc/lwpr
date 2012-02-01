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
   LWPR_ReceptiveField RF;
   LWPR_Model model;
   LWPR_Workspace ws;
   const mxArray *ar;
   
   const double *x;
   double w,y;
   double e_cv;
   double yp,e;
   
   if (nrhs<4) mexErrMsgTxt("Too few arguments.");
   
   ar = mxGetField(prhs[0],0,"U");
   if (ar==NULL) mexErrMsgTxt("RF does not contain element 'U'.");
     
   model.nIn = mxGetM(ar);   
   model.nInStore = (model.nIn&1) ? (model.nIn+1) : model.nIn;
   
   create_RF_from_matlab(&RF, &model, prhs[0], 0);
   
   if (mxGetM(prhs[1])!=model.nIn || mxGetN(prhs[1])!=1) mexErrMsgTxt("2nd parameter must be 'nIn x 1' vector.");
   x = mxGetPr(prhs[1]);
   
   y = mxGetScalar(prhs[2]);
   w = mxGetScalar(prhs[3]);
      
   lwpr_mem_alloc_ws(&ws,model.nIn);
   lwpr_aux_update_regression(&RF, &yp, &e_cv, &e, x, y, w, &ws);
   lwpr_mem_free_ws(&ws);

   plhs[0] = mxCreateStructMatrix(1,1, RF_FIELDS, RF_FIELD_NAMES);
   fill_matlab_from_RF(&RF,plhs[0],0); 
   
   plhs[1] = mxCreateDoubleScalar(yp);   
   plhs[2] = mxCreateDoubleScalar(e_cv);
   plhs[3] = mxCreateDoubleScalar(e);   

   lwpr_mem_free_rf(&RF);
}
