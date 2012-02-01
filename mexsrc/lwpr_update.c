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
   int n,N;
   LWPR_Model model;
   LWPR_Model *pmodel;
   
   const double *x;
   const double *y;
   double *yp;
   double *max_w;
   
   if (nrhs<3) mexErrMsgTxt("Too few arguments.");
   
   pmodel = get_pointer_from_array(prhs[0]);   
   if (pmodel == NULL) {
      create_model_from_matlab(&model,prhs[0]);
      pmodel = &model;
   }
   
   x = mxGetPr(prhs[1]);
   if (mxGetM(prhs[1])!=pmodel->nIn) {
      if (pmodel == &model) lwpr_free_model(&model);
      mexErrMsgTxt("2nd parameter (x) does not match model dimensions.\n");
   }
   N = mxGetN(prhs[1]);
   
   y = mxGetPr(prhs[2]);
   if (mxGetM(prhs[2])!=pmodel->nOut || mxGetN(prhs[2])!=N) {
      if (pmodel == &model) lwpr_free_model(&model);
      mexErrMsgTxt("3rd parameter (y) does not match model dimension or number of input vectors.\n");
   }
   
   plhs[1] = mxCreateDoubleMatrix(pmodel->nOut,N,mxREAL);
   yp = mxGetPr(plhs[1]);
   
   if (nlhs>2) {
      plhs[2] = mxCreateDoubleMatrix(pmodel->nOut,N,mxREAL);
      max_w = mxGetPr(plhs[2]);
      
      for (n=0;n<N;n++) {
         lwpr_update(pmodel, x, y, yp, max_w);   
         x+=pmodel->nIn;
         y+=pmodel->nOut;
         yp+=pmodel->nOut;
         max_w+=pmodel->nOut;         
      }
      
   } else {
      
      for (n=0;n<N;n++) {
         lwpr_update(pmodel, x, y, yp, NULL);   
         x+=pmodel->nIn;
         y+=pmodel->nOut;
         yp+=pmodel->nOut;
      }
      
   }
   
   if (pmodel != &model) {
      plhs[0] = mxDuplicateArray(prhs[0]);
   } else {
      plhs[0] = create_matlab_from_model(&model);
      lwpr_free_model(&model);
   }
}

