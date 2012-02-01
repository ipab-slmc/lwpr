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
   
   const double *xn;
   double cutoff;
      
   double *yp;
   double *max_w;  
   double *J,*H; 
   int sizeH[3];
      
   if (nrhs<2) mexErrMsgTxt("Too few arguments.");   

   pmodel = get_pointer_from_array(prhs[0]);   
   if (pmodel == NULL) {
      create_model_from_matlab(&model,prhs[0]);
      pmodel = &model;
   }
      
   xn = mxGetPr(prhs[1]);
   if (mxGetM(prhs[1])!=pmodel->nIn) {
      if (pmodel == &model) lwpr_free_model(&model);
      mexErrMsgTxt("3rd parameter (x) does not match model dimensions.\n");
   }
   N = mxGetN(prhs[1]);
   
   if (N!=1) mexErrMsgTxt("Jacobian can only be computed for one input vector.\n");
   
   if (nrhs==3) {
      cutoff = mxGetScalar(prhs[2]);
   } else {
      cutoff = 0.0;
   }
   
   plhs[0] = mxCreateDoubleMatrix(pmodel->nOut,N,mxREAL);
   yp = mxGetPr(plhs[0]);
   
   plhs[1] = mxCreateDoubleMatrix(pmodel->nOut,pmodel->nIn,mxREAL);
   J = mxGetPr(plhs[1]);
   
   sizeH[0]=pmodel->nIn;
   sizeH[1]=pmodel->nIn;
   sizeH[2]=pmodel->nOut;

   plhs[2] = mxCreateNumericArray(3, sizeH, mxDOUBLE_CLASS,mxREAL);   
   H = mxGetPr(plhs[2]);

   lwpr_predict_JH(pmodel, xn, cutoff, yp, J, H);
   
   if (pmodel == &model) lwpr_free_model(&model);
}

