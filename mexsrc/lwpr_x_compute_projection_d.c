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
   int nIn,nReg;
   
   const double *x,*U,*P;
   double *s,*dsdx;
   
   LWPR_Workspace ws;
   
   if (nrhs<3) mexErrMsgTxt("Too few arguments.");
   
   nIn = mxGetM(prhs[0]);
   if (mxGetN(prhs[0])!=1) mexErrMsgTxt("First argument must be a column vector.");
   
   if (nIn != mxGetM(prhs[1])) mexErrMsgTxt("Second argument has wrong dimensions."); 

   nReg = mxGetN(prhs[1]);
   
   if (nIn != mxGetM(prhs[2]) || nReg!=mxGetN(prhs[2])) mexErrMsgTxt("Second argument has wrong dimensions.");       
   
   x = mxGetPr(prhs[0]);
   U = mxGetPr(prhs[1]);
   P = mxGetPr(prhs[2]);
   
   plhs[0] = mxCreateDoubleMatrix(nReg,1,mxREAL);   
   s = mxGetPr(plhs[0]);
   
   plhs[1] = mxCreateDoubleMatrix(nIn,nReg,mxREAL);   
   dsdx = mxGetPr(plhs[1]);
   
   lwpr_mem_alloc_ws(&ws,nIn);
      
   lwpr_aux_compute_projection_d(nIn,nIn,nReg,s,dsdx,x,U,P,&ws);
   
   lwpr_mem_free_ws(&ws);
}

