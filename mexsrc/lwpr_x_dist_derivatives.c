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
   const mxArray *ar;
   
   double *dwdM, *dJ2dM, *ddwdMdM, *ddJ2dMdM;
   const double *RF_M,*RF_D,*dx;
   
   double w, dwdq, ddwdqdq;

   int diag_only,meta;
   int m,n;
   double penalty;

   if (nrhs<8) mexErrMsgTxt("Too few arguments.");
   
   w = mxGetScalar(prhs[0]);
   dwdq = mxGetScalar(prhs[1]);
   ddwdqdq = mxGetScalar(prhs[2]);
   
   ar = mxGetField(prhs[3],0,"M");
   if (ar==NULL) mexErrMsgTxt("Couldn't find rf.M");   
   m=mxGetM(ar);
   n=mxGetN(ar);
   if (m!=n) mexErrMsgTxt("rf.M must be square");
   RF_M = mxGetPr(ar);
   
   ar = mxGetField(prhs[3],0,"D");
   if (ar==NULL) mexErrMsgTxt("Couldn't find rf.M");   
   if (n!=mxGetM(ar) || n!=mxGetN(ar)) mexErrMsgTxt("rf.M does not match rf.D");
   RF_D = mxGetPr(ar);
   
   if (n!=mxGetM(prhs[4]) || 1!=mxGetN(prhs[4])) mexErrMsgTxt("dx must be nIn x 1");
   
   dx = mxGetPr(prhs[4]);
   
   diag_only = mxGetScalar(prhs[5]);
   penalty = mxGetScalar(prhs[6]);
   meta = mxGetScalar(prhs[7]);
   
   plhs[0] = mxCreateDoubleMatrix(n,n,mxREAL);
   plhs[1] = mxCreateDoubleMatrix(n,n,mxREAL);
   plhs[2] = mxCreateDoubleMatrix(n,n,mxREAL);
   plhs[3] = mxCreateDoubleMatrix(n,n,mxREAL);
  
   dwdM = mxGetPr(plhs[0]);
   dJ2dM = mxGetPr(plhs[1]);
   ddwdMdM = mxGetPr(plhs[2]);
   ddJ2dMdM = mxGetPr(plhs[3]);
   
   lwpr_aux_dist_derivatives(n, n, dwdM, dJ2dM, ddwdMdM, ddJ2dMdM, w, dwdq, ddwdqdq, RF_D, RF_M, dx, diag_only, penalty, meta);
}

