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
   int ok;
   LWPR_ReceptiveField RF,RFT;
   LWPR_Model model;
   
   const double *xc;
   double y;
   
   if (nrhs<4) mexErrMsgTxt("Too few arguments.");
   
   create_model_from_matlab(&model,prhs[0]);
   
   xc = mxGetPr(prhs[2]);
   if (mxGetM(prhs[2])!=model.nIn || mxGetN(prhs[2])!=1) {
      lwpr_free_model(&model);
      mexErrMsgTxt("3rd parameter (center) does not match model dimensions.\n");
   }
   y = mxGetScalar(prhs[3]);
   
   if (mxIsEmpty(prhs[1])) {
      ok = lwpr_aux_init_rf(&RF,&model,NULL,xc,y);
   } else {
      create_RF_from_matlab(&RFT,&model, prhs[1], 0);
      ok = lwpr_aux_init_rf(&RF,&model,&RFT,xc,y);
      lwpr_mem_free_rf(&RFT);
   } 
   
   lwpr_free_model(&model);
   
   if (!ok) mexErrMsgTxt("Couldn't allocate storage for RF.\n");
   
   plhs[0] = mxCreateStructMatrix(1,1, RF_FIELDS, RF_FIELD_NAMES);
   fill_matlab_from_RF(&RF,plhs[0],0); 
   
   lwpr_mem_free_rf(&RF);
}

