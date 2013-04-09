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
   LWPR_Model *pmodel;   
      
   if (nrhs<1) mexErrMsgTxt("Too few arguments.");   
   
   pmodel = get_pointer_from_array(prhs[0]);
   if (pmodel == NULL) {
      create_model_from_matlab(&model,prhs[0]);
      pmodel = &model;
   }
   
   plhs[0] = mxCreateDoubleScalar(pmodel->n_data);
   
   if (pmodel == &model) lwpr_free_model(&model);
}

