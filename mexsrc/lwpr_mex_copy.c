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
#include <stdio.h>

#ifndef MAX_PATH
#define MAX_PATH  512
#endif

/* For debugging purposes only */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[]) {
   LWPR_Model model, model2;

   create_model_from_matlab(&model, prhs[0]);
   
   if (!lwpr_duplicate_model(&model2, &model)) mexErrMsgTxt("Cannot copy internally!");
   
   plhs[0] = create_matlab_from_model(&model);
   plhs[1] = create_matlab_from_model(&model2);   
   
   lwpr_free_model(&model);
   lwpr_free_model(&model2);   
}
