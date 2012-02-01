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
#include <lwpr_binio.h>
#include <stdio.h>

#ifndef MAX_PATH
#define MAX_PATH  512
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[]) {
   LWPR_Model model;
   FILE *fp;
   char filename[MAX_PATH];
   int ok;
   
   if (nrhs!=1 || !mxIsChar(prhs[0])) mexErrMsgTxt("Second argument must be a filename (string).\n");

   mxGetString(prhs[0],filename,MAX_PATH);
   
   fp = fopen(filename, "rb");
   if (fp==NULL) {
      mexErrMsgTxt("Could not open the file. Please check filename and access permissions.\n");
   }
   
   ok = lwpr_read_binary_fp(&model, fp);
   if (!ok) mexErrMsgTxt("LWPR file seems to be invalid, error(s) occured.\n");
   fclose(fp);   
   
   plhs[0] = create_matlab_from_model(&model);
   
   lwpr_free_model(&model);
}
