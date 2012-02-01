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
#include <lwpr_xml.h>
#include <stdio.h>

#ifndef MAX_PATH
#define MAX_PATH  512
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[]) {
   LWPR_Model model;
   char filename[MAX_PATH];
   int numErrors, numWarnings;
   
   if (nrhs!=1 || !mxIsChar(prhs[0])) mexErrMsgTxt("Second argument must be a filename (string).\n");

   mxGetString(prhs[0],filename,MAX_PATH);
   numErrors = lwpr_read_xml(&model, filename, &numWarnings);
   
   if (numErrors==-2) {
      mexErrMsgTxt("LWPR library has been compiled without support for reading XML files (depends on EXPAT)\n");
   } else if (numErrors==-1) {
      mexErrMsgTxt("Could not read XML file. Please check filename and access permissions.\n");
   } else if (numErrors>0) {
      mexErrMsgTxt("XML file seems to be invalid, error(s) occured.\n");
   } 
   
   if (numWarnings>0) {
      printf("Parsing XML file '%s' produced %d warnings.\n",filename,numWarnings);
   }
   
   plhs[0] = create_matlab_from_model(&model);
   
   lwpr_free_model(&model);
}
