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
#include <lwpr.hh>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef WIN32

#define SEED_RAND()     srand(time(NULL))
#define URAND()         (((double)rand())/ (double)RAND_MAX)

#else

#define SEED_RAND()     srand48(time(NULL))
#define URAND()         drand48()

#endif

double cross(double x1,double x2) {
   double a = exp(-10*x1*x1);
   double b = exp(-50*x2*x2);
   double c = 1.25*exp(-5*(x1*x1 + x2*x2));
   
   if (a>b) {
      return (a>c) ? a:c;
   } else {
      return (b>c) ? b:c;
   }
}


int main() {

   doubleVec x(2);
   doubleVec y(1);

   double mse;
   unsigned int i,j,numTest;
   
   FILE *fp;
   LWPR_Object model(2,1);
      
   /* Set initial distance metric to 50*(identity matrix) */
   model.setInitD(50);
   
   /* Set init_alpha to 250 in all elements */
   model.setInitAlpha(250);
   
   /* Set w_gen to 0.2 */
   model.wGen(0.2);

   /* See above definition, we either use srand() on Windows or srand48 everywhere else */   
   SEED_RAND();
   
   doubleVec u(3);
      
   for (j=0;j<20;j++) {
      mse = 0.0;
      
      for (i=0;i<1000;i++) {
         
         x[0] = 2.0*URAND()-1.0;
         x[1] = 2.0*URAND()-1.0;
         y[0] = cross(x[0],x[1]) + 0.1*URAND()-0.05;
         
         // Update the model with one sample
         
         doubleVec yp = model.update(x,y);
         
         mse+=(y[0]-yp[0])*(y[0]-yp[0]);
      }
      mse/=500;
      printf("#Data = %d   #RFS = %d   MSE = %f\n",model.nData(), model.numRFS(0), mse);
   }
   
   fp = fopen("output.txt","w");
   
   mse = 0.0; 
   numTest=0;  

   for (x[1]=-1.0; x[1]<=1.01; x[1]+=0.05) {   
      for (x[0]=-1.0; x[0]<=1.01; x[0]+=0.05) {
         y[0] = cross(x[0],x[1]);
         
         // Use the model for predicting an output
         doubleVec yp = model.predict(x);
         
         mse += (y[0]-yp[0])*(y[0]-yp[0]);
         numTest++;
         
         fprintf(fp,"%8.5f %8.5f %8.5f\n",x[0],x[1],yp[0]);
      }
      fprintf(fp,"\n\n");
   }
   fclose(fp);
   
   printf("MSE on test data (%d) = %f\n",numTest,mse/(double) numTest);
   
   
   // retrieve a wrapper object of the first receptive field 
   LWPR_ReceptiveFieldObject rf = model.getRF(0,0);

   printf("\nCholesky factors of RF(0,0) distance metric\n(printed in lower triangular form):\n");
   std::vector<doubleVec> M = rf.M();
   for (i=0;i<M.size();i++) {
      for (j=0;j<M[i].size();j++) {
         printf("%8.4f  ",M[i][j]);
      }
      printf("\n");
   }
   
   printf("\nCenter, offset and slope of that RF:\n");
   doubleVec center = rf.center();
   doubleVec slope  = rf.slope();
   printf("(%8.4f, %8.4f)  %8.4f  (%8.4f, %8.4f)\n",
      center[0],center[1],rf.beta0(),slope[0],slope[1]);

   printf("\nPLS coefficients 'beta' of that RF:\n");
   doubleVec beta = rf.beta();
   for (i=0;i<beta.size();i++) {
      printf("%8.4f  ",beta[i]);
   }
   printf("\n");
         
   model.writeXML("lwpr_cross_2d.xml");
   
   printf("\nTo view the output, start gnuplot, and type:\n");
   printf("   splot \"output.txt\"\n\n");
}
