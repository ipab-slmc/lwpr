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
#include <lwpr.h>
#include <lwpr_xml.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef WIN32

#include <time.h>
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

   double x[2];
   double y,yp;
   double mse;

   FILE *fp;
   LWPR_Model model;
   int i,j;

   /* This allocates some memory and sets initial values
   ** Note that the model structure itself already exists (on the stack)
   */
   lwpr_init_model(&model,2,1,"2D_Cross");

   /* Set initial distance metric to 50*(identity matrix) */
   lwpr_set_init_D_spherical(&model,50);

   /* Set init_alpha to 250 in all elements */
   lwpr_set_init_alpha(&model,250);

   /* Set w_gen to 0.2 */
   model.w_gen = 0.2;

   /* See above definition, we either use srand() on Windows or srand48 everywhere else */
   SEED_RAND();

   for (j=0;j<20;j++) {
      mse = 0.0;

      for (i=0;i<1000;i++) {
         x[0] = 2.0*URAND()-1.0;
         x[1] = 2.0*URAND()-1.0;
         y = cross(x[0],x[1]) + 0.1*URAND()-0.05;

         /* Update the model with one sample
         **
         ** x points to (x[0],x[1])  (input vector)
         ** &y points to y           (output "vector")
         ** &yp points to yp         (prediction "vector")
         **
         ** If you are interested in maximum activation, call
         ** lwpr_update(&model, x, &y, &yp, &max_w);
         */
         lwpr_update(&model, x, &y, &yp, NULL);

         mse+=(y-yp)*(y-yp);
      }
      mse/=500;
      printf("#Data = %d   #RFS = %d   MSE = %f\n",model.n_data, model.sub[0].numRFS, mse);
   }

   /* Write the model to an XML file */
   lwpr_write_xml(&model,"lwpr_cross_2d.xml");

#if HAVE_LIBEXPAT
   /* Free the memory that was allocated for receptive fields etc. */
   lwpr_free_model(&model);

   /* Read a model from an XML file, memory allocation is done automatically,
   ** but later lwpr_free_model has to be called again */
   j=lwpr_read_xml(&model,"lwpr_cross_2d.xml",&i);

   printf("Re-read the model from the XML file\n");
   printf("%d errors   %d warnings\n",j,i);

   if (j!=0) {
      printf("Errors detected, aborting\n");
      exit(1);
   }
#endif

   fp = fopen("output.txt","w");

   mse = 0.0;
   i=0;

   for (x[1]=-1.0; x[1]<=1.01; x[1]+=0.05) {
      for (x[0]=-1.0; x[0]<=1.01; x[0]+=0.05) {
         y = cross(x[0],x[1]);

         /* Use the model for predicting an output
         **
         ** x points to (x[0],x[1])     (input vector)
         ** 0.001  is the cutoff value  (clip Gaussian kernel)
         ** &yp points to yp            (prediction "vector")
         **
         ** If you are interested in confidence bounds or
         ** maximum activation, call
         ** lwpr_predict(&model, x, 0.001, &yp, &conf, &max_w);
         */
         lwpr_predict(&model, x, 0.001, &yp, NULL, NULL);

         mse += (y-yp)*(y-yp);
         i++;

         fprintf(fp,"%8.5f %8.5f %8.5f\n",x[0],x[1],yp);
      }
      fprintf(fp,"\n\n");
   }
   fclose(fp);

   printf("MSE on test data (%d) = %f\n",i,mse/(double) i);

   printf("\nTo view the output, start gnuplot, and type:\n");
   printf("   splot \"output.txt\"\n\n");

   /* Free the memory that was allocated for receptive fields etc.
   ** Note again that this does not free the LWPR_Model structure
   ** itself (but it exists on the stack, so it's automatically free'd) */
   lwpr_free_model(&model);
   return 0;
}
