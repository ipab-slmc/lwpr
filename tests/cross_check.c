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
#include <lwpr_binio.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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



void testErrors(const LWPR_Model *model, double *MSE, double *wMSE) {
   int i,numPoints = 0;
   double x[2],y[2],yp[2];
   double conf[2],weight[2];
   
   weight[0]=0.0;
   weight[1]=0.0;
   MSE[0]=MSE[1]=wMSE[0]=wMSE[1]=0.0;

   for (x[1]=-1.0; x[1]<=1.01; x[1]+=0.05) {   
      for (x[0]=-1.0; x[0]<=1.01; x[0]+=0.05) {
         y[0] = cross(x[0],x[1]);
         y[1] = y[0] + 10;
         
         lwpr_predict(model, x, 0.001, yp, conf, NULL);
         
         for (i=0;i<2;i++) {
            double diff = y[i]-yp[i];
            double sig2 = conf[i]*conf[i];
            
            weight[i] += 1.0/sig2;
            MSE[i]    += diff*diff;
            wMSE[i]   += diff*diff/sig2;
         }
         numPoints++;
      }
   }
   for (i=0;i<2;i++) {
      MSE[i]/=(double) numPoints;
      wMSE[i]/=weight[i];
   }
}


int main() {

   double x[2],y[2],yp[2];
   double mseTr[2];
   double testErr[2],wTestErr[2];
   double binErr[2], wBinErr[2];
   double xmlErr[2], wXmlErr[2];
   double sumErr;
   
   LWPR_Model model;
   int i,j;
   int numRFS;
   
   /* This allocates some memory and sets initial values 
   ** Note that the model structure itself already exists (on the stack)
   */
   lwpr_init_model(&model,2,2,"2D_Cross");
   
   /* Set initial distance metric to 50*(identity matrix) */
   lwpr_set_init_D_spherical(&model,50);
   
   /* Set init_alpha to 250 in all elements */
   lwpr_set_init_alpha(&model,250);
   
   /* Set w_gen to 0.2 */
   model.w_gen = 0.2;

   /* See above definition, we either use srand() on Windows or srand48 everywhere else */   
   SEED_RAND();
   
   for (j=0;j<20;j++) {
      mseTr[0] = mseTr[1] = 0.0;
      
      for (i=0;i<1000;i++) {
         x[0] = 2.0*URAND()-1.0;
         x[1] = 2.0*URAND()-1.0;
         y[0] = cross(x[0],x[1]) + 0.1*URAND()-0.05;
         y[1] = y[0] + 10; /* sanity check */
         
         /* Update the model with one sample
         **
         ** x points to (x[0],x[1])  (input vector) 
         ** &y points to y           (output "vector")
         ** &yp points to yp         (prediction "vector")
         **
         ** If you are interested in maximum activation, call
         ** lwpr_update(&model, x, &y, &yp, &max_w); 
         */
         lwpr_update(&model, x, y, yp, NULL);
         
         mseTr[0]+=(y[0]-yp[0])*(y[0]-yp[0]);
         mseTr[1]+=(y[1]-yp[1])*(y[1]-yp[1]);         
      }
      mseTr[0]/=500;
      mseTr[1]/=500;      
      printf("#Data = %d   #RFS = %d / %d  MSE = %f / %f\n",model.n_data, model.sub[0].numRFS, model.sub[1].numRFS, mseTr[0], mseTr[1]);
   }
   
   if (model.n_data != 20000) {
      fprintf(stderr,"model.n_data  should have been 20*1000 = 20000. Something is very wrong!\n");
      exit(1);
   }
   
   if (model.sub[0].numRFS != model.sub[1].numRFS) {
      fprintf(stderr,"There should have been an equal number of receptive fields for both outputs :-(\n");
      exit(1);
   }       
   numRFS = model.sub[0].numRFS;
   
   testErrors(&model, testErr, wTestErr);
   printf("MSE on test data: %f / %f\n",testErr[0],  testErr[1]);
   
   if (fabs(testErr[0]-testErr[1]) > 1e-4) {
      fprintf(stderr,"MSE should be equal for both outputs, but the difference is > 1e-4\n");
      exit(1);
   }
   
   printf("Weighted MSE....: %f / %f\n",wTestErr[0], wTestErr[1]);
   if (fabs(wTestErr[0]-wTestErr[1]) > 1e-4) {
      fprintf(stderr,"Weighted MSE should be equal for both outputs, but the difference is > 1e-4\n");
      exit(1);
   }
  
   printf("Writing the model to a binary file\n");
   /* Write the model to an XML file */
   lwpr_write_binary(&model,"lwpr_cross_2d.dat");

   /* Free the memory that was allocated for receptive fields etc. */
   lwpr_free_model(&model);

   printf("Re-read the model from the binary file\n");
   /* Read a model from an XML file, memory allocation is done automatically,
   ** but later lwpr_free_model has to be called again */
   j=lwpr_read_binary(&model,"lwpr_cross_2d.dat");
   remove("lwpr_cross_2d.dat");
   
   if (j==0) {
      fprintf(stderr,"File could not be read, aborting\n");
      exit(1);
   }
   printf("#Data = %d   #RFS = %d / %d\n",model.n_data, model.sub[0].numRFS, model.sub[1].numRFS);
   if (model.n_data != 20000 || model.sub[0].numRFS!=numRFS || model.sub[1].numRFS!=numRFS) {
      fprintf(stderr,"Model (from binary file) seems to be broken :-(\n");
      exit(1);
   }
   
   testErrors(&model, binErr, wBinErr);
   printf("MSE on test data: %f / %f\n",binErr[0],  binErr[1]);
   printf("Weighted MSE....: %f / %f\n",wBinErr[0], wBinErr[1]);
   
   sumErr = fabs(binErr[0] - testErr[0]) + fabs(binErr[1] - testErr[1]);
   sumErr+= fabs(wBinErr[0] - wTestErr[0]) + fabs(wBinErr[1] - wTestErr[1]);
   
   if (sumErr>1e-8) {
      fprintf(stderr,"Error statistics from the binary-IO LWPR model are not the same :-(\n");
      exit(1);
   }
   
   
#if HAVE_LIBEXPAT
   printf("Writing the model to an XML file\n");


   /* Write the model to an XML file */
   lwpr_write_xml(&model,"lwpr_cross_2d.xml");

   /* Free the memory that was allocated for receptive fields etc. */
   lwpr_free_model(&model);

   /* Read a model from an XML file, memory allocation is done automatically,
   ** but later lwpr_free_model has to be called again */
   j=lwpr_read_xml(&model,"lwpr_cross_2d.xml",&i);
   remove("lwpr_cross_2d.xml");

   printf("Re-read the model from the XML file\n");
   printf("%d errors   %d warnings\n",j,i);
   if (j!=0) {
      printf("Errors detected, aborting\n");
      exit(1);
   }

   printf("#Data = %d   #RFS = %d / %d\n",model.n_data, model.sub[0].numRFS, model.sub[1].numRFS);
   if (model.n_data != 20000 || model.sub[0].numRFS!=numRFS || model.sub[1].numRFS!=numRFS) {
      fprintf(stderr,"Model (from XML file) seems to be broken :-(\n");
      exit(1);
   }

   testErrors(&model, xmlErr, wXmlErr);
   printf("MSE on test data: %f / %f\n",xmlErr[0],  xmlErr[1]);
   printf("Weighted MSE....: %f / %f\n",wXmlErr[0], wXmlErr[1]);
   
   sumErr = fabs(xmlErr[0] - testErr[0]) + fabs(xmlErr[1] - testErr[1]);
   sumErr+= fabs(wXmlErr[0] - wTestErr[0]) + fabs(wXmlErr[1] - wTestErr[1]);
   
   sumErr/=fabs(testErr[0]) + fabs(testErr[1]) + fabs(wTestErr[0]) + fabs(wTestErr[1]);
   printf("Relative difference to the original model: %f\n",sumErr);
   
   if (sumErr>0.0001) {
      fprintf(stderr,"Error statistics from the XML-IO LWPR model differ too much :-(\n");
      exit(1);
   }
   
#else

   printf("LWPR library has been compiled without EXPAT support, XML IO will not be tested.\n");

#endif
         
   /* Free the memory that was allocated for receptive fields etc. 
   ** Note again that this does not free the LWPR_Model structure
   ** itself (but it exists on the stack, so it's automatically free'd) */
   lwpr_free_model(&model);
   exit(0);
}
