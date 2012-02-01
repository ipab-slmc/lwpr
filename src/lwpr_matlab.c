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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

const char *MODEL_FIELD_NAMES[MODEL_FIELDS] = {
   "nIn",
   "nOut",
   "n_data",
   "mean_x",
   "var_x",
   "name",
   "diag_only",
   "meta",
   "meta_rate",
   "penalty",
   "init_alpha",
   "norm_in",
   "norm_out",
   "init_D",
   "init_M",
   "w_gen",
   "w_prune",
   "init_lambda",
   "final_lambda",
   "tau_lambda",
   "init_S2",
   "add_threshold",
   "kernel",
   "update_D",
   "sub"
};

const char *SUB_FIELD_NAMES[SUB_FIELDS] = {
   "rfs",
   "n_pruned"
};

const char *RF_FIELD_NAMES[RF_FIELDS] = {
   "D",
   "M",
   "alpha",
   "beta0",
   "beta",
   "c",
   "SXresYres",
   "SSs2",
   "SSYres",
   "SSXres",
   "U",
   "P",
   "H",
   "r",
   "h",
   "b",
   "sum_w",
   "sum_e_cv2",
   "sum_e2",
   "n_data",
   "trustworthy",
   "lambda",
   "mean_x",
   "var_x",
   "w",
   "s",
   "SSp",
};


double get_scalar_field(const mxArray *S,int num, const char *name) {
   const mxArray *ar; 
   ar = mxGetField(S,num,name); 
   if (ar == NULL || 1!=mxGetM(ar) || 1!=mxGetN(ar)) {
      printf("Tried to get scalar element of field '%s'.\n",name);
      mexErrMsgTxt("Field is missing or has wrong size.");
   }   
   return *mxGetPr(ar);
}

void get_field(const mxArray *S,int num, const char *name,int m,int n, double *dest) {
   const mxArray *ar; 
   ar = mxGetField(S,num,name); 
   if (ar == NULL || m!=mxGetM(ar) || n!=mxGetN(ar)) {
      printf("Tried to get %ix%i elements of field '%s'.\n",m,n,name);
      mexErrMsgTxt("Field is missing or has wrong size.");
   }   
   if (n==1) {
      memcpy(dest,mxGetPr(ar),m*sizeof(double)); 
   } else {
      const double *src;
      int i,ld = m;
      
      if (ld&1) ld++;
      src = mxGetPr(ar);
      for (i=0;i<n;i++, dest+=ld, src+=m) memcpy(dest,src,m*sizeof(double));
   }
}

void set_scalar_field(mxArray *S,int num, int numField, double value) {
   mxArray *ar = mxCreateDoubleScalar(value);
	mxSetFieldByNumber(S,num,numField,ar);
}

void set_field(mxArray *S,int num, int numField, int m, int n, const double *src) {
   mxArray *ar = mxCreateDoubleMatrix(m,n,mxREAL);
   if (n==1) {
      memcpy(mxGetPr(ar),src,m*sizeof(double)); 
   } else {
      double *dest;
      int i,ld = m;

      if (ld&1) ld++; 
      dest = mxGetPr(ar);
      for (i=0;i<n;i++,dest+=m,src+=ld) memcpy(dest,src,m*sizeof(double));
   }
	mxSetFieldByNumber(S,num,numField,ar);
}

void create_RF_from_matlab(LWPR_ReceptiveField *RF, const LWPR_Model *model, const mxArray *S, int num) {
   int nIn,nReg;
   const mxArray *ar;
   double trust;
   
   ar = mxGetField(S,num,"U");
   if (ar==NULL) mexErrMsgTxt("RF does not contain element 'U'.");
   nIn = model->nIn;
   nReg = mxGetN(ar);
   
   if (!lwpr_mem_alloc_rf(RF, model, nReg, nReg)) mexErrMsgTxt("Out of memory: Couldn't allocate RF.");
   
   /* Note that lwpr_mem_alloc_rf   will have set RF->slopeReady = 0
   ** RF->slope is not part of the MATLAB implementation */
   
   get_field(S,num,"D",nIn,nIn,RF->D);
   get_field(S,num,"M",nIn,nIn,RF->M);
   get_field(S,num,"alpha",nIn,nIn,RF->alpha);
   get_field(S,num,"beta0",1,1,&RF->beta0);
   get_field(S,num,"beta",nReg,1,RF->beta);
   get_field(S,num,"c",nIn,1,RF->c);
   get_field(S,num,"SXresYres",nIn,nReg,RF->SXresYres);   
   get_field(S,num,"SSs2",nReg,1,RF->SSs2);
   get_field(S,num,"SSYres",nReg,1,RF->SSYres);
   get_field(S,num,"SSXres",nIn,nReg,RF->SSXres);
   get_field(S,num,"U",nIn,nReg,RF->U);
   get_field(S,num,"P",nIn,nReg,RF->P);
   get_field(S,num,"H",nReg,1,RF->H);
   get_field(S,num,"r",nReg,1,RF->r);
   get_field(S,num,"h",nIn,nIn,RF->h);   
   get_field(S,num,"b",nIn,nIn,RF->b);      
   get_field(S,num,"sum_w",nReg,1,RF->sum_w);   
   get_field(S,num,"sum_e_cv2",nReg,1,RF->sum_e_cv2);      
   get_field(S,num,"sum_e2",1,1,&RF->sum_e2);   
   get_field(S,num,"n_data",nReg,1,RF->n_data); 
   get_field(S,num,"trustworthy",1,1,&trust);  RF->trustworthy = (trust>0.0);
   get_field(S,num,"lambda",nReg,1,RF->lambda);      
   get_field(S,num,"mean_x",nIn,1,RF->mean_x);
   get_field(S,num,"var_x",nIn,1,RF->var_x);
   get_field(S,num,"w",1,1,&RF->w);            
   get_field(S,num,"s",nReg,1,RF->s);         
   get_field(S,num,"SSp",1,1,&RF->SSp);         
}

void fill_matlab_from_RF(LWPR_ReceptiveField *RF, mxArray *S, int num) {
   double trust = RF->trustworthy;
   int nIn = RF->model->nIn;
   int nReg = RF->nReg;
   
   set_field(S,num, 0,nIn,nIn,RF->D);
   set_field(S,num, 1,nIn,nIn,RF->M);
   set_field(S,num, 2,nIn,nIn,RF->alpha);
   set_field(S,num, 3,1,1,&RF->beta0);
   set_field(S,num, 4,nReg,1,RF->beta);
   set_field(S,num, 5,nIn,1,RF->c);
   set_field(S,num, 6,nIn,nReg,RF->SXresYres);   
   set_field(S,num, 7,nReg,1,RF->SSs2);
   set_field(S,num, 8,nReg,1,RF->SSYres);
   set_field(S,num, 9,nIn,nReg,RF->SSXres);
   set_field(S,num,10,nIn,nReg,RF->U);
   set_field(S,num,11,nIn,nReg,RF->P);
   set_field(S,num,12,nReg,1,RF->H);
   set_field(S,num,13,nReg,1,RF->r);
   set_field(S,num,14,nIn,nIn,RF->h);   
   set_field(S,num,15,nIn,nIn,RF->b);      
   set_field(S,num,16,nReg,1,RF->sum_w);   
   set_field(S,num,17,nReg,1,RF->sum_e_cv2);      
   set_field(S,num,18,1,1,&RF->sum_e2);   
   set_field(S,num,19,nReg,1,RF->n_data); 
   set_field(S,num,20,1,1,&trust);  
   set_field(S,num,21,nReg,1,RF->lambda);    
   set_field(S,num,22,nIn,1,RF->mean_x);
   set_field(S,num,23,nIn,1,RF->var_x);
   set_field(S,num,24,1,1,&RF->w);                 
   set_field(S,num,25,nReg,1,RF->s);   
   set_field(S,num,26,1,1,&RF->SSp);         
}

void model_consts_from_matlab(LWPR_Model *model, const mxArray *S) {
   double val;
   
   model->nIn = (int) get_scalar_field(S,0,"nIn");
   model->nInStore = (model->nIn&1) ? (model->nIn+1) : model->nIn;
      
   model->nOut = (int) get_scalar_field(S,0,"nOut");
   model->n_data = (int) get_scalar_field(S,0,"n_data");
   
   /* mean_x, var_x, name ignored */
   
   val = get_scalar_field(S,0,"diag_only");
   model->diag_only = (val!=0.0);
   
   val = get_scalar_field(S,0,"update_D");
   model->update_D = (val!=0.0);

   val = get_scalar_field(S,0,"meta");
   model->meta = (val!=0.0);

   model->meta_rate = get_scalar_field(S,0,"meta_rate");
   model->penalty = get_scalar_field(S,0,"penalty");

   /* init_alpha, norm_in, norm_out, init_D, init_M ignored */

   model->w_gen = get_scalar_field(S,0,"w_gen");
   model->w_prune = get_scalar_field(S,0,"w_prune");
   
   model->init_lambda  = get_scalar_field(S,0,"init_lambda");
   model->final_lambda = get_scalar_field(S,0,"final_lambda");
   model->tau_lambda   = get_scalar_field(S,0,"tau_lambda");
   
   model->init_S2 = get_scalar_field(S,0,"init_S2");
   model->add_threshold = get_scalar_field(S,0,"add_threshold");
   
   /* .kernel */
}   

void fill_matlab_from_sub(LWPR_SubModel *sub, mxArray *S, int dim) {
   mxArray *ar;
   int i;
   
   ar = mxCreateStructMatrix(1, sub->numRFS, RF_FIELDS, RF_FIELD_NAMES);
   for (i=0;i < sub->numRFS; i++) {
      fill_matlab_from_RF(sub->rf[i],ar,i);
   }
   mxSetFieldByNumber(S,dim, 0, ar);
   set_scalar_field(S,dim, 1, (double) sub->n_pruned);
}

mxArray *create_matlab_from_model(LWPR_Model *model) {
   int nIn = model->nIn;
   int nOut = model->nOut;
   int i;
   mxArray *S;
   mxArray *sub;
   
   S = mxCreateStructMatrix(1,1, MODEL_FIELDS, MODEL_FIELD_NAMES);
   if (S==NULL) mexErrMsgTxt("Couldn't create MATLAB model structure.");
   
   set_scalar_field(S,0, 0, (double) nIn);         
   set_scalar_field(S,0, 1, (double) nOut); 
   set_scalar_field(S,0, 2, (double) model->n_data); 
   set_field(S,0, 3, nIn, 1, model->mean_x);
   set_field(S,0, 4, nIn, 1, model->var_x);

   if (model->name!=NULL) {
      mxSetFieldByNumber(S,0,5,mxCreateString(model->name));
   } else {
      mxSetFieldByNumber(S,0,5,mxCreateString(""));
   }
      
   set_scalar_field(S,0, 6, (double) model->diag_only);
   set_scalar_field(S,0, 7, (double) model->meta);
   set_scalar_field(S,0, 8, model->meta_rate);
   set_scalar_field(S,0, 9, model->penalty);
   set_field(S,0, 10, nIn, nIn, model->init_alpha);
   set_field(S,0, 11, nIn, 1, model->norm_in);
   set_field(S,0, 12, nOut, 1, model->norm_out);
   set_field(S,0, 13, nIn, nIn, model->init_D);
   set_field(S,0, 14, nIn, nIn, model->init_M);
   
   set_scalar_field(S,0, 15, model->w_gen);
   set_scalar_field(S,0, 16, model->w_prune);
   
   set_scalar_field(S,0, 17, model->init_lambda);
   set_scalar_field(S,0, 18, model->final_lambda);
   set_scalar_field(S,0, 19, model->tau_lambda);
   
   set_scalar_field(S,0, 20, model->init_S2);
   set_scalar_field(S,0, 21, model->add_threshold);

   switch(model->kernel) {
      case LWPR_GAUSSIAN_KERNEL:
         mxSetFieldByNumber(S,0, 22, mxCreateString("Gaussian"));
         break;
      case LWPR_BISQUARE_KERNEL:
         mxSetFieldByNumber(S,0, 22, mxCreateString("BiSquare"));
         break;
   }
   
   set_scalar_field(S,0, 23, (double) model->update_D); 
   
   sub = mxCreateStructMatrix(nOut,1, SUB_FIELDS, SUB_FIELD_NAMES);
   mxSetFieldByNumber(S,0,24,sub);
   for (i=0;i<nOut;i++) {
      fill_matlab_from_sub(&(model->sub[i]), sub, i);
   }
   return S;
}


void create_model_from_matlab(LWPR_Model *model, const mxArray *S) {
   int nIn,nOut;
   int i;
   const mxArray *ar;
   const char KERN_MSG[]="Field 'kernel' is missing or invalid.";
   
   model_consts_from_matlab(model,S);
   nIn = model->nIn;
   nOut = model->nOut;
     
   /* First allocate space for nOut SubModels and 64 pointers to RFs in each */
   lwpr_mem_alloc_model(model,nIn,nOut,64);
   
   get_field(S,0,"mean_x",nIn,1,model->mean_x);
   get_field(S,0,"var_x",nIn,1,model->var_x);
   
   ar = mxGetField(S,0,"name");
   i = (mxGetM(ar) * mxGetN(ar) * sizeof(mxChar)) + 1;
   model->name = (char *) mxMalloc(i);
   if (model->name == NULL) mexErrMsgTxt("Couldn't allocate memory for model.name");
   mxGetString(ar,model->name,i);
   
   get_field(S,0,"init_D",nIn,nIn,model->init_D);
   get_field(S,0,"init_M",nIn,nIn,model->init_M);
   get_field(S,0,"init_alpha",nIn,nIn,model->init_alpha);            
   get_field(S,0,"norm_in",nIn,1,model->norm_in);            
   get_field(S,0,"norm_out",nOut,1,model->norm_out);            
   
   ar = mxGetField(S,0,"kernel");
   if (ar==NULL || !mxIsChar(ar)) mexErrMsgTxt(KERN_MSG);
   switch(*mxGetChars(ar)) {
      case 'G':
         model->kernel = LWPR_GAUSSIAN_KERNEL;
         break;
      case 'B':
         model->kernel = LWPR_BISQUARE_KERNEL;
         break;
      default:
         mexErrMsgTxt(KERN_MSG);
   }
   
   for (i=0;i<nOut;i++) {
      const mxArray *sub = mxGetField(S,0,"sub");
      create_sub_from_matlab(&(model->sub[i]), sub, i);
   }
}

void create_sub_from_matlab(LWPR_SubModel *sub, const mxArray *S, int dim) {
   mxArray *ar;
   int i,numRFS;
   
   sub->n_pruned = (int) get_scalar_field(S,dim,"n_pruned");

   ar = mxGetField(S,dim,"rfs");
   numRFS = mxGetN(ar);
   
   for (i=0;i<numRFS;i++) {
      LWPR_ReceptiveField *RF;
      
      /* do not allocate internal space yet, this is done in create_RF_from_matlab */
      RF = lwpr_aux_add_rf(sub, 0); 
      if (RF == NULL) {
         printf("Couldn't allocate space for RF %i/%i\n",dim,i);
         mexErrMsgTxt("Error: see above.");
      }
      create_RF_from_matlab(RF, sub->model, ar, i);
   }
}


LWPR_Model *get_pointer_from_array(const mxArray *A) {
   LWPR_Model **address;
   
   if (sizeof(void *)==4) {
      if (!mxIsUint32(A)) return NULL;
   } else if (sizeof(void *)==8) {
      if (!mxIsUint64(A)) return NULL;
   } else {
      return NULL;
   }
   address = (LWPR_Model **) mxGetData(A);
   return *address;
}

mxArray *create_array_from_pointer(LWPR_Model *model) {
   static int dims[2]={1,1};
   LWPR_Model **address;
   mxArray *A;
   
   if (sizeof(void *)==4) {
      A = mxCreateNumericArray(2,dims,mxUINT32_CLASS, mxREAL);
   } else if (sizeof(void *)==8) {
      A = mxCreateNumericArray(2,dims,mxUINT64_CLASS, mxREAL);   
   } else {
      return NULL;
   }
   if (A == NULL) return NULL;
   address = (LWPR_Model **) mxGetData(A);
   *address = model;
   return A;
}
