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
#include <string.h>
#include <stdio.h>

#define MAX_NUM_MODELS  128

static LWPR_Model *models[MAX_NUM_MODELS];

int find_empty_slot() {
   int i;
   for (i=0;i<MAX_NUM_MODELS;i++) {
      if (models[i]==NULL) return i;
   }
   return -1;
}

int find_slot_by_model(LWPR_Model *model) {
   int i;
   for (i=0;i<MAX_NUM_MODELS;i++) {
      if (models[i]==model) return i;
   }
   return -1;
}
         
         
void make_model_persistent(LWPR_Model *model) {
   int i,j;
   
   mexMakeMemoryPersistent(model);     
   if (model->name!=NULL) mexMakeMemoryPersistent(model->name);     
   mexMakeMemoryPersistent(model->storage);
   mexMakeMemoryPersistent(model->sub);
   mexMakeMemoryPersistent(model->ws);
   for (i=0;i<NUM_THREADS;i++) {
      mexMakeMemoryPersistent(model->ws[i].storage);
      mexMakeMemoryPersistent(model->ws[i].derivOk);
   }

   for (i=0;i<model->nOut;i++) {
      LWPR_SubModel *sub = &(model->sub[i]);
      mexMakeMemoryPersistent(sub->rf);
      for (j=0;j<sub->numRFS;j++) {
         mexMakeMemoryPersistent(sub->rf[j]);
         mexMakeMemoryPersistent(sub->rf[j]->fixStorage);
         mexMakeMemoryPersistent(sub->rf[j]->varStorage);         
      }
   }
   model->isPersistent = 1;   
}      

void free_all_models(void) {
   int i;
   
   for (i=0;i<MAX_NUM_MODELS;i++) {
      if (models[i]!=NULL) {
         if (models[i]->name!=NULL) {
            printf("Freeing model %s (%d -> %d)\n",models[i]->name,models[i]->nIn,models[i]->nOut);
         } else {
            printf("Freeing nameless model (%d -> %d)\n",models[i]->nIn,models[i]->nOut);         
         }
         lwpr_free_model(models[i]);
         mxFree(models[i]);
         models[i]=NULL;
      }
   }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[]) {
   char command[20];
   LWPR_Model *model;
   int num;
         
   mexAtExit(free_all_models);
      
   mxGetString(prhs[0],command,20);
   if (!strcmp(command,"Store")) {
      num = find_empty_slot();
      if (num==-1) mexErrMsgTxt("No more free model slots.\n");
      
      model = (LWPR_Model *) mxCalloc(1,sizeof(LWPR_Model));
      create_model_from_matlab(model, prhs[1]);
      make_model_persistent(model);
      models[num]=model;
      
      plhs[0]=create_array_from_pointer(model);
   } else if (!strcmp(command,"FreeAll")) {
      free_all_models();
   } else {
      model = get_pointer_from_array(prhs[1]);
      if (model==NULL) mexErrMsgTxt("2nd argument must be a valid storage ID.");
      num = find_slot_by_model(model);
      if (num==-1) mexErrMsgTxt("Sorry, could not find desired model.\n");
      
      if (!strcmp(command,"Get")) {
         plhs[0] = create_matlab_from_model(models[num]);
      } else if (!strcmp(command,"GetFree")) {
         plhs[0] = create_matlab_from_model(models[num]);
         lwpr_free_model(models[num]);
         mxFree(models[num]);
         models[num]=NULL;
      } else if (!strcmp(command,"Free")) {
         lwpr_free_model(models[num]);
         mxFree(models[num]);
         models[num]=NULL;
      } else {
         mexErrMsgTxt("Unknown command.");
      }
   }
}
