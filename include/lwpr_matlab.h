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


#ifndef __LWPR_MATLAB_H
#define __LWPR_MATLAB_H

#ifndef MATLAB
#define MATLAB
#endif

#include <mex.h>
#include <lwpr.h>
#include <lwpr_aux.h>
#include <lwpr_mem.h>

#define RF_FIELDS     27
#define SUB_FIELDS     2
#define MODEL_FIELDS  25

extern const char *RF_FIELD_NAMES[RF_FIELDS];
extern const char *SUB_FIELD_NAMES[SUB_FIELDS];
extern const char *MODEL_FIELD_NAMES[MODEL_FIELDS];

double get_scalar_field(const mxArray *S,int num, const char *name);
void set_scalar_field(mxArray *S,int num, int numField, double value);

void get_field(const mxArray *S,int num, const char *name,int m,int n, double *dest);
void set_field(mxArray *S,int num, int numField, int m, int n, const double *src);
void create_RF_from_matlab(LWPR_ReceptiveField *RF, const LWPR_Model *model, const mxArray *S, int num);
void fill_matlab_from_RF(LWPR_ReceptiveField *RF, mxArray *S, int num);
void model_consts_from_matlab(LWPR_Model *model, const mxArray *S);

void fill_matlab_from_sub(LWPR_SubModel *sub, mxArray *S, int dim);
void create_sub_from_matlab(LWPR_SubModel *sub, const mxArray *S, int dim);
void create_model_from_matlab(LWPR_Model *model, const mxArray *S);
mxArray *create_matlab_from_model(LWPR_Model *model);

LWPR_Model *get_pointer_from_array(const mxArray *A);
mxArray *create_array_from_pointer(LWPR_Model *model);

#endif
