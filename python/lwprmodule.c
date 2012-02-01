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

#include <Python.h>
#include <lwpr.h>
#include <lwpr_math.h>
#include <lwpr_xml.h>
#include <lwpr_binio.h>
#include <numpy/arrayobject.h>

typedef struct {
    PyObject_HEAD
    LWPR_Model model;
    double *extra_in;
    double *extra_out;
    double *extra_out2;    
    double *extra_out3;
    double *extra_J;
} PyLWPR;

static const char *TrueFalse[]={"False","True"};
static const char *GaussBiSq[]={"Gaussian","BiSquare"};

static void PyLWPR_dealloc(PyLWPR* self) {
   lwpr_free_model(&self->model);
   free(self->extra_in);
   self->ob_type->tp_free((PyObject*)self);
}

static PyObject *get_array_from_vector(int n, const double *data) {
   npy_intp len = n;
   PyArrayObject *vecout = (PyArrayObject *) PyArray_SimpleNew(1, &len, NPY_DOUBLE);
   memcpy(vecout->data, data, sizeof(double) * n);
   return PyArray_Return(vecout);
}

static PyObject *get_array_from_matrix(int m, int ms, int n, const double *data) {
   int i;
   npy_intp dims[2];
   PyArrayObject *matout;
   
   dims[0] = m;
   dims[1] = n;
   matout = (PyArrayObject *) PyArray_NewFromDescr(&PyArray_Type,
      PyArray_DescrFromType(NPY_DOUBLE), 2, dims, NULL, NULL, 1, NULL);
      
   for (i=0;i<n;i++) {
      memcpy(matout->data + i*PyArray_STRIDE(matout,1), data + i*ms, sizeof(double) * m);   
   }
   return PyArray_Return(matout);
}

static int set_vector_from_array(int n, double *dest, PyArrayObject *obj) {
   int i;
   if (PyArray_DESCR(obj) != PyArray_DescrFromType(NPY_DOUBLE)) {
      PyErr_SetString(PyExc_TypeError, "Expected a double precision numpy array.");
      return -1;
   }
   if (PyArray_NDIM(obj) == 1) {
      if (PyArray_DIM(obj,0) !=n) {
         PyErr_SetString(PyExc_TypeError, "Invalid number of elements.");
         return -1;
      }
      for (i=0;i<n;i++) {
         char *data = PyArray_BYTES(obj) + PyArray_STRIDE(obj,0)*i;
         dest[i] = *((double *) data);
      }
      return 0;
   }
   
   if (PyArray_NDIM(obj) == 2) {
      if (PyArray_DIM(obj,0)==n && PyArray_DIM(obj,1)==1) {
         for (i=0;i<n;i++) {
            char *data = PyArray_BYTES(obj) + PyArray_STRIDE(obj,0)*i;
            dest[i] = *((double *) data);
         }
         return 0;
      }
      if (PyArray_DIM(obj,0)==1 && PyArray_DIM(obj,1)==n) {
         for (i=0;i<n;i++) {
            char *data = PyArray_BYTES(obj) + PyArray_STRIDE(obj,1)*i;
            dest[i] = *((double *) data);
         }
         return 0;
      }
      PyErr_SetString(PyExc_TypeError, "Invalid number of elements.");
      return -1;
   }
   
   PyErr_SetString(PyExc_TypeError, "Expected a one-dimensional numpy array.");
   return -1;  
}

static int set_matrix_from_array(int m,int ms,int n, double *dest, PyArrayObject *obj) {
   int i,j;
   if (PyArray_DESCR(obj) != PyArray_DescrFromType(NPY_DOUBLE)) {
      PyErr_SetString(PyExc_TypeError, "Expected a double precision numpy array.");
      return -1;
   }
   if (PyArray_NDIM(obj) != 2) {
      PyErr_SetString(PyExc_TypeError, "Expected a two-dimensional numpy array (matrix).");
      return -1;
   }
   if (PyArray_DIM(obj,0) !=m || PyArray_DIM(obj,1) !=n ) {
      PyErr_SetString(PyExc_TypeError, "Number of columns or rows do not match.");
      return -1;
   }
   
   for (j=0;j<n;j++) {
      for (i=0;i<m;i++) {
         char *data = PyArray_BYTES(obj) + PyArray_STRIDE(obj,0)*i + PyArray_STRIDE(obj,1)*j;
         dest[i+j*ms] = *((double *) data);
      }
   }
   return 0;
}

/** Getters for scalar parameters *************************************************/
static PyObject *PyLWPR_G_nIn(PyLWPR *self, void *closure) { return Py_BuildValue("i",self->model.nIn); }
static PyObject *PyLWPR_G_nOut(PyLWPR *self, void *closure) { return Py_BuildValue("i",self->model.nOut); }
static PyObject *PyLWPR_G_n_data(PyLWPR *self, void *closure) { return Py_BuildValue("i",self->model.n_data); }
static PyObject *PyLWPR_G_meta(PyLWPR *self, void *closure) { return PyBool_FromLong(self->model.meta); }
static PyObject *PyLWPR_G_diag_only(PyLWPR *self, void *closure) { return PyBool_FromLong(self->model.diag_only); }
static PyObject *PyLWPR_G_update_D(PyLWPR *self, void *closure) { return PyBool_FromLong(self->model.update_D); }
static PyObject *PyLWPR_G_w_prune(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.w_prune); }
static PyObject *PyLWPR_G_w_gen(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.w_gen); }
static PyObject *PyLWPR_G_meta_rate(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.meta_rate); }
static PyObject *PyLWPR_G_penalty(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.penalty); }
static PyObject *PyLWPR_G_init_S2(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.init_S2); }
static PyObject *PyLWPR_G_init_lambda(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.init_lambda); }
static PyObject *PyLWPR_G_tau_lambda(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.tau_lambda); }
static PyObject *PyLWPR_G_final_lambda(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.final_lambda); }
static PyObject *PyLWPR_G_add_threshold(PyLWPR *self, void *closure) { return PyFloat_FromDouble(self->model.add_threshold); }


/** Getter for vector & matrix parameters *****************************************/
static PyObject *PyLWPR_G_norm_in(PyLWPR *self, void *closure) { 
   return get_array_from_vector(self->model.nIn, self->model.norm_in);
}

static PyObject *PyLWPR_G_norm_out(PyLWPR *self, void *closure) { 
   return get_array_from_vector(self->model.nOut, self->model.norm_out);
}

static PyObject *PyLWPR_G_mean_x(PyLWPR *self, void *closure) { 
   return get_array_from_vector(self->model.nIn, self->model.mean_x);
}

static PyObject *PyLWPR_G_var_x(PyLWPR *self, void *closure) { 
   return get_array_from_vector(self->model.nIn, self->model.var_x);
}

static PyObject *PyLWPR_G_init_D(PyLWPR *self, void *closure) { 
   return get_array_from_matrix(self->model.nIn, self->model.nInStore, self->model.nIn, self->model.init_D);
}

static PyObject *PyLWPR_G_init_M(PyLWPR *self, void *closure) { 
   return get_array_from_matrix(self->model.nIn, self->model.nInStore, self->model.nIn, self->model.init_M);
}

static PyObject *PyLWPR_G_init_alpha(PyLWPR *self, void *closure) { 
   return get_array_from_matrix(self->model.nIn, self->model.nInStore, self->model.nIn, self->model.init_alpha);
}

/**  "Getter" for num_rfs and n_pruned ****************************************/
static PyObject *PyLWPR_G_num_rfs(PyLWPR *self, void *closure) {
   int i;
   PyArrayObject *matout;
   npy_intp len = self->model.nOut;
   
   matout = (PyArrayObject *) PyArray_SimpleNew(1, &len, NPY_INT);
      
   for (i=0;i<len;i++) {
      *((int *) PyArray_GETPTR1(matout, i)) = self->model.sub[i].numRFS; 
   }
   return PyArray_Return(matout);
}

static PyObject *PyLWPR_G_n_pruned(PyLWPR *self, void *closure) {
   int i;
   PyArrayObject *matout;
   npy_intp len = self->model.nOut;
   
   matout = (PyArrayObject *) PyArray_SimpleNew(1, &len, NPY_INT);

   for (i=0;i<len;i++) {
      *((int *) PyArray_GETPTR1(matout, i)) = self->model.sub[i].n_pruned;
   }
   return PyArray_Return(matout);
}

/**  "Getter and Setter" for kernel ***********************************************/
static PyObject *PyLWPR_G_kernel(PyLWPR *self, void *closure) {
   int num = self->model.kernel == LWPR_BISQUARE_KERNEL ? 1:0;
   return PyString_FromString(GaussBiSq[num]);
}

static int PyLWPR_S_kernel(PyLWPR *self, PyObject *value, void *closure) {
   const char *str;
   if (!PyString_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'kernel' must be a string (either 'Gaussian' or 'BiSquare').");
      return -1;
   }
   str = PyString_AsString(value);
   if (!strcasecmp(str,"Gaussian")) {
      self->model.kernel = LWPR_GAUSSIAN_KERNEL;
   } else if (!strcasecmp(str,"BiSquare")) {
      self->model.kernel = LWPR_BISQUARE_KERNEL;
   } else {
      PyErr_SetString(PyExc_TypeError, "Attribute 'kernel' must be either 'Gaussian' or 'BiSquare'.");
      return -1;
   }
   return 0;
}


/** Setters ***********************************************************************/
#define CHECK_DELETE(value, attr) \
   if ((value)==NULL) {\
      PyErr_SetString(PyExc_TypeError, "Cannot delete attribute '" attr "'");\
      return -1;\
   }   

#define CHECK_BOOL(value, attr) \
   if (!PyBool_Check(value)) {\
      PyErr_SetString(PyExc_TypeError, "Attribute '" attr "' must be a boolean.");\
      return -1;\
   }\
   
#define CHECK_GET_SCALAR(value, attr, dest) \
   if (PyFloat_Check(value)) {\
      dest = PyFloat_AsDouble(value);\
   } else if (PyInt_Check(value)) {\
      dest = PyInt_AsLong(value);\
   } else {\
      PyErr_SetString(PyExc_TypeError, "Attribute '" attr "' must be a number.");\
      return -1;\
   }\

static int PyLWPR_S_meta(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"meta");
   CHECK_BOOL(value,"meta");
   self->model.meta = (value == Py_True) ? 1 : 0;
   return 0;
}

static int PyLWPR_S_diag_only(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"diag_only");
   CHECK_BOOL(value,"diag_only");
   self->model.diag_only = (value == Py_True) ? 1 : 0;
   return 0;
}

static int PyLWPR_S_update_D(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"update_D");
   CHECK_BOOL(value,"update_D");
   self->model.update_D = (value == Py_True) ? 1 : 0;
   return 0;
}

static int PyLWPR_S_w_prune(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"w_prune");
   CHECK_GET_SCALAR(value,"w_prune",self->model.w_prune);
   return 0;
}

static int PyLWPR_S_w_gen(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"w_gen");
   CHECK_GET_SCALAR(value,"w_gen",self->model.w_gen);
   return 0;
}

static int PyLWPR_S_meta_rate(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"meta_rate");
   CHECK_GET_SCALAR(value,"meta_rate",self->model.meta_rate);
   return 0;
}

static int PyLWPR_S_penalty(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"penalty");
   CHECK_GET_SCALAR(value,"penalty",self->model.penalty);
   return 0;
}

static int PyLWPR_S_init_S2(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"init_S2");
   CHECK_GET_SCALAR(value,"init_S2",self->model.init_S2);
   return 0;
}

static int PyLWPR_S_init_lambda(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"init_lambda");
   CHECK_GET_SCALAR(value,"init_lambda",self->model.init_lambda);
   return 0;
}

static int PyLWPR_S_tau_lambda(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"tau_lambda");
   CHECK_GET_SCALAR(value,"tau_lambda",self->model.tau_lambda);
   return 0;
}

static int PyLWPR_S_final_lambda(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"final_lambda");
   CHECK_GET_SCALAR(value,"final_lambda",self->model.final_lambda);
   return 0;
}

static int PyLWPR_S_add_threshold(PyLWPR *self, PyObject *value, void *closure) {
   CHECK_DELETE(value,"add_threshold");
   CHECK_GET_SCALAR(value,"add_threshold",self->model.add_threshold);
   return 0;
}

static int PyLWPR_S_init_D(PyLWPR *self,PyObject *value, void *closure) { 
   int err;
   LWPR_Model *m = &(self->model);
   if (!PyArray_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'init_D' must be a numpy array.");
      return -1;
   }
   /* First, set init_M to the matrix, and do a Cholesky decomposition in place
   ** If this fails, keep and decompose the original init_D */
   err = set_matrix_from_array(m->nIn, m->nInStore, m->nIn, m->init_M, (PyArrayObject *)value);
   if (err) {
      lwpr_math_cholesky(m->nIn, m->nInStore, m->init_M, m->init_D);
      return -1;
   }                            
   if (!lwpr_math_cholesky(m->nIn, m->nInStore, m->init_M, NULL)) {
      /* Revert to original init_M */
      lwpr_math_cholesky(m->nIn, m->nInStore, m->init_M, m->init_D);   
      PyErr_SetString(PyExc_ValueError, "'init_D' must be a positive definite matrix.");   
      return -1;
   }
   /* Ok, everything was fine, init_M is already the factor, 
      copy the contents again to init_D */
   set_matrix_from_array(m->nIn, m->nInStore, m->nIn, m->init_D, (PyArrayObject *)value);
   return 0;
}

static int PyLWPR_S_init_M(PyLWPR *self,PyObject *value, void *closure) { 
   int err;
   int i,j;
   LWPR_Model *m = &(self->model);   
   int nIn = m->nIn;
   int nInS = m->nInStore;
   
   if (!PyArray_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'init_M' must be a numpy array.");
      return -1;
   }
   err = set_matrix_from_array(nIn, nInS, nIn, m->init_M, (PyArrayObject *)value);
   if (err) {
      /* There was a problem with the matrix, revert to original cholesky factor of init_D */
      lwpr_math_cholesky(nIn, nInS, m->init_M, m->init_D);
      return -1;
   } 
   for (j=0;j<nIn;j++) {
      for (i=j+1;i<nIn;i++) {
         if (m->init_M[i+j*nInS] != 0.0) {
            PyErr_SetString(PyExc_ValueError, "Attribute 'init_M' must be upper triangular.");
            return -1;
         }
      }
   }
   /* Everything seems to be ok, update init_D from init_M */
   for (j=0;j<nIn;j++) {   
      /* Calculate in lower triangle, fill upper */
      for (i=0;i<j;i++) {   
         m->init_D[i+j*nInS] = m->init_D[j+i*nInS];
      }
      for (i=j;i<nIn;i++) {
         m->init_D[i+j*nInS] = lwpr_math_dot_product(m->init_M + i*nInS, m->init_M + j*nInS,j+1);
      }
   }
   return 0;
}

static int PyLWPR_S_init_alpha(PyLWPR *self,PyObject *value, void *closure) { 
   if (!PyArray_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'init_alpha' must be a numpy array.");
      return -1;
   }
   return set_matrix_from_array(self->model.nIn, self->model.nInStore, self->model.nIn, 
                                 self->model.init_alpha, (PyArrayObject *)value);
}

static int PyLWPR_S_norm_in(PyLWPR *self,PyObject *value, void *closure) { 
   if (!PyArray_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'norm_in' must be a numpy array.");
      return -1;
   }
   return set_vector_from_array(self->model.nIn, self->model.norm_in, (PyArrayObject *)value);
}

static int PyLWPR_S_norm_out(PyLWPR *self,PyObject *value, void *closure) { 
   if (!PyArray_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "Attribute 'norm_out' must be a numpy array.");
      return -1;
   }
   return set_vector_from_array(self->model.nOut, self->model.norm_out, (PyArrayObject *)value);
}

static PyGetSetDef PyLWPR_getseters[] = {
   {"nIn", (getter) PyLWPR_G_nIn, NULL, 
      "Input dimension", NULL},

   {"nOut", (getter) PyLWPR_G_nOut, NULL, 
      "Output dimension", NULL},
      
   {"n_data", (getter) PyLWPR_G_n_data, NULL, 
      "Number of training data the model has seen", NULL},

   {"meta", (getter) PyLWPR_G_meta, (setter) PyLWPR_S_meta, 
      "Enable meta learning (2nd order distance metric updates)", NULL},

   {"diag_only", (getter) PyLWPR_G_diag_only, (setter) PyLWPR_S_diag_only,
      "Limit distance metrics to be diagonal", NULL},

   {"update_D", (getter) PyLWPR_G_update_D, (setter) PyLWPR_S_update_D, 
      "Enable distance metric updates", NULL},        
            
   {"w_prune", (getter) PyLWPR_G_w_prune, (setter) PyLWPR_S_w_prune, 
      "Threshold parameter for pruning receptive fields", NULL},        

   {"w_gen", (getter) PyLWPR_G_w_gen, (setter) PyLWPR_S_w_gen, 
      "Threshold parameter for adding new receptive fields", NULL},        

   {"meta_rate", (getter) PyLWPR_G_meta_rate, (setter) PyLWPR_S_meta_rate, 
      "Learning rate for 2nd order distance metric updates", NULL},        
      
   {"penalty", (getter) PyLWPR_G_penalty, (setter) PyLWPR_S_penalty, 
      "Pre-factor for regularisation term in distance metric updates", NULL},        

   {"init_S2", (getter) PyLWPR_G_init_S2, (setter) PyLWPR_S_init_S2, 
      "Initial value for sufficient statistics SSs2", NULL},        

   {"add_threshold", (getter) PyLWPR_G_add_threshold, (setter) PyLWPR_S_add_threshold, 
      "Threshold parameter determining when to add a new PLS regression axis", NULL},        

   {"init_lambda", (getter) PyLWPR_G_init_lambda, (setter) PyLWPR_S_init_lambda, 
      "Initial forgetting factor", NULL},        

   {"tau_lambda", (getter) PyLWPR_G_tau_lambda, (setter) PyLWPR_S_tau_lambda, 
      "Determines annealing rate for forgetting factor", NULL},        

   {"final_lambda", (getter) PyLWPR_G_final_lambda, (setter) PyLWPR_S_final_lambda, 
      "Final forgetting factor", NULL},        
      
   {"norm_in", (getter) PyLWPR_G_norm_in, (setter) PyLWPR_S_norm_in, 
      "Input normalisation factors", NULL},        
      
   {"norm_out", (getter) PyLWPR_G_norm_out, (setter) PyLWPR_S_norm_out, 
      "Output normalisation factors", NULL},        

   {"mean_x", (getter) PyLWPR_G_mean_x, NULL, 
      "Mean of training data (inputs)", NULL},        

   {"var_x", (getter) PyLWPR_G_var_x, NULL, 
      "Variance of training data (inputs)", NULL},        
   
   {"init_D", (getter) PyLWPR_G_init_D, (setter) PyLWPR_S_init_D, 
      "Initial distance metric", NULL},           
      
   {"init_M", (getter) PyLWPR_G_init_M, (setter) PyLWPR_S_init_M, 
      "Initial distance metric", NULL},           
      
   {"init_alpha", (getter) PyLWPR_G_init_alpha, (setter) PyLWPR_S_init_alpha, 
      "Initial distance update learning rate", NULL},  
      
   {"num_rfs", (getter) PyLWPR_G_num_rfs, NULL,
      "Number of receptive fields (per output dimension)", NULL},         
      
   {"n_pruned", (getter) PyLWPR_G_n_pruned, NULL,
      "Number of receptive fields pruned during training", NULL},   
   
   {"kernel", (getter) PyLWPR_G_kernel, (setter) PyLWPR_S_kernel,
      "Kernel function used within receptive fields", NULL},      
      
   {NULL}  
};

static PyObject *PyLWPR_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
   PyLWPR *self;                                                 
   int nIn,nOut;                                                 
   if (PyTuple_Size(args)==1) {                                  
      char *filename;                                            
      int ok;
                                                                 
      if (!PyArg_ParseTuple(args, "s", &filename)) return NULL;  

      self = (PyLWPR *)type->tp_alloc(type, 0);                  
                                                                 
      ok = lwpr_read_binary(&(self->model), filename);           

#if HAVE_LIBEXPAT
      if (!ok) {
         int numErrs, numWarnings;                              
         
         numErrs = lwpr_read_xml(&(self->model), filename, &numWarnings);
         if (numErrs != 0) {
            PyErr_SetString(PyExc_IOError, "Binary or XML file could not be read or parsed correctly.");
            Py_DECREF(self);
            return NULL;
         }
      }          
#else
      if (!ok) {
         PyErr_SetString(PyExc_IOError, "Binary LWPR file could not be read correctly.");
         Py_DECREF(self);
         return NULL;
      }
#endif      
      nIn = self->model.nIn;
      nOut = self->model.nOut;
   } else {
      if (!PyArg_ParseTuple(args, "ii", &nIn,&nOut)) return NULL;
   
      self = (PyLWPR *)type->tp_alloc(type, 0);

      lwpr_init_model(&self->model, nIn, nOut, NULL);
   }
    
   self->extra_in = malloc(sizeof(double) * (nIn*(nOut +1) + 3*nOut));
   self->extra_out = self->extra_in + nIn;
   self->extra_out2 = self->extra_out + nOut;
   self->extra_out3 = self->extra_out2 + nOut;
   self->extra_J = self->extra_out3 + nOut;

   return (PyObject *)self;
}

static PyObject *PyLWPR_repr(PyLWPR *obj) {
   LWPR_Model *m = &(obj->model);
   char str[1001];
   
   snprintf(str,1000,
      "LWPR model\n"
      "          nIn : %d\n"
      "         nOut : %d\n"
      "       n_data : %d\n"
      "      penalty : %g\n"
      "      init_S2 : %g\n"
      "      w_prune : %g\n"
      "        w_gen : %g\n"
      "    diag_only : %s\n"
      "     update_D : %s\n"      
      "         meta : %s\n"      
      "    meta_rate : %g\n"
      "  init_lambda : %g\n"
      " final_lambda : %g\n"
      "   tau_lambda : %g\n"
      "add_threshold : %g\n"
      "       kernel : %s\n"
      "(+ norm_in, norm_out, init_M, init_D, init_alpha, mean_x, var_x, num_rfs)\n",
         m->nIn, m->nOut, m->n_data, 
         m->penalty, m->init_S2, m->w_prune, m->w_gen, 
         TrueFalse[m->diag_only], TrueFalse[m->update_D], TrueFalse[m->meta],
         m->meta_rate, m->init_lambda, m->final_lambda, m->tau_lambda, 
         m->add_threshold, GaussBiSq[m->kernel==LWPR_BISQUARE_KERNEL?1:0]);
         
   return PyString_FromString(str);
}

static PyObject *PyLWPR_update(PyLWPR *self, PyObject *args) {
   LWPR_Model *model = &(self->model);
   PyArrayObject *x, *y;
   if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &y))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;
   if (set_vector_from_array(model->nOut, self->extra_out, y)) return NULL;   
   
   lwpr_update(model,self->extra_in, self->extra_out, self->extra_out2, NULL);
   
   return get_array_from_vector(model->nOut, self->extra_out2);
}


static PyObject *PyLWPR_update_maxw(PyLWPR *self, PyObject *args) {
   LWPR_Model *model = &(self->model);
   PyArrayObject *x, *y;
   PyObject *o1,*o2,*result;
   
   if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &y))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;
   if (set_vector_from_array(model->nOut, self->extra_out, y)) return NULL;   
   
   lwpr_update(model,self->extra_in, self->extra_out, self->extra_out2, self->extra_out3);
   
   o1 = get_array_from_vector(model->nOut, self->extra_out2);
   o2 = get_array_from_vector(model->nOut, self->extra_out3);

   result = Py_BuildValue("(O,O)",o1,o2);

   Py_DECREF(o1);
   Py_DECREF(o2);

   return result;
}


static PyObject *PyLWPR_predict(PyLWPR *self, PyObject *args) {
   double cutoff = 0.0;
   LWPR_Model *model = &(self->model);
   PyArrayObject *x;

   if (!PyArg_ParseTuple(args, "O!|d", &PyArray_Type, &x, &cutoff))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;

   lwpr_predict(model,self->extra_in, cutoff, self->extra_out, NULL, NULL);

   return get_array_from_vector(model->nOut, self->extra_out);
}

static PyObject *PyLWPR_predict_conf(PyLWPR *self, PyObject *args) {
   double cutoff = 0.0;
   LWPR_Model *model = &(self->model);
   PyArrayObject *x;
   PyObject *o1,*o2,*result;

   if (!PyArg_ParseTuple(args, "O!|d", &PyArray_Type, &x, &cutoff))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;

   lwpr_predict(model,self->extra_in, cutoff, self->extra_out, self->extra_out2, NULL);

   o1 = get_array_from_vector(model->nOut, self->extra_out);
   o2 = get_array_from_vector(model->nOut, self->extra_out2);

   result = Py_BuildValue("(O,O)",o1,o2);

   Py_DECREF(o1);
   Py_DECREF(o2);

   return result;
}

static PyObject *PyLWPR_predict_conf_maxw(PyLWPR *self, PyObject *args) {
   double cutoff = 0.0;
   LWPR_Model *model = &(self->model);
   PyArrayObject *x;
   PyObject *o1,*o2,*o3,*result;

   if (!PyArg_ParseTuple(args, "O!|d", &PyArray_Type, &x, &cutoff))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;

   lwpr_predict(model,self->extra_in, cutoff, self->extra_out, self->extra_out2, self->extra_out3);

   o1 = get_array_from_vector(model->nOut, self->extra_out);
   o2 = get_array_from_vector(model->nOut, self->extra_out2);
   o3 = get_array_from_vector(model->nOut, self->extra_out3);

   result = Py_BuildValue("(O,O,O)",o1,o2,o3);

   Py_DECREF(o1);
   Py_DECREF(o2);
   Py_DECREF(o3);

   return result;
}

static PyObject *PyLWPR_predict_J(PyLWPR *self, PyObject *args) {
   double cutoff = 0.0;
   LWPR_Model *model = &(self->model);
   PyArrayObject *x;
   PyObject *o1,*o2,*result;

   if (!PyArg_ParseTuple(args, "O!|d", &PyArray_Type, &x, &cutoff))  return NULL;
   if (set_vector_from_array(model->nIn, self->extra_in, x)) return NULL;

   lwpr_predict_J(model,self->extra_in, cutoff, self->extra_out, self->extra_J);

   o1 = get_array_from_vector(model->nOut, self->extra_out);
   o2 = get_array_from_matrix(model->nOut, model->nOut, model->nIn, self->extra_J);

   result = Py_BuildValue("(O,O)",o1,o2);

   Py_DECREF(o1);
   Py_DECREF(o2);

   return result;
}

static PyObject *PyLWPR_rf_center(PyLWPR *self, PyObject *args) {
   int dim, n;
   LWPR_Model *model = &(self->model);

   if (!PyArg_ParseTuple(args, "ii", &dim, &n))  return NULL;
   
   if (dim<0 || dim>=model->nOut) {
      PyErr_SetString(PyExc_TypeError, "First parameter must indicate output dimension (0 <= dim < model.nOut).");
      return NULL;
   }
   
   if (n<0 || n>=model->sub[dim].numRFS) {
      PyErr_SetString(PyExc_TypeError, "Second parameter must indicate receptive field (0 <= n < model.num_rf[dim]).");
      return NULL;
   }
   
   return get_array_from_vector(model->nIn, model->sub[dim].rf[n]->c);
}

static PyObject *PyLWPR_rf_D(PyLWPR *self, PyObject *args) {
   int dim, n;
   LWPR_Model *model = &(self->model);

   if (!PyArg_ParseTuple(args, "ii", &dim, &n))  return NULL;
   
   if (dim<0 || dim>=model->nOut) {
      PyErr_SetString(PyExc_TypeError, "First parameter must indicate output dimension (0 <= dim < model.nOut).");
      return NULL;
   }
   
   if (n<0 || n>=model->sub[dim].numRFS) {
      PyErr_SetString(PyExc_TypeError, "Second parameter must indicate receptive field (0 <= n < model.num_rf[dim]).");
      return NULL;
   }
   
   return get_array_from_matrix(model->nIn, model->nInStore, model->nIn, model->sub[dim].rf[n]->D);
}

static PyObject *PyLWPR_write_XML(PyLWPR *self, PyObject *args) {
   char *filename;
   FILE *fp;
   LWPR_Model *model = &(self->model);

   if (!PyArg_ParseTuple(args, "s", &filename))  return NULL;
   fp = fopen(filename, "w");
   if (fp==NULL) {
      PyErr_SetString(PyExc_IOError, "File cannot be opened for writing.");
      return NULL;
   }
   lwpr_write_xml_fp(model, fp);
   fclose(fp);
   
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject *PyLWPR_write_binary(PyLWPR *self, PyObject *args) {
   char *filename;
   FILE *fp;
   LWPR_Model *model = &(self->model);
   int ok;

   if (!PyArg_ParseTuple(args, "s", &filename))  return NULL;
   fp = fopen(filename, "wb");
   if (fp==NULL) {
      PyErr_SetString(PyExc_IOError, "File cannot be opened for writing.");
      return NULL;
   }
   ok=lwpr_write_binary_fp(model, fp);
   fclose(fp);
   
   if (!ok) {
      PyErr_SetString(PyExc_IOError, "Errors occured during writing.");
      return NULL;
   }
   
   Py_INCREF(Py_None);
   return Py_None;
}


static PyMethodDef PyLWPR_methods[] = {
    {"update", (PyCFunction)PyLWPR_update, METH_VARARGS,
     "Update an LWPR model given an (input, output) training sample. Returns current prediction."},
    {"update_maxw", (PyCFunction)PyLWPR_update_maxw, METH_VARARGS,
     "Update an LWPR model given an (input, output) training sample. Returns current prediction and maximum activation."},
    {"predict", (PyCFunction)PyLWPR_predict, METH_VARARGS,
     "Compute prediction of LWPR model for a given input sample"},
    {"predict_conf", (PyCFunction)PyLWPR_predict_conf, METH_VARARGS,
     "Compute prediction and confidence bounds of LWPR model for a given input sample"},
    {"predict_conf_maxw", (PyCFunction)PyLWPR_predict_conf_maxw, METH_VARARGS,
     "Compute prediction, confidence bounds, and maximal activation of LWPR model for a given input sample"},
    {"predict_J", (PyCFunction)PyLWPR_predict_J, METH_VARARGS,
     "Compute prediction and Jacobi matrix of LWPR model for a given input sample"},
    {"rf_center", (PyCFunction)PyLWPR_rf_center, METH_VARARGS,
     "rf_center(dim,n) retrieves the center of the n-th receptive field in output dimension dim."},
    {"rf_D", (PyCFunction)PyLWPR_rf_D, METH_VARARGS,
     "rf_D(dim,n) retrieves the distance metric of the n-th receptive field in output dimension dim."},
    {"write_XML", (PyCFunction)PyLWPR_write_XML, METH_VARARGS,
     "write_XML(filename) writes the LWPR model to an XML file."},
    {"write_binary", (PyCFunction)PyLWPR_write_binary, METH_VARARGS,
     "write_binary(filename) writes the LWPR model to a binary, platform-dependent file."},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyLWPR_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                           /* ob_size */
    "lwpr.LWPR",                 /* tp_name */
    sizeof(PyLWPR)   ,           /* tp_basicsize */
    0,                           /* tp_itemsize */
    (destructor) PyLWPR_dealloc, /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    (reprfunc) PyLWPR_repr,      /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    0,                           /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,          /* tp_flags */
    "This class encapsulates an LWPR model for learning regression functions\n"
    "with a possibly high number of input dimensions. You can create a new\n"
    "LWPR model by something like\n"
    "   model = LWPR(12,4)   # 12 inputs, 4 outputs\n"
    "or you can read a model from an XML file\n"
    "   model = LWPR('file.xml')\n", /* tp_doc */
    0,		                     /* tp_traverse */
    0,		                     /* tp_clear */
    0,      		               /* tp_richcompare */
    0,		                     /* tp_weaklistoffset */
    0,		                     /* tp_iter */
    0,		                     /* tp_iternext */
    PyLWPR_methods,              /* tp_methods */
    0,                           /* tp_members */
    PyLWPR_getseters,            /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    0,                           /* tp_init */
    0,                           /* tp_alloc */
    PyLWPR_new                   /* tp_new */
};  

static PyMethodDef lwpr_methods[] = {{NULL}};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif

PyMODINIT_FUNC initlwpr(void) {
   PyObject* m;

   if (PyType_Ready(&PyLWPR_Type) < 0) return;

   m = Py_InitModule3("lwpr", lwpr_methods, "Python wrapper for the C implementation of LWPR.");

   Py_INCREF(&PyLWPR_Type);
   PyModule_AddObject(m, "LWPR", (PyObject *)&PyLWPR_Type);
   import_array();
}
