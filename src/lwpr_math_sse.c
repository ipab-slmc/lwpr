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
#include <math.h>
#include <string.h>

typedef double v2df __attribute__ ((vector_size (16)));

double lwpr_math_dot_product(const double *x, const double *y,int n) {
   v2df a,b,c,d,e;
   
   double p[2];
  
   /* e = __builtin_ia32_loadupd(x); */
   e = __builtin_ia32_subpd(e,e);
   while (n>=8) {
      a = __builtin_ia32_loadupd(x);
      b = __builtin_ia32_loadupd(y);
      c = __builtin_ia32_mulpd(a,b);
      e = __builtin_ia32_addpd(e,c);
      a = __builtin_ia32_loadupd(x+2);
      b = __builtin_ia32_loadupd(y+2);
      c = __builtin_ia32_mulpd(a,b);
      e = __builtin_ia32_addpd(e,c);
      a = __builtin_ia32_loadupd(x+4);
      b = __builtin_ia32_loadupd(y+4);
      c = __builtin_ia32_mulpd(a,b);
      e = __builtin_ia32_addpd(e,c);
      a = __builtin_ia32_loadupd(x+6);
      b = __builtin_ia32_loadupd(y+6);
      c = __builtin_ia32_mulpd(a,b);
      e = __builtin_ia32_addpd(e,c);
      n-=8;
      y+=8;
      x+=8;
   }
   while (n>=2) {
      a = __builtin_ia32_loadupd(x);
      b = __builtin_ia32_loadupd(y);
      c = __builtin_ia32_mulpd(a,b);
      e = __builtin_ia32_addpd(e,c);
      n-=2;
      y+=2;
      x+=2;
   }   
   if (n) {
      a = __builtin_ia32_loadlpd(a,x);
      b = __builtin_ia32_loadlpd(b,y); 
      c = __builtin_ia32_mulsd(a,b);
      e = __builtin_ia32_addsd(e,c);   
   }
   __builtin_ia32_storeupd (p,e);

   return p[0] + p[1];
}


void lwpr_math_scalar_vector(double *y, double alpha,const double *x,int n) {
   v2df a,b,c,d,e;
   
   a = __builtin_ia32_loadlpd(a,&alpha); 
   a = __builtin_ia32_loadhpd(a,&alpha); 
   
   while (n>=8) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_mulpd(a,b);
      d = __builtin_ia32_loadupd(x+2);      
      __builtin_ia32_storeupd(y,c);
      e = __builtin_ia32_mulpd(a,d);
      __builtin_ia32_storeupd(y+2,e);

      b = __builtin_ia32_loadupd(x+4);
      c = __builtin_ia32_mulpd(a,b);
      d = __builtin_ia32_loadupd(x+6);      
      __builtin_ia32_storeupd(y+4,c);
      e = __builtin_ia32_mulpd(a,d);
      __builtin_ia32_storeupd(y+6,e);

      n-=8;
      y+=8;
      x+=8;
   }
   while (n>=1) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_mulpd(a,b);
      __builtin_ia32_storeupd(y,c);
      n-=2;
      y+=2;
      x+=2;
   }   
   
}


void lwpr_math_add_scalar_vector(double *y, double alpha,const double *x,int n) {
   v2df a,b,c,d,e;
   
   a = __builtin_ia32_loadlpd(a,&alpha); 
   a = __builtin_ia32_loadhpd(a,&alpha); 
   
   while (n>=8) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_loadupd(y);
      d = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_addpd(d,c);
      __builtin_ia32_storeupd(y,c);
      
      b = __builtin_ia32_loadupd(x+2);
      c = __builtin_ia32_loadupd(y+2);
      d = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_addpd(d,c);
      __builtin_ia32_storeupd(y+2,c);

      b = __builtin_ia32_loadupd(x+4);
      c = __builtin_ia32_loadupd(y+4);
      d = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_addpd(d,c);
      __builtin_ia32_storeupd(y+4,c);

      b = __builtin_ia32_loadupd(x+6);
      c = __builtin_ia32_loadupd(y+6);
      d = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_addpd(d,c);
      __builtin_ia32_storeupd(y+6,c);

      n-=8;
      y+=8;
      x+=8;
   }
   while (n>=1) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_loadupd(y);
      d = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_addpd(d,c);
      __builtin_ia32_storeupd(y,c);

      n-=2;
      y+=2;
      x+=2;
   }   
}


void lwpr_math_scale_add_scalar_vector(double beta, double *y, double alpha,const double *x,int n) {
   v2df a,b,c,d,e;
   
   a = __builtin_ia32_loadlpd(a,&alpha); 
   a = __builtin_ia32_loadhpd(a,&alpha); 
   e = __builtin_ia32_loadlpd(e,&beta);
   e = __builtin_ia32_loadhpd(e,&beta);
   
   while (n>=8) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_loadupd(y);
      b = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_mulpd(e,c);
      c = __builtin_ia32_addpd(b,c);
      __builtin_ia32_storeupd(y,c);
      
      b = __builtin_ia32_loadupd(x+2);
      c = __builtin_ia32_loadupd(y+2);
      b = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_mulpd(e,c);
      c = __builtin_ia32_addpd(b,c);
      __builtin_ia32_storeupd(y+2,c);

      b = __builtin_ia32_loadupd(x+4);
      c = __builtin_ia32_loadupd(y+4);
      b = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_mulpd(e,c);
      c = __builtin_ia32_addpd(b,c);
      __builtin_ia32_storeupd(y+4,c);

      b = __builtin_ia32_loadupd(x+6);
      c = __builtin_ia32_loadupd(y+6);
      b = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_mulpd(e,c);
      c = __builtin_ia32_addpd(b,c);
      __builtin_ia32_storeupd(y+6,c);

      n-=8;
      y+=8;
      x+=8;
   }
   while (n>=1) {
      b = __builtin_ia32_loadupd(x);
      c = __builtin_ia32_loadupd(y);
      b = __builtin_ia32_mulpd(a,b);
      c = __builtin_ia32_mulpd(e,c);
      c = __builtin_ia32_addpd(b,c);
      __builtin_ia32_storeupd(y,c);

      n-=2;
      y+=2;
      x+=2;
   }   
}




int lwpr_math_cholesky(int N,int Ns,double *R,const double *A) {
	int i,j,k;
   double A_00, R_00;
   
	if (A!=NULL) {
      memcpy(R,A,N*Ns*sizeof(double));
	}
	
   A_00 = R[0];
   if (A_00 <= 0) return 0;
      
	R[0] = R_00 = sqrt(A_00);
	
   if (N > 1) {
      double A_01 = R[Ns];
      double A_11 = R[1+Ns];
      double R_01,diag;
      
      R_01 = A_01 / R_00;
      diag = A_11 - R_01 * R_01;
		
		if (diag<=0) return 0;
		
		R[Ns]=R_01;
		R[1+Ns]=sqrt(diag); /* R_11 */
		
      for (k = 2; k < N; k++) {
			double A_kk = R[k*Ns+k];
			double diag;
         
         for (i = 0; i < k; i++) {
            double sum;
            double A_ik = R[i+k*Ns];
            double A_ii = R[i+i*Ns];
				  
				sum = lwpr_math_dot_product(R+i*Ns,R+k*Ns,i);
				  
				A_ik = (A_ik-sum)/A_ii;
				R[i+k*Ns]=A_ik;
			}

         diag = A_kk - lwpr_math_dot_product(R+k*Ns,R+k*Ns,k);
		   if (diag <= 0) return 0; 
			R[k+k*Ns]=sqrt(diag); /* R_kk */
		}
	}
	
	for (j=0;j<N-1;j++) {
		for (i=j+1;i<N;i++) {
			R[i+j*Ns]=0;
		}
	}
	return 1;
}


