#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate the wall normal (y) grid
//                     and compact finite difference derivatives.
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 11/01/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++*/
static void get_LU(double *A,double *L,double *U,int a);
static void get_coef(double *cf,double *A,double *rhs,int N);
static void get_invM(double *B,double *A,int N);
static void get_matmul(double *DC,double *LH,double *RH,int N);
static void get_CD_dev1(double *DC,double *LH,double *RH,double *y);
static void get_CD_dev2(double *LH,double *RH,double *y);
static double factorial(int h);

/*+++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++*/
// INITIALIZE Y GRID......................................................./
void init_cfd(double *y_h,double *LH_1,double *RH_1,double *DC1,
		          double *LH_2,double *RH_2,double *FD_h){
  // COMPUTE THE WALL NORMAL DIRECTION GRID +++++++++++++++++++++++++++++++
  for(int i=0;i<NY;i++){
     y_h[i]=2.0*(1.0-p_y)*(0.5-0.5*cos(0.5*PI2*double(i)/double(NY-1)))
	       + 2.0*p_y*double(i)/double(NY-1) - 1.0;
  }
  // COMPUTE MATRICES OF THE FIRST DERIVATIVE ++++++++++++++++++++++
  double *LH1=(double *)malloc(NY*NY*sizeof(double)); 
  double *RH1=(double *)malloc(NY*NY*sizeof(double)); 
  get_CD_dev1(DC1,LH1,RH1,y_h); 
  // COMPUTE MATRICES OF THE FIRST DERIVATIVE ++++++++++++++++++++++    
  double *LH2=(double *)malloc(NY*NY*sizeof(double)); 
  double *RH2=(double *)malloc(NY*NY*sizeof(double));
  get_CD_dev2(LH2,RH2,y_h); 
  // SAVE TO THE SPARSE MATRICES +++++++++++++++++++++++++++++++++++
  for(int i=0;i<3*NY;i++){LH_1[i]=0.0; LH_2[i]=0.0;}
  for(int i=0;i<5*NY;i++){RH_1[i]=0.0; RH_2[i]=0.0;}
  // Initialize indexes
  int l_LH, r_LH, l_RH, r_RH, j0LH, j0RH;
  // Loop on rows:
  for(int i=0;i<NY;i++){
   // Indexes original matrix
   l_LH=max(i-1,0);r_LH=min(i+1,NY-1);l_RH=max(i-2,0);r_RH=min(i+2,NY-1);
   // Select the initial column
   j0LH=l_LH-(i-1)+r_LH-(i+1)+i-1;  j0RH=l_RH-(i-2)+r_RH-(i+2)+i-2;
   // Loop on columns of L1 and L2
   for(int j=0;j<3;j++){
     LH_1[i*3+j]=LH1[i*NY+j0LH+j]; LH_2[i*3+j]=LH2[i*NY+j0LH+j];
   }// end loop on columns of L1 and L2
   // Loop on columns of L1 and L2
   for(int j=0;j<5;j++){
     RH_1[i*5+j]=RH1[i*NY+j0RH+j]; RH_2[i*5+j]=RH2[i*NY+j0RH+j];
   }// end loop on columns of L1 and L2
  }// end loop rows
  // OUTRO
  free(LH1); free(LH2); free(RH1); free(RH2);
  // COMPUTE THE FIRST ORDER DERIVATIVE USING ANOTHER METHOD+++++++++++
  double dy1,dy2,a0,b0,c0,a1,b1,c1; 
  // Bottom wall
  dy1=y_h[1]-y_h[0];       dy2=y_h[2]-y_h[0]; 
  c0=(dy2/dy1)*(dy2+dy1); c0=1/c0;  b0=c0*dy2*dy2/(dy1*dy1); a0=-b0-c0;
  // Top wall
  dy1=y_h[NY-2]-y_h[NY-1]; dy2=y_h[NY-3]-y_h[NY-1];
  c1=(dy2/dy1)*(dy2+dy1); c1=1/c1;  b1=c1*dy2*dy2/(dy1*dy1); a1=-b1-c1;
  // Register
  FD_h[0]=a0; FD_h[1]=b0; FD_h[2]=c0; FD_h[3]=c1; FD_h[4]=b1; FD_h[5]=a1;
  // Verbose
  return;
}// END INITIALIZE Y GRID................................................../







/*+++++++++++++++++++++++++++++ STATIC FUNCTIONS +++++++++++++++++++++++++*/
// FUNCTION TO COMPUTE THE COMPACT DIFFERENCES WITH 3/5 STENCIL...(1st)..../
static void get_CD_dev1(double *DC,double *LH,double *RH,double *y){
 // INITIALIZE . . . . . . . . . . . . . . . . . . . . . . . . . . .  
 for(int i=0;i<NY*NY;i++){LH[i]=0.0; RH[i]=0.0;} double *A,*rhs,*cf;
 int l_RH,r_RH,l_LH,r_LH,lenRH,lenLH,len,cn;     double dx;
 double *LH_1=(double *)malloc(NY*NY*sizeof(double));
 // LOOP ON ROWS . . . . . . . . . . . . . . . . . . . . . . . . . .
 for(int i=0;i<NY;i++){
  // Initialize the indices of the RH components
  l_RH=max(0,i-2); r_RH=min(NY-1,i+2); lenRH=r_RH-l_RH+1; 
  // Initialize the indices of the LH components
  l_LH=max(0,i-1); r_LH=min(NY-1,i+1); lenLH=r_LH-l_LH+1; 
  // Matrix and rhs to solve Ax=b
  len=lenRH+lenLH-1;            A=(double *)malloc(len*len*sizeof(double));
  rhs=(double *)malloc(len*sizeof(double)); 
  for(int h=0;h<len;h++){rhs[h]=0.0;}
  for(int h=0;h<len*len;h++){A[h]=0.0;} rhs[1]=1.0; A[0]=1.0;  cn=0;
  // Columns of first order.........................................
  for(int j=l_RH;j<r_RH+1;j++){
  // Continue only if you are not in the column of interest 
   if(j!=i){cn+=1;  dx=y[j]-y[i];   A[cn]=1;
    // Loop on rows 
    for(int h=1;h<len;h++){A[h*len+cn]=pow(dx,double(h))/factorial(h);}   
   }// end if  
  }// end loop on columns of first order............................
  // Columns of second order........................................
  for(int j=l_LH;j<r_LH+1;j++){
  // Continue only if you are not in the position you actually want to comp 
   if(j!=i){cn+=1; dx=y[j]-y[i];   A[len+cn]=1.0;
    // Loop on rows
   for(int h=1;h<len-1;h++){A[(h+1)*len+cn]=pow(dx,double(h))/factorial(h);}    }// end if  
  }// end loop on columns of second order..........................
  // Compute the coefficients...........................................
  cf=(double *)malloc(len*sizeof(double)); get_coef(cf,A,rhs,len);
  // Record to the RH matrix............................................
  RH[i*NY+i]=cf[0]; cn=0; 
  // loop on columns
  for(int j=l_RH;j<r_RH+1;j++){if(j!=i){cn+=1; RH[i*NY+j]=cf[cn];}}
  // Record to the LH matrix............................................
  LH[i*NY+i]=1.0; 
  // Loop on columns
  for(int j=l_LH;j<r_LH+1;j++){if(j!=i){cn+=1; LH[i*NY+j]=-cf[cn];}}
 }// END LOOP ON ROWS. . . . . . . . . . . . . . . . . . . . . . . .
 // CHANGE THE FIRST & LAST ROW IF p_y==1...............................
 if(p_y>0.99){ double h=y[1]-y[0];
  // First row of LH matrix
  LH[0]=1; LH[1]=3; 
  //First row of RH matrix
  RH[0]=-17/(6*h); RH[1]=3/(2*h); RH[2]=3/(2*h); RH[3]=-1/(6*h);
  // Last row of LH matrix
  cn=NY-1;    LH[cn*NY+cn]=1;    LH[cn*NY+cn-1]=3; 
  // Last row of RH matrix
  RH[cn*NY+cn]=17/(6*h);     RH[cn*NY+cn-1]=-3/(2*h); 
  RH[cn*NY+cn-2]=-3/(2*h);   RH[cn*NY+cn-3]= 1/(6*h);
 } // end if uniform grid 
 // COMPUTE THE MATRIX DC
 get_invM(LH_1,LH,NY); get_matmul(DC,LH_1,RH,NY);
 // OUTRO  . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 free(A); free(rhs); free(LH_1);
}// END FUNCTION TO COMPUTE THE COMPACT DIFFERENCES WITH 3/5 STENCIL.(1st)./





// FUNCTION TO COMPUTE THE COMPACT DIFFERENCES WITH 3/5 STENCIL.(2nd)....../
static void get_CD_dev2(double *LH,double *RH,double *y){
 // INITIALIZE . . . . . . . . . . . . . . . . . . . . . . . . . . .
 for(int i=0;i<NY*NY;i++){LH[i]=0.0; RH[i]=0.0;} double *A,*rhs,*cf;
 int l_RH,r_RH,l_LH,r_LH,lenRH,lenLH,len,cn;     double dx;
 double *LH_1=(double *)malloc(NY*NY*sizeof(double));
 // LOOP ON ROWS . . . . . . . . . . . . . . . . . . . . . . . . . .
 for(int i=0;i<NY;i++){
  // Initialize the indices of the RH components
  l_RH=max(0,i-2); r_RH=min(NY-1,i+2); lenRH=r_RH-l_RH+1;
  // Initialize the indices of the LH components
  l_LH=max(0,i-1); r_LH=min(NY-1,i+1); lenLH=r_LH-l_LH+1;
  // Matrix and rhs to solve Ax=b
  len=lenRH+lenLH-1;            A=(double *)malloc(len*len*sizeof(double));
  rhs=(double *)malloc(len*sizeof(double));
  for(int h=0;h<len;h++){rhs[h]=0.0;}
  for(int h=0;h<len*len;h++){A[h]=0.0;} rhs[2]=1.0; A[0]=1.0;  cn=0;
  // Columns of first order.........................................
  for(int j=l_RH;j<r_RH+1;j++){
  // Continue only if you are not in the column of interest
   if(j!=i){cn+=1;  dx=y[j]-y[i];   A[cn]=1;
    // Loop on rows
    for(int h=1;h<len;h++){A[h*len+cn]=pow(dx,double(h))/factorial(h);}
   }// end if
  }// end loop on columns of first order............................
  // Columns of second order........................................
  for(int j=l_LH;j<r_LH+1;j++){
  // Continue only if you are not in the position you actually want to comp
   if(j!=i){cn+=1; dx=y[j]-y[i];   A[2*len+cn]=1.0;
    // Loop on rows
   for(int h=1;h<len-2;h++){A[(h+2)*len+cn]=pow(dx,double(h))/factorial(h);}
   }// end if
  }// end loop on columns of second order..........................
  // Compute the coefficients...........................................
  cf=(double *)malloc(len*sizeof(double)); get_coef(cf,A,rhs,len);
  // Record to the RH matrix............................................
  RH[i*NY+i]=cf[0]; cn=0;
  // loop on columns
  for(int j=l_RH;j<r_RH+1;j++){if(j!=i){cn+=1; RH[i*NY+j]=cf[cn];}}
  // Record to the LH matrix............................................
  LH[i*NY+i]=1.0;
  // Loop on columns
  for(int j=l_LH;j<r_LH+1;j++){if(j!=i){cn+=1; LH[i*NY+j]=-cf[cn];}}
 }// END LOOP ON ROWS. . . . . . . . . . . . . . . . . . . . . . . .   
  // CHANGE THE FIRST & LAST ROW IF p_y==1...............................
 if(p_y>0.99){double h=y[1]-y[0];
  // First row of LH matrix
  LH[0]=1; LH[1]=11;
  //First row of RH matrix
  RH[0]=13/(h*h); RH[1]=-27/(h*h); RH[2]=15/(h*h); RH[3]=-1/(h*h);
  // Last row of LH matrix
  cn=NY-1;    LH[cn*NY+cn]=1;    LH[cn*NY+cn-1]=11;
  // Last row of RH matrix
  RH[cn*NY+cn]=13/(h*h);     RH[cn*NY+cn-1]=-27/(h*h);
  RH[cn*NY+cn-2]=15/(h*h);   RH[cn*NY+cn-3]=-1/(h*h);
 } // end if uniform grid
 // COMPUTE THE MATRIX DC
 // get_invM(LH_1,LH,NY); get_matmul(DC,LH_1,RH,NY);    
 // OUTRO  . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
 free(A); free(rhs); free(LH_1); 
}// END FUNCTION TO COMPUTE THE COMPACT DIFFERENCES WITH 3/5 STENCIL.(2nd)./









// FUNCTION TO PERFORM MATRIX MATRIX MULTIPLICATION......................../
static void get_matmul(double *DC,double *LH,double *RH,int N){
 // Initialize
 double aux=0.0;
 // Loop on rows
 for(int i=0;i<N;i++){
   // Loop on columns
   for(int j=0;j<N;j++){aux=0.0;
     // Loop on elements on row/column
     for(int h=0;h<N;h++){aux+=LH[i*N+h]*RH[h*N+j];}
     // Register to matrix 	   
     DC[i*N+j]=aux;
   }// end loop on columns
 }// end loop on rows	 
}// END FUNCTION TO PERFORM MATRIX MATRIX MULTILPLICATION................../



// FUNCTION TO COMPUTE THE INVERSE MATRIX................................../
static void get_invM(double *B,double *A,int N){
 // Initialize
 double *rhs=(double *)malloc(N*sizeof(double));	
 double *cf=(double *)malloc(N*sizeof(double));	
 for(int i=0;i<N*N;i++){B[i]=0.0;}
 // Loop on columns of the identity matrix
 for(int i=0;i<N;i++){
  // RHS
  for(int j=0;j<N;j++){rhs[j]=0.0;}  rhs[i]=1.0;
  // Invert and save to matrix
  get_coef(cf,A,rhs,N); for(int j=0;j<N;j++){B[j*N+i]=cf[j];}
 }// end loop on columns	 
 // Free
 free(rhs); free(cf); return;
}// END FUNCTION TO COMPUTE THE INVERSE MATRIX............................./





// FUNCTION TO COMPUTE THE COEFFICIENT BY INVERTING A....................../
static void get_coef(double *cf,double *A,double *rhs,int N){
  // Initialize auxiliary vectors.............................
  double *L=(double *)malloc(N*N*sizeof(double));	
  double *U=(double *)malloc(N*N*sizeof(double));	
  double *rhs2=(double *)malloc(N*sizeof(double));	
  // LU decomposition 
  get_LU(A,L,U,N);   double aux;
  // INVERT L :::::::::::::::::::::::::::::::::::::::::::::::
  rhs2[0]=rhs[0]/L[0]; 
  // Loop on rows
  for(int i=1;i<N;i++){ aux=0; 
     // Loop on columns
     for (int h=0;h<i;h++){aux+=L[i*N+h]*rhs2[h];} 
     // write the data
     rhs2[i]=(rhs[i]-aux)/L[i*N+i];
  }// end loop on rows
  // INVERT U :::::::::::::::::::::::::::::::::::::::::::::::
  cf[N-1]=rhs2[N-1]/U[N*N-1];
  // Loop on rows
  for(int i=N-2;i>-1;i--){aux=0;
    // Loop on columns
    for(int h=i+1;h<N;h++){aux+=U[i*N+h]*cf[h];} 
    cf[i]=(rhs2[i]-aux)/U[i*N+i];
  }// end loop on rows
  // Free data
  free(L); free(U); free(rhs2); return;
}// END FUNCTION TO COMPUTE THE COEFFICIENT BY INVERTING A................./







// FUNCTION TO COMPUTE LU DECOMPOSITION OF THE MATRIX....................../
static void get_LU(double *A,double *L,double *U,int a){
  // Initialize the matrices to 0's
  for(int i=0;i<a*a;i++){L[i]=0.0; U[i]=0.0;} double aux;
  // Write 1's in the diagonal of U
  for(int i=0;i<a;i++){U[i*a+i]=1.0;}   aux=A[0];  L[0]=aux;
  // FIRST ROW:.................................................
  for(int j=1;j<a;j++){U[j]=A[j]/L[0];}
  // LOOP ON ROWS:..............................................
  for(int i=1;i<a;i++){      aux=A[i*a];    L[i*a]=aux;
    // Loop on the elements of the columns in L
    for(int j=1;j<i+1;j++){      aux=0; 
	// Loop 
	for(int h=0;h<j;h++){aux+=L[i*a+h]*U[h*a+j];}
        // Actual value of the L matrix
        L[i*a+j]=A[i*a+j]-aux;
    }// End loop on elements
    // Compute the elements in the U matrix
    if(i<a-1){
        // Loop on the elements of the columns in U
        for(int j=i+1;j<a;j++){              aux=0; 
	    // Loop 
	    for(int h=0;h<i;h++){aux+=L[i*a+h]*U[h*a+j];}
            // The actual value
            U[i*a+j]=(A[i*a+j]-aux)/L[i*a+i];
	}// end loop on the elements
    }// end if on the index i
  }// END LOOP ON ROWS..............................................
  return;
}// END FUNCTION TO COMPUTE LU DECOMPOSITION OF THE MATRIX................./



// FUNCTION TO COMPUTE THE FACTORIAL......................................./
static double factorial(int h){
  // Initialize
  double fac=1;
  // Loop 
  for(int i=1;i<h+1;i++){fac=double(i)*fac;} return fac;
}// END FUNCTION TO COMPUTE THE FACTORIAL................................../

