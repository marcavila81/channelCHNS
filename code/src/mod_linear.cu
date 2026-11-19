#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to solve systems of equations 
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 22/07/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// KERNEL TO PERFORM LU DECOMPOSITION OF THE MEAN PROFILE................../
static __global__ void LU_x_k(double2*u,double*LH,double*RH,double2*rhs){
 // Kernel to perform LU decomposition for a Dirichlet boundary condition
 // Indexes & init........................................................ 
 int k=blockIdx.x*blockDim.x+threadIdx.x; 
 // Stopper & Indexes 
 if(k>=1){return;} 
 // Wavenumbers
 double kx=0.0; 
 double kz=0.0;
 // Factors to multiply the matrices
 double f1=-(dt/2.0)*(1.0/double(Re));
 double fac=1.0-f1*(kx*kx+kz*kz);
 // Initialize doubles
 double A,L1,L2,U1,U2,U3,U[3*NY];
 // FIRST ROW: BOUNDARY CONDITION....................................
 int j=0; u[j]=rhs[j]; U[0]=1.0; U[1]=0.0; U[2]=0.0;
 // SECOND ROW: REDUCED STENCIL......................................
 // Construct A matrix and save to L
 j=1;  A=fac*LH[j*3]+f1*RH[j*5]; L2=A; 
 // Substitute directly in u
 u[j].x=rhs[j].x - L2*u[(j-1)].x;
 u[j].y=rhs[j].y - L2*u[(j-1)].y;
 // Construct A matrix and save it directly to the U matrix
 A=fac*LH[j*3+1] + f1*RH[j*5+1]; U[j*3  ]=A;
 A=fac*LH[j*3+2] + f1*RH[j*5+2]; U[j*3+1]=A;
 A=              + f1*RH[j*5+3]; U[j*3+2]=A;
 // FROM THIRD TO NY-3 ROWS..........................................
 for(int i=2;i<NY-2;i++){
   // Construct the matrix A and L: first column
   A=             +f1*RH[i*5  ]; j=i-2; L1=A/U[j*3];
   // Construct the matrix A and L: second column
   A=fac*LH[i*3  ]+f1*RH[i*5+1]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];
   // Perform L substitution
   u[i].x=rhs[i].x-L1*u[(i-2)].x-L2*u[(i-1)].x;
   u[i].y=rhs[i].y-L1*u[(i-2)].y-L2*u[(i-1)].y;
   // Construct the matrix A&U: at the main diagonal
   A=fac*LH[i*3+1]+f1*RH[i*5+2];U[i*3  ]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];    
   // Construct the matrix A&U: at the main diagonal+1
   A=fac*LH[i*3+2]+f1*RH[i*5+3];U[i*3+1]=A                -L2*U[(i-1)*3+2];    
   // Construct the matrix A&U: at the main diagonal+2
   A=             +f1*RH[i*5+4];U[i*3+2]=A;
 }// end loop on rows
 // ROW NY-2: REDUCED STENCIL........................................
 int i=NY-2;
 // Construct the matrix A and L: first column
 A=             +f1*RH[i*5+1]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=fac*LH[i*3  ]+f1*RH[i*5+2]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];  
 // Perform L substitution        
 u[i].x=rhs[i].x-L1*u[(i-2)].x-L2*u[(i-1)].x;
 u[i].y=rhs[i].y-L1*u[(i-2)].y-L2*u[(i-1)].y;
 // First element of U in this row is 0
 U[i*3]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=fac*LH[i*3+1]+f1*RH[i*5+3];U[i*3+1]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];
 // Construct the matrix A&U: at the main diagonal+1
 A=fac*LH[i*3+2]+f1*RH[i*5+4];U[i*3+2]=A                -L2*U[(i-1)*3+2];
 // ROW NY-1: BOUNDARY CONDITION.....................................
 i=NY-1; u[i]=rhs[i];
 // last row of U
 U[i*3]=0.0;U[i*3+1]=0.0;U[i*3+2]=1.0;
 // .................................................................
 // U SUBSTITUTION ..................................................
 // .................................................................
 i=NY-2; U1=U[i*3+1]; U2=U[i*3+2]; 
 // substitute previous to last row
 u[i].x=(u[i].x-U2*u[(i+1)].x)/U1;
 u[i].y=(u[i].y-U2*u[(i+1)].y)/U1;
 // LOOP ON ROWS.....................................................
 for(int i=NY-3;i>0;i--){
   // Read values in the row
   U1=U[i*3]; U2=U[i*3+1]; U3=U[i*3+2];
   // Compute the solution
   u[i].x=(u[i].x-U2*u[(i+1)].x-U3*u[(i+2)].x)/U1;
   u[i].y=(u[i].y-U2*u[(i+1)].y-U3*u[(i+2)].y)/U1;
 }// end loop on rows
}// END KERNEL TO PERFORM LU DECOMPOSITION OF THE MEAN PROFILE............./




// KERNEL TO PERFORM LU DECOMPOSITION of PHI OR G........................../
static __global__ void LU_pg_k(double2*u,double*LH,double*RH,double2*rhs){
 // Kernel to perform LU decomposition for a Dirichlet boundary condition
 // Indexes & init........................................................ 
 int k=blockIdx.x*blockDim.x+threadIdx.x; 
 // Stopper & Indexes 
 if(k>=NX*NZ){return;} int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int iz=k%NZ;
 // Wavenumbers
 double kx=(PI2/LX)*(double(ix)); 
 double kz=(PI2/LZ)*(double(iz));
 // Factors to multiply the matrices
 double f1=-(dt/2.0)*(1.0/double(Re));
 double fac=1.0-f1*(kx*kx+kz*kz);
 // Initialize doubles
 double A,L1,L2,U1,U2,U3,U[3*NY];
 // FIRST ROW: BOUNDARY CONDITION....................................
 int j=0; u[j*NX*NZ+k]=rhs[j*NX*NZ+k]; U[0]=1.0; U[1]=0.0; U[2]=0.0;
 // SECOND ROW: REDUCED STENCIL......................................
 // Construct A matrix and save to L
 j=1;  A=fac*LH[j*3]+f1*RH[j*5]; L2=A; 
 // Substitute directly in u
 u[j*NX*NZ+k].x=rhs[j*NX*NZ+k].x - L2*u[(j-1)*NX*NZ+k].x;
 u[j*NX*NZ+k].y=rhs[j*NX*NZ+k].y - L2*u[(j-1)*NX*NZ+k].y;
 // Construct A matrix and save it directly to the U matrix
 A=fac*LH[j*3+1] + f1*RH[j*5+1]; U[j*3  ]=A;
 A=fac*LH[j*3+2] + f1*RH[j*5+2]; U[j*3+1]=A;
 A=              + f1*RH[j*5+3]; U[j*3+2]=A;
 // FROM THIRD TO NY-3 ROWS..........................................
 for(int i=2;i<NY-2;i++){
   // Construct the matrix A and L: first column
   A=             +f1*RH[i*5  ]; j=i-2; L1=A/U[j*3];
   // Construct the matrix A and L: second column
   A=fac*LH[i*3  ]+f1*RH[i*5+1]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];
   // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
   // Construct the matrix A&U: at the main diagonal
   A=fac*LH[i*3+1]+f1*RH[i*5+2];U[i*3  ]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];    
   // Construct the matrix A&U: at the main diagonal+1
   A=fac*LH[i*3+2]+f1*RH[i*5+3];U[i*3+1]=A                -L2*U[(i-1)*3+2];    
   // Construct the matrix A&U: at the main diagonal+2
   A=             +f1*RH[i*5+4];U[i*3+2]=A;
 }// end loop on rows
 // ROW NY-2: REDUCED STENCIL........................................
 int i=NY-2;
 // Construct the matrix A and L: first column
 A=             +f1*RH[i*5+1]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=fac*LH[i*3  ]+f1*RH[i*5+2]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];  
 // Perform L substitution        
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First element of U in this row is 0
 U[i*3]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=fac*LH[i*3+1]+f1*RH[i*5+3];U[i*3+1]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];
 // Construct the matrix A&U: at the main diagonal+1
 A=fac*LH[i*3+2]+f1*RH[i*5+4];U[i*3+2]=A                -L2*U[(i-1)*3+2];
 // ROW NY-1: BOUNDARY CONDITION.....................................
 i=NY-1; u[i*NX*NZ+k]=rhs[i*NX*NZ+k];
 // last row of U
 U[i*3]=0.0;U[i*3+1]=0.0;U[i*3+2]=1.0;
 // .................................................................
 // U SUBSTITUTION ..................................................
 // .................................................................
 i=NY-2; U1=U[i*3+1]; U2=U[i*3+2]; 
 // substitute previous to last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y)/U1;
 // LOOP ON ROWS.....................................................
 for(int i=NY-3;i>0;i--){
   // Read values in the row
   U1=U[i*3]; U2=U[i*3+1]; U3=U[i*3+2];
   // Compute the solution
   u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x
		                 -U3*u[(i+2)*NX*NZ+k].x)/U1;
   u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y
		                 -U3*u[(i+2)*NX*NZ+k].y)/U1;
 }// end loop on rows
}// END KERNEL TO PERFORM LU DECOMPOSITION of PHI OR G...................../





// KERNEL TO PERFORM LU DECOMPOSITION of V................................./
static __global__ void LU_v_k(double2*u,double*LH,double*RH,double2*rhs){
 // Kernel to perform LU decomposition for a Dirichlet boundary condition
 // Indexes & init........................................................
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 // Stopper & Indexes
 if(k>=NX*NZ){return;} int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int iz=k%NZ;
 // Wavenumbers
 double kx=(PI2/LX)*(double(ix));
 double kz=(PI2/LZ)*(double(iz));
 // Factors to multiply the matrices
 double fac=-kx*kx-kz*kz;
 // Initialize doubles
 double A,L1,L2,U1,U2,U3,U[3*NY];
 // FIRST ROW: BOUNDARY CONDITION....................................
 int j=0; u[j*NX*NZ+k]=rhs[j*NX*NZ+k]; U[0]=1.0; U[1]=0.0; U[2]=0.0;
 // SECOND ROW: REDUCED STENCIL......................................
 // Construct A matrix and save to L
 j=1;  A=fac*LH[j*3]+RH[j*5]; L2=A;
 // Substitute directly in u
 u[j*NX*NZ+k].x=rhs[j*NX*NZ+k].x - L2*u[(j-1)*NX*NZ+k].x;
 u[j*NX*NZ+k].y=rhs[j*NX*NZ+k].y - L2*u[(j-1)*NX*NZ+k].y;
 // Construct A matrix and save it directly to the U matrix
 A=fac*LH[j*3+1] + RH[j*5+1]; U[j*3  ]=A;
 A=fac*LH[j*3+2] + RH[j*5+2]; U[j*3+1]=A;
 A=              + RH[j*5+3]; U[j*3+2]=A;
 // FROM THIRD TO NY-3 ROWS..........................................
 for(int i=2;i<NY-2;i++){
   // Construct the matrix A and L: first column
   A=             +RH[i*5  ]; j=i-2; L1=A/U[j*3];
   // Construct the matrix A and L: second column
   A=fac*LH[i*3  ]+RH[i*5+1]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];
   // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
   // Construct the matrix A&U: at the main diagonal
   A=fac*LH[i*3+1]+RH[i*5+2];U[i*3  ]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1]; 
   // Construct the matrix A&U: at the main diagonal+1
   A=fac*LH[i*3+2]+RH[i*5+3];U[i*3+1]=A                -L2*U[(i-1)*3+2];
   // Construct the matrix A&U: at the main diagonal+2 
   A=             +RH[i*5+4];U[i*3+2]=A; 
 }// end loop on rows
 // ROW NY-2: REDUCED STENCIL........................................
 int i=NY-2;
 // Construct the matrix A and L: first column
 A=             +RH[i*5+1]; j=i-2; L1=A/U[j*3];  
 // Construct the matrix A and L: second column
 A=fac*LH[i*3  ]+RH[i*5+2]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3]; 
 // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First element of U in this row is 0
 U[i*3]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=fac*LH[i*3+1]+RH[i*5+3];U[i*3+1]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];
 // Construct the matrix A&U: at the main diagonal+1  
 A=fac*LH[i*3+2]+RH[i*5+4];U[i*3+2]=A                -L2*U[(i-1)*3+2];
 // ROW NY-1: BOUNDARY CONDITION..................................... 
 i=NY-1; u[i*NX*NZ+k]=rhs[i*NX*NZ+k]; 
 // last row of U  
 U[i*3]=0.0;U[i*3+1]=0.0;U[i*3+2]=1.0;  
 // .................................................................
 // U SUBSTITUTION ..................................................
 // .................................................................
 i=NY-2; U1=U[i*3+1]; U2=U[i*3+2];  
 // substitute previous to last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y)/U1;
 // LOOP ON ROWS.....................................................
 for(int i=NY-3;i>0;i--){
   // Read values in the row
   U1=U[i*3]; U2=U[i*3+1]; U3=U[i*3+2];
   // Compute the solution
   u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x
                                 -U3*u[(i+2)*NX*NZ+k].x)/U1;
   u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y
                                 -U3*u[(i+2)*NX*NZ+k].y)/U1;
 }// end loop on rows
}// END KERNEL TO PERFORM LU DECOMPOSITION of V............................/


// KERNEL TO PERFORM LU DECOMPOSITION of THE AUX CH VAR..................../
static __global__ void LU_th_k(double2*u,double*LH,double*RH,
		                        double2*rhs,double *FD){
 // Kernel to perform LU decomposition for a Dirichlet boundary condition
 // Indexes & init........................................................ 
 int k=blockIdx.x*blockDim.x+threadIdx.x; 
 // Stopper & Indexes 
 if(k>=NX*NZ){return;} int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int iz=k%NZ;
 // Wavenumbers
 double kx=(PI2/LX)*(double(ix)); 
 double kz=(PI2/LZ)*(double(iz));
 // Related parameters
 float g_c=(-0.5*s_c/float(Cn*Cn));
 //g_c = g_c*(1.0+sqrtf(1.0-((4.0*Pe*Cn*Cn)/(d_im*dt*s_c*s_c))));
 // Factors to multiply the matrices: Bth=fac; Ath=f1;
 double fac=-double(d_im/Pe)*(double(g_c)*double(Cn*Cn)+double(s_c)); //Bth
 double f1=double(d_im/Pe)*double(Cn*Cn);                             //Ath 
 // Cth = fac = Bth-Ath*(kx*kx+kz*kz)
 fac-=f1*(kx*kx+kz*kz);
 // Initialize doubles
 double A,L1,L2,U1,U2,U3,U[3*NY];
 // FIRST ROW: BOUNDARY CONDITION....................................
 int j=0; u[j*NX*NZ+k]=rhs[j*NX*NZ+k]; 
 // Write FD
 U[0]=FD[0]; U[1]=FD[1]; U[2]=FD[2];
 // SECOND ROW: REDUCED STENCIL......................................
 // Construct A matrix and save to L
 j=1;  A=fac*LH[j*3]+f1*RH[j*5]; L2=A/U[0]; 
 // Substitute directly in u
 u[j*NX*NZ+k].x=rhs[j*NX*NZ+k].x - L2*u[(j-1)*NX*NZ+k].x;
 u[j*NX*NZ+k].y=rhs[j*NX*NZ+k].y - L2*u[(j-1)*NX*NZ+k].y;
 // Construct A matrix and save it directly to the U matrix
 A=fac*LH[j*3+1] + f1*RH[j*5+1]; U[j*3  ]=A-L2*U[1];
 A=fac*LH[j*3+2] + f1*RH[j*5+2]; U[j*3+1]=A-L2*U[2];
 A=              + f1*RH[j*5+3]; U[j*3+2]=A;
 // FROM THIRD TO NY-3 ROWS..........................................
 for(int i=2;i<NY-2;i++){
   // Construct the matrix A and L: first column
   A=             +f1*RH[i*5  ]; j=i-2; L1=A/U[j*3];
   // Construct the matrix A and L: second column
   A=fac*LH[i*3  ]+f1*RH[i*5+1]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];
   // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
   // Construct the matrix A&U: at the main diagonal
   A=fac*LH[i*3+1]+f1*RH[i*5+2];U[i*3  ]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];    
   // Construct the matrix A&U: at the main diagonal+1
   A=fac*LH[i*3+2]+f1*RH[i*5+3];U[i*3+1]=A                -L2*U[(i-1)*3+2];    
   // Construct the matrix A&U: at the main diagonal+2
   A=             +f1*RH[i*5+4];U[i*3+2]=A;
 }// end loop on rows
 // ROW NY-2: REDUCED STENCIL........................................
 int i=NY-2;
 // Construct the matrix A and L: first column
 A=             +f1*RH[i*5+1]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=fac*LH[i*3  ]+f1*RH[i*5+2]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];  
 // Perform L substitution        
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First element of U in this row is 0
 U[i*3]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=fac*LH[i*3+1]+f1*RH[i*5+3];U[i*3+1]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];
 // Construct the matrix A&U: at the main diagonal+1
 A=fac*LH[i*3+2]+f1*RH[i*5+4];U[i*3+2]=A                -L2*U[(i-1)*3+2];
 // ROW NY-1: BOUNDARY CONDITION.....................................
 i=NY-1;
 // Construct the matrix A and L: first column
 A=FD[3]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=FD[4]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3+1];
 // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First 2 elements of U in this row r 0
 U[i*3]=0.0;  U[i*3+1]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=FD[5];     U[i*3+2]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+2]; 
 // .................................................................
 // U SUBSTITUTION ..................................................
 // .................................................................
 // LAST ROW
 U1=U[i*3+2]; 
 // substitute previous last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y)/U1;
 // SECOND ROW FROM THE END..........................................
 i=NY-2; U1=U[i*3+1]; U2=U[i*3+2]; 
 // substitute previous to last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y)/U1;
 // LOOP ON ROWS.....................................................
 for(int i=NY-3;i>-1;i--){
   // Read values in the row
   U1=U[i*3]; U2=U[i*3+1]; U3=U[i*3+2];
   // Compute the solution
   u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x
		                 -U3*u[(i+2)*NX*NZ+k].x)/U1;
   u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y
		                 -U3*u[(i+2)*NX*NZ+k].y)/U1;
 }// end loop on rows
}// END KERNEL TO PERFORM LU DECOMPOSITION of THE AUX CH VAR.............../



// KERNEL TO PERFORM LU DECOMPOSITION of THE CH VAR......................../
static __global__ void LU_ch_k(double2*u,double*LH,double*RH,
		                        double2*rhs,double *FD){
 // Kernel to perform LU decomposition for a Dirichlet boundary condition
 // Indexes & init........................................................ 
 int k=blockIdx.x*blockDim.x+threadIdx.x; 
 // Stopper & Indexes 
 if(k>=NX*NZ){return;} int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int iz=k%NZ;
 // Wavenumbers
 double kx=(PI2/LX)*(double(ix)); 
 double kz=(PI2/LZ)*(double(iz));
 // Related parameters
 float g_c=(-0.5*s_c/float(Cn*Cn));
 //g_c=g_c*(1.0+sqrtf(1.0-((4.0*Pe*Cn*Cn)/(d_im*dt*s_c*s_c))));
 // Factors to multiply the matrices: Bth=fac; Ath=f1;
 double fac=double(g_c);     //Bch
 double f1=1.0;              //Ach 
 // Cch = fac = Bch-Ach*(kx*kx+kz*kz)
 fac-=f1*(kx*kx+kz*kz);
 // Initialize doubles
 double A,L1,L2,U1,U2,U3,U[3*NY];
 // FIRST ROW: BOUNDARY CONDITION....................................
 int j=0; u[j*NX*NZ+k]=rhs[j*NX*NZ+k]; 
 // Write FD
 U[0]=FD[0]; U[1]=FD[1]; U[2]=FD[2];
 // SECOND ROW: REDUCED STENCIL......................................
 // Position A21=L21*U11-> Compute L21 and save to L2
 j=1;  A=fac*LH[j*3]+f1*RH[j*5]; L2=A/U[0]; 
 // Substitute directly in u to obtain the intermediate solution
 u[j*NX*NZ+k].x=rhs[j*NX*NZ+k].x - L2*u[(j-1)*NX*NZ+k].x;
 u[j*NX*NZ+k].y=rhs[j*NX*NZ+k].y - L2*u[(j-1)*NX*NZ+k].y;
 // Construct A matrix and save it directly to the U matrix
 A=fac*LH[j*3+1] + f1*RH[j*5+1]; U[j*3  ]=A-L2*U[1];
 A=fac*LH[j*3+2] + f1*RH[j*5+2]; U[j*3+1]=A-L2*U[2];
 A=              + f1*RH[j*5+3]; U[j*3+2]=A;
 // FROM THIRD TO NY-3 ROWS..........................................
 for(int i=2;i<NY-2;i++){
   // Construct the matrix A and L: first column
   A=             +f1*RH[i*5  ]; j=i-2; L1=A/U[j*3];
   // Construct the matrix A and L: second column
   A=fac*LH[i*3  ]+f1*RH[i*5+1]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];
   // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
   // Construct the matrix A&U: at the main diagonal
   A=fac*LH[i*3+1]+f1*RH[i*5+2];U[i*3  ]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];    
   // Construct the matrix A&U: at the main diagonal+1
   A=fac*LH[i*3+2]+f1*RH[i*5+3];U[i*3+1]=A                -L2*U[(i-1)*3+2];    
   // Construct the matrix A&U: at the main diagonal+2
   A=             +f1*RH[i*5+4];U[i*3+2]=A;
 }// end loop on rows
 // ROW NY-2: REDUCED STENCIL........................................
 int i=NY-2;
 // Construct the matrix A and L: first column
 A=             +f1*RH[i*5+1]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=fac*LH[i*3  ]+f1*RH[i*5+2]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3];  
 // Perform L substitution        
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First element of U in this row is 0
 U[i*3]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=fac*LH[i*3+1]+f1*RH[i*5+3];U[i*3+1]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+1];
 // Construct the matrix A&U: at the main diagonal+1
 A=fac*LH[i*3+2]+f1*RH[i*5+4];U[i*3+2]=A                -L2*U[(i-1)*3+2];
 // ROW NY-1: BOUNDARY CONDITION.....................................
 i=NY-1;
 // Construct the matrix A and L: first column
 A=FD[3]; j=i-2; L1=A/U[j*3];
 // Construct the matrix A and L: second column
 A=FD[4]; j=i-1; L2=(A-L1*U[(i-2)*3+1])/U[j*3+1];
 // Perform L substitution
u[i*NX*NZ+k].x=rhs[i*NX*NZ+k].x-L1*u[(i-2)*NX*NZ+k].x-L2*u[(i-1)*NX*NZ+k].x;
u[i*NX*NZ+k].y=rhs[i*NX*NZ+k].y-L1*u[(i-2)*NX*NZ+k].y-L2*u[(i-1)*NX*NZ+k].y;
 // First 2 elements of U in this row is 0
 U[i*3]=0.0;  U[i*3+1]=0.0;
 // Construct the matrix A&U: at the main diagonal
 A=FD[5];     U[i*3+2]=A-L1*U[(i-2)*3+2]-L2*U[(i-1)*3+2];
 // .................................................................
 // U SUBSTITUTION ..................................................
 // LAST ROW
 U1=U[i*3+2]; 
 // substitute previous last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y)/U1;
 // SECOND ROW FROM THE END..........................................
 i=NY-2; U1=U[i*3+1]; U2=U[i*3+2]; 
 // substitute previous to last row
 u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x)/U1;
 u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y)/U1;
 // LOOP ON ROWS.....................................................
 for(int i=NY-3;i>-1;i--){
   // Read values in the row
   U1=U[i*3]; U2=U[i*3+1]; U3=U[i*3+2];
   // Compute the solution
   u[i*NX*NZ+k].x=(u[i*NX*NZ+k].x-U2*u[(i+1)*NX*NZ+k].x
		                 -U3*u[(i+2)*NX*NZ+k].x)/U1;
   u[i*NX*NZ+k].y=(u[i*NX*NZ+k].y-U2*u[(i+1)*NX*NZ+k].y
		                 -U3*u[(i+2)*NX*NZ+k].y)/U1;
 }// end loop on rows
}// END KERNEL TO PERFORM LU DECOMPOSITION of THE CH VAR.................../






// KERNEL TO PERFORM MULTIPLICATION OF LH*v1.............................../
static __global__ void mul_LH_k(double2 *rhs,double *LH,float2 *v1){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;   int au,jleft,jright;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;}double2 u;
  // Initialize variables
  int h=i*NX*NZ+k; u.x=0.0; u.y=0.0;
  // Prepare two limits
  jleft=max(i-1,0)-(i-1);  jright=(i+1)-min(i+1,NY-1);
  // Loop on columns
  for(int j=jright;j<3-jleft;j++){au=i-1+j-jright+jleft;
   u.x+=LH[i*3+j]*double(v1[au*NX*NZ+k].x);  
   u.y+=LH[i*3+j]*double(v1[au*NX*NZ+k].y);}
  // Register the value.....................................
  rhs[h]=u;
}// END KERNEL TO PERFORM MULTIPLICATION OF LH*v1........................../

// KERNEL TO PERFORM MULTIPLICATION OF LH*v1.............................../
static __global__ void mul_LHx_k(double2 *rhs,double *LH,float2 *v1,
		double2 bcon,double2 bco2){
   // Indexes...............................................................
   int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NY){return;} 
   double2 u;  u.x=0.0;  u.y=0.0;   int au;
   // Option 1: Boundary condition at y=-1
   if(i==0){rhs[i].x=bcon.x; rhs[i].y=bcon.y; return;}
   // Option 2: Boundary condition at y=+1
   else if(i==NY-1){rhs[i].x=bco2.x; rhs[i].y=bco2.y; return;}
   // Option 3: no boundary condition........................
   else{for(int j=0;j<3;j++){au=i+j-1; 
      u.x+=LH[i*3+j]*double(v1[au].x); 
      u.y+=LH[i*3+j]*double(v1[au].y);}        
   // Register the value.....................................  
   rhs[i]=u;                           return; }
}// END KERNEL TO PERFORM MULTIPLICATION OF LH*v1........................../


/* +++++++++++++++++++++++++++ NEW KERNELS +++++++++++++++++++++++++++++++*/
// KERNEL TO PERFORM MULTIPLICATION OF RH*v1.............................../
static __global__ void mul_RH_k(double2 *rhs,double *RH,float2 *v1){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;   int au,jleft,jright;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper
  if(k>=NX*NZ || i>=NY){return;} double2 u;
  // Initialize variables
  int h=i*NX*NZ+k; u.x=0.0;u.y=0.0;
  // Prepare two limits
  jleft=max(i-2,0)-(i-2);  jright=(i+2)-min(i+2,NY-1);
  // Loop on columns
  for(int j=jright;j<5-jleft;j++){au=i-2+j-jright+jleft;
   u.x+=RH[i*5+j]*double(v1[au*NX*NZ+k].x);  
   u.y+=RH[i*5+j]*double(v1[au*NX*NZ+k].y);}
  // Register the value.....................................
  rhs[h]=u;
}// END KERNEL TO PERFORM MULTIPLICATION OF RH*v1........................../

// KERNEL TO PERFORM THE THOMAS ALGORITHM................................../
static __global__ void th_k(double2 *rhs,double *LH,double *c){
 // INDEX AND STOPPER
 int k=blockIdx.x*blockDim.x+threadIdx.x;  if(k>=NX*NZ){return;}
 // Initialize doubles and d
 double2 d[NY]; double a,b;
 // Loop 1-> compute d's starting from the first position
 b=LH[0];  d[0].x=rhs[k].x/b;  d[0].y=rhs[k].y/b;
 // Loop except last row
 for(int i=1;i<NY-1;i++){a=LH[i*3]; b=LH[i*3+1];
   d[i].x=(rhs[k+i*NX*NZ].x-a*d[i-1].x)/(b-a*c[i-1]);
   d[i].y=(rhs[k+i*NX*NZ].y-a*d[i-1].y)/(b-a*c[i-1]);}
 // Last row (reduced stencil)
 int i=NY-1; a=LH[i*3+1]; b=LH[i*3+2];
   d[i].x=(rhs[k+i*NX*NZ].x-a*d[i-1].x)/(b-a*c[i-1]);
   d[i].y=(rhs[k+i*NX*NZ].y-a*d[i-1].y)/(b-a*c[i-1]);
 // LOOP 2-> backward substitution starting from the last time step
 rhs[(NY-1)*NX*NZ+k].x=d[NY-1].x; rhs[(NY-1)*NX*NZ+k].y=d[NY-1].y;
 // Loop
 for(int i=NY-2;i>-1;i--){
    rhs[i*NX*NZ+k].x=d[i].x-c[i]*rhs[(i+1)*NX*NZ+k].x;
    rhs[i*NX*NZ+k].y=d[i].y-c[i]*rhs[(i+1)*NX*NZ+k].y;}
}// END KERNEL TO PERFORM THE THOMAS ALGORITHM............................./


/* ++++++++++++++++ KERNEL TO COMPUTE THE RHS 0,0 MODE +++++++++++++++++++*/
// KERNEL TO COMPUTE THE RHS OF EQUATION UX................................/
static __global__ void rhsux_k(float2 *rux,float2 *hux,float2 *ux,
		                         double *R2,double *L2,double *c){
 // Index & Stopper
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=1){return;}
 // Factor to use
 double fac=double(dt/2.0)*(1.0/double(Re)); 
 // Initialize
 double du[NY],d[NY],a,b;    int hleft,hright;
 // Loop 1-> Perform the multiplication R2*ux: loop on rows (j)
 for(int j=0;j<NY;j++){    du[j]=0.0; 
  // Limits
  hleft=max(j-2,0)-(j-2);  hright=(j+2)-min(j+2,NY-1);
  // Loop on columns (h)
  for(int h=hright;h<5-hleft;h++){
   du[j]+=R2[j*5+h]*double(ux[j-2+h-hright+hleft].x);}
 }// end the loop on rows
 // Loop 2-> compute d's starting from the first position (j)
 b=L2[0]; d[0]=du[0]/b; 
 // loop except last row
 for(int j=1;j<NY-1;j++){a=L2[j*3]; b=L2[j*3+1];
     d[j]=(du[j] - a*d[j-1])/(b-a*c[j-1]);}
 // last row (reduced stencil) 
 int k=NY-1; a=L2[k*3+1]; b=L2[k*3+2]; d[k]=(du[k]-a*d[k-1])/(b-a*c[k-1]);
 // Loop 3-> Backward substitution starting from the last step (j)
 du[NY-1]=d[NY-1]; for(int j=NY-2;j>-1;j--){du[j]=d[j]-c[j]*du[j+1];}
 // Loop 4-> Compute the final rhs
 for(int j=0;j<NY;j++){
     du[j]=double(ux[j].x) + fac*du[j] + 
           double(dt/2.0)*double(3.0*rux[j].x - hux[j].x); 
 // Save to vectors
 hux[j].x=rux[j].x;hux[j].y=rux[j].y;rux[j].x=float(du[j]);rux[j].y=0.0;}
}// END KERNEL TO COMPUTE THE RHS OF EQUATION UX.........................../


// KERNEL TO COPY A DOUBLE TO A FLOAT....................................../
static __global__ void db2flbuffer(float2 *fl,double2 *db){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;
  // Copy
  fl[h].x=float(db[h].x);  fl[h].y=float(db[h].y);
}// END KERNEL TO COPY A DOUBLE TO A FLOAT................................./


// KERNEL TO COPY A DOUBLE TO A FLOAT....................................../
static __global__ void db2flx(float2 *fl,double2 *db){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper
  if(k>=NY || i>=1){return;} 
  // Copy
  fl[k].x=float(db[k].x);  fl[k].y=float(db[k].y);
}// END KERNEL TO COPY A DOUBLE TO A FLOAT................................./



/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
static double *L,*R,*c,*c2,*LH,*RH,*FD; static double2 *arh,*aso,*arx,*asx;
static void comp_c(double *c_h,double *L);



/*+++++++++++++++++++++++++++ MAIN FUNCTION ++++++++++++++++++++++++++++++*/
// SETTER OF LUSOLVE......................................................./
void setLinear(double *L_h,double *R_h,double *L2h,double *R2h,double*FD_h){
   // Internal variables
   double *c_h=(double *)malloc(NY*sizeof(double));
   double *c2_h=(double *)malloc(NY*sizeof(double));
   comp_c(c_h,L_h);  comp_c(c2_h,L2h);
   // Malloc
   CHECK_CUDART(cudaMalloc( &c,NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&c2,NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&FD,6*sizeof(double)));
   CHECK_CUDART(cudaMalloc( &L,3*NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc( &R,5*NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&LH,3*NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&RH,5*NY*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&arh,NY*NX*NZ*sizeof(double2)));
   CHECK_CUDART(cudaMalloc(&aso,NY*NX*NZ*sizeof(double2)));
   CHECK_CUDART(cudaMalloc(&arx,NY*sizeof(double2)));
   CHECK_CUDART(cudaMalloc(&asx,NY*sizeof(double2)));
   // CudaMemcpy
CHECK_CUDART(cudaMemcpy(c,c_h,NY*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(c2,c2_h,NY*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(FD,FD_h,6*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(L,L_h,3*NY*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(R,R_h,5*NY*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(LH,L2h,3*NY*sizeof(double),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(RH,R2h,5*NY*sizeof(double),cudaMemcpyHostToDevice));
   // Free memory and return
   free(c_h); free(c2_h);  return;
}// END MAIN FUNCTION OF SETTER............................................/






// WRAPPER TO LU SOLUTION OF PHI OR G....................................../
void LU_pg(float2 *u,float2 *rhs,double2 bcon,double2 bco2){
  // Initialize the dimensions of the grids and blocks
  dim3 block,grid; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
  // Kernel to multiply rhs by L2
  grid.y=NY; mul_LH_k<<<grid,block>>>(arh,LH,rhs);
  // Write bcon
  write_bcon(arh,bcon,bco2);
  // Call the kernel to solve linear system of equations
  grid.y=1; LU_pg_k<<<grid,block>>>(aso,LH,RH,arh);
  // Call the kernel to copy a double to a float
  grid.y=NY;   db2flbuffer<<<grid,block>>>(u,aso); return;
}// END WRAPPER TO LU SOLUTION OF PHI OR G................................./

// WRAPPER TO LU SOLUTION OF V............................................./
void LU_v(float2 *u,float2 *rhs,double2 bcon,double2 bco2){
  // Initialize the dimensions of the grids and blocks
  dim3 block,grid; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
  // Kernel to multiply rhs by L2
  grid.y=NY; mul_LH_k<<<grid,block>>>(arh,LH,rhs);
  // Write bcon
  write_bcon(arh,bcon,bco2);
  // Call the kernel to solve the linear system of equations
  grid.y=1;    LU_v_k<<<grid,block>>>(aso,LH,RH,arh);
  // Call the kernel to copy a double to a float
  grid.y=NY;   db2flbuffer<<<grid,block>>>(u,aso); return;
}// END WRAPPER TO LU SOLUTION OF V......................................../

// WRAPPER TO LU SOLUTION OF CH............................................/
void LU_ch(float2 *cH,float2 *th,float2 *ch_BC,float2 *chn_BC){
  // Initialize the dimensions of the grids and blocks
  dim3 block,grid; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
  // Kernel to multiply rhs by L2
  grid.y=NY; mul_LH_k<<<grid,block>>>(arh,LH,th);
  // Write bcon
  write_bconCH(arh,ch_BC,chn_BC);
  // Call the kernel to solve the linear system of equations
  grid.y=1;    LU_ch_k<<<grid,block>>>(aso,LH,RH,arh,FD);
  // Call the kernel to copy a double to a float
  grid.y=NY;   db2flbuffer<<<grid,block>>>(cH,aso); return;
}// END WRAPPER TO LU SOLUTION OF CH......................................./

// WRAPPER TO LU SOLUTION OF TH............................................/
void LU_th(float2 *th,float2 *rhs,float2 *th_BC,float2 *thn_BC){
  // Initialize the dimensions of the grids and blocks
  dim3 block,grid; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
  // Kernel to multiply rhs by L2
  grid.y=NY; mul_LH_k<<<grid,block>>>(arh,LH,rhs);
  // Write bcon
  write_bconCH(arh,th_BC,thn_BC);
  // Call the kernel to solve the linear system of equations
  grid.y=1;    LU_th_k<<<grid,block>>>(aso,LH,RH,arh,FD);
  // Call the kernel to copy a double to a float
  grid.y=NY;   db2flbuffer<<<grid,block>>>(th,aso); return;
}// END WRAPPER TO LU SOLUTION OF TH......................................./

// WRAPPER TO LU SOLUTION OF V............................................./
void LU_x(float2 *u,float2 *rhs,double2 bcon,double2 bco2){
  // Initialize the dimensions of the grids and blocks
  dim3 block,grid; block.x=block_size; grid.x=(NY+block.x-1)/block.x;
  // Multiplicate the rhs with L2 and also add the bcon...
  grid.y=1;   mul_LHx_k<<<grid,block>>>(arx,LH,rhs,bcon,bco2);
  // Call the kernel to solve the linear system of equations
  grid.x=1;   LU_x_k<<<grid,block>>>(asx,LH,RH,arx); 
  // Call the kernel to copy a double to a float
  grid.x=(NY+block.x-1)/block.x;  db2flx<<<grid,block>>>(u,asx); return;
}// END WRAPPER TO LU SOLUTION OF V......................................../







// WALL-NORMAL DERIVATIVE................................................../
void derivy(float2 *grad,float2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x;   grid.y=NY;
   // Kernel 1: multiply p by RH
   mul_RH_k<<<grid,block>>>(arh,R,p);
   // Kernel 2: Invert LH
   grid.y=1; th_k<<<grid,block>>>(arh,L,c);
   // Kernel 3: Copy from double to float
   grid.y=NY;db2flbuffer<<<grid,block>>>(grad,arh); return;
}// END OF WALL-NORMAL DERIVATIVE........................................../

// WALL-NORMAL DERIVATIVE 2................................................/
void derivyy(float2 *grad,float2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x;   grid.y=NY;
   // Kernel 1: multiply p by RH
   mul_RH_k<<<grid,block>>>(arh,RH,p);
   // Kernel 2: Invert LH
   grid.y=1; th_k<<<grid,block>>>(arh,LH,c2);
   // Kernel 3: Copy from double to float
   grid.y=NY;db2flbuffer<<<grid,block>>>(grad,arh); return;
}// END OF WALL-NORMAL DERIVATIVE 2......................................../

// COMPUTE THE RHS OF THE UX EQUATION....................................../
void rhsux(float2 *rux,float2 *hux,float2 *ux){
  // Dimensions
  dim3 grid,block; block.x=block_size;
  grid.x=(1+block.x-1)/block.x; grid.y=1;
  // Kernel
  rhsux_k<<<grid,block>>>(rux,hux,ux,RH,LH,c2);
}// END COMPUTE THE RHS OF THE UX EQUATION................................./


// DESTROYER OF LUSOLVE..................................................../
void LinearDestroy(void){
 cudaFree(L);cudaFree(R);cudaFree(c);cudaFree(c2);cudaFree(LH);cudaFree(RH);
 cudaFree(arh);cudaFree(aso);cudaFree(asx);cudaFree(arx);cudaFree(FD);
 return;
}// END DESTROYER OF LU..................................................../


/*++++++++++++++++++++++++++INTERNAL FUNCTIONS++++++++++++++++++++++++++++*/
// COMPUTE C & D.........................................................../
static void comp_c(double *c_h,double *L){
 // Initialize stuff
 double a,b,co;
 // First row also last
 b=L[0];  co=L[1];   c_h[0]=co/b;  c_h[NY-1]=0;
 // Rest of rows except last one
 for(int i=1;i<NY-1;i++){a=L[3*i]; b=L[3*i+1]; co=L[3*i+2];
                           c_h[i]=co/(b-a*c_h[i-1]);}
}// END COMPUTE C & D....................................................../


