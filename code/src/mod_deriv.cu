#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate derivatives
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 22/07/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// KERNEL TO CALCULATE THE STREAMWISE DERIV................................/
static __global__ void derivx_k(float2 *grad,float2 *p){
   // Indexes...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   int i=blockIdx.y*blockDim.y+threadIdx.y;
   // Stopper and initialize variables
   if(k>=NX*NZ || i>=NY){return;} int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX;
   // Wavenumber & global index
   float kx=(PI2/LX)*(float(ix));  int h=i*NX*NZ+k; float2 u;
   u.x=p[h].x; u.y=p[h].y;
   // Write to auxiliary variable...........................................
   grad[h].x=-kx*u.y;  grad[h].y=kx*u.x;
}// END STREAMWISE DERIV.................................................../                                                                            

// KERNEL TO CALCULATE THE SPANWISE DERIV................................../
static __global__ void derivz_k(float2 *grad,float2 *p){
   // Indexes...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   int i=blockIdx.y*blockDim.y+threadIdx.y;
   // Stopper and initialize variables
   if(k>=NX*NZ || i>=NY){return;} int l=k%NZ;
   // Wavenumber & global index
   float kz=(PI2/LZ)*(float(l));  int h=i*NX*NZ+k; float2 u;
   u.x=p[h].x; u.y=p[h].y;
   // Write to auxiliary variable...........................................
   grad[h].x=-kz*u.y; grad[h].y=kz*u.x;
}// END AXIAL DERIV......................................................../



// KERNEL TO COMPUTE WALL NORMAL DERIVATIVE @Wall (TO IMPROVE)............./
static __global__ void derivyw_k(double *D,float2 *grad,float2 *p){          
// Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; 
  int i=blockIdx.y*blockDim.y+threadIdx.y;                       
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=2){return;}  int h=i*NX*NZ+k;
  // Initialize stuff
  double2 u;  u.x=0.0; u.y=0.0; i=i*(NY-1);
  // Loop on columns
  for(int j=0;j<NY;j++){                   
     u.x+=D[i*NY+j]*double(p[j*NX*NZ+k].x);  
     u.y+=D[i*NY+j]*double(p[j*NX*NZ+k].y);
  }// end loop on columns
  // Save the value
  grad[h].x=float(u.x);    grad[h].y=float(u.y);
}//END WALL-NORMAL DERIVATIVE @Wall......................................../


// KERNEL TO COMPUTE PHI FROM WALL NORMAL VELOCITY........................./
static __global__ void v_to_phi_k(float2 *phi,float2 *v){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;
  // Index of wavenumbers
  int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ;
  // Wavelength
  float kz=(PI2/LZ)*(float(l)); float kx=(PI2/LX)*(float(ix));
  // Write to vector
  phi[h].x=phi[h].x-(kx*kx+kz*kz)*v[h].x;
  phi[h].y=phi[h].y-(kx*kx+kz*kz)*v[h].y;
}// END KERNEL TO COMPUTE PHI FROM WALL NORMAL VELOCITY..................../


// KERNEL TO COMPUTE PHI FROM WALL NORMAL VELOCITY........................./
static __global__ void u_to_g_k(float2 *g,float2 *ux,float2 *uz){ 
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;  
  // Index of wavenumbers
  int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ; 
  // Wavelength
  float kz=(PI2/LZ)*(float(l)); float kx=(PI2/LX)*(float(ix));
  // Boundary condition
  if(i<1 || i>=NY-1){g[h].x=0.0;g[h].y=0.0;}
  // Else
  else{  
  // Register value to g
  g[h].x=-kz*ux[h].y + kx*uz[h].y;  
  g[h].y= kz*ux[h].x - kx*uz[h].x;
  } // end else
}// END KERNEL TO COMPUTE PHI FROM WALL NORMAL VELOCITY..................../


// KERNEL TO COMPUTE U FROM V/G............................................/
static __global__ void vg_to_u_k(float2 *ux,float2 *uz,float2 *g){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;
  // Index of wavenumbers
  int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ;  float2 du,dg;
  // Wavelength
  float kz=(PI2/LZ)*(float(l));
  float kx=(PI2/LX)*(float(ix));
  // Read du and g
  dg=g[h];   du=ux[h];
  // Case k==0
  if(k<1){ux[h].x=0.0; ux[h].y=0.0; uz[h].x=0.0; uz[h].y=0.0;}
  // Boundary Condition Case i==0 or i==NY-1
  else if(i<1 || i>=NY-1){ux[h].x=0.0;ux[h].y=0.0;uz[h].x=0.0;uz[h].y=0.0;}
  // Else 
  else{
   // Compute the denominator
   float mod=kx*kx+kz*kz;  kx=kx/mod;  kz=kz/mod;
   // Compute velocity ux
   uz[h].x= -kz*du.y - kx*dg.y;      uz[h].y= kz*du.x + kx*dg.x;
   ux[h].x= -kx*du.y + kz*dg.y;      ux[h].y= kx*du.x - kz*dg.x;
  }// end else
}// END KERNEL TO COMPUTE U FROM V/G......................................./


// KERNEL TO COMPUTE HP AND HG FROM HX HY HZ.............................../
static __global__ void get_hp_hg_k(float2 *dux,float2 *duy,float2 *duz,
		                            float2 *dHxdy,float2 *dHzdy){
 // Index
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 int i=blockIdx.y*blockDim.y+threadIdx.y;   
 // Stopper and initialize variables 
 if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k; 
 // Index of wavenumbers    
 int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ;  float2 dp,dg;
 // Wavelength 
 float kz=(PI2/LZ)*(float(l));   
 float kx=(PI2/LX)*(float(ix));  
 // H_phi: 
 dp.x =  kx*dHxdy[h].y + kz*dHzdy[h].y - (kx*kx+kz*kz)*duy[h].x;
 dp.y = -kx*dHxdy[h].x - kz*dHzdy[h].x - (kx*kx+kz*kz)*duy[h].y;
 // H_g  :
 dg.x = -kz*dux[h].y + kx*duz[h].y;   dg.y= kz*dux[h].x - kx*duz[h].x;
 // Save to dux, duz the dp and dg
 dux[h]=dp;  duz[h]=dg;
}// END KERNEL TO COMPUTE HP AND HG FROM HX HY HZ........................../


// KERNEL TO COMPUTE THE LAPLACIAN for u.................................../
static __global__ void compLap_k(float2 *Lap,float2 *u){
  // Indexes............................................................... 
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;
  // Factor to use
  float fac=float(dt/2)*float(1.0/float(Re));
  // Index of wavenumbers
  int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ;
  // Wavelength 
  float kz=(PI2/LZ)*(float(l));
  float kx=(PI2/LX)*(float(ix));
  // Save to Laplacian
  Lap[h].x=u[h].x - fac*(kx*kx+kz*kz)*u[h].x + fac*Lap[h].x;
  Lap[h].y=u[h].y - fac*(kx*kx+kz*kz)*u[h].y + fac*Lap[h].y;
}// END KERNEL TO COMPUTE THE LAPLACIAN for u............................../


// KERNEL TO COMPUTE THE LAPLACIAN FOR c.................................../
static __global__ void Lap_k(float2 *LapK,float2 *cK){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and initialize variables
  if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k;
  // Read
  float2 Lap=LapK[h], c=cK[h]; 
  // Index of wavenumbers
  int ix=k/NZ; ix=ix<NX/2 ? ix:ix-NX; int l=k%NZ;
  // Wavelength
  float kz=(PI2/LZ)*(float(l));
  float kx=(PI2/LX)*(float(ix));
  // Add to Lap the new values
  Lap.x+=-(kx*kx+kz*kz)*c.x; Lap.y+=-(kx*kx+kz*kz)*c.y;
  // Write to vector
  LapK[h].x=Lap.x;  LapK[h].y=Lap.y;  return;
}// END KERNEL TO COMPUTE THE LAPLACIAN FOR c............................../


// KERNEL TO COMPUTE dudt................................................./
static __global__ void derivt_k(float *grad,float *uK,float *ubK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data
  float u=uK[h], ub=ubK[h], drv;
  // Compute derivative. Take care of mode 0...
  if(k<2){drv=0.0;}else{drv=(u-ub)/float(dt);} grad[h]=drv; return;
}// END KERNEL TO COMPUTE vphi............................................/





/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
static double *D;






/*+++++++++++++++++++++++++++ MAIN FUNCTION ++++++++++++++++++++++++++++++*/
// SETTER OF DERIVATIVES.................................................../
void setDeriv(double *D_h){
   // Allocate
   CHECK_CUDART(cudaMalloc( &D,NY*NY*sizeof(double)));
   // CudaMemcpy
CHECK_CUDART(cudaMemcpy(D,D_h,NY*NY*sizeof(double),cudaMemcpyHostToDevice));
   // Free memory and return
   return;
}// END MAIN FUNCTION OF SETTER............................................/


// STREAMWISE DERIV......................................................../
void derivx(float2 *grad,float2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
   // Kernel
   derivx_k<<<grid,block>>>(grad,p); return;
}// END OF STREAMWISE DERIV................................................/


// SPANWISE DERIV........................................................../
void derivz(float2 *grad,float2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
   // Kernel
   derivz_k<<<grid,block>>>(grad,p); return;
}// END OF AXIAL DERIV...................................................../



// WALL-NORMAL DERIVATIVE @WALL............................................/
void derivyw(float2 *grad,float2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=2;
   // Kernel
   derivyw_k<<<grid,block>>>(D,grad,p); return;
}// END OF WALL-NORMAL DERIVATIVE 2......................................../



// FROM V TO PHI.........................................................../
void v_to_phi(float2 *phi,float2 *v){
   // Derivative
   derivyy(phi,v);
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
   // Kernel 3: Compute the actual phi
   v_to_phi_k<<<grid,block>>>(phi,v); return;
}// END FROM V TO PHI....................................................../


// FROM U TO G............................................................./
void u_to_g(float2 *g,float2 *ux,float2 *uz){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
   // Kernel
   u_to_g_k<<<grid,block>>>(g,ux,uz); return;
}// END FROM U TO G......................................................../

// FROM V/G TO U.........................................................../
void vg_to_u(float2 *ux,float2 *uz,float2 *g,float2 *v){
   // Derivative
   derivy(ux,v);
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
   // Kernel 3: compute the actual ux and uz
   vg_to_u_k<<<grid,block>>>(ux,uz,g); return;
}// END FROM V/G TO U....................................................../


// FROM HX/HY/HZ TO HP AND HG............................................../
void get_hp_hg(float2 *dux,float2 *duy,float2 *duz,
		                        float2 *dHxdy,float2 *dHzdy){
  // Dimensions
  dim3 grid,block; block.x=block_size;
  grid.x=(NX*NZ+block.x-1)/block.x; grid.y=NY;
  // Kernel
  get_hp_hg_k<<<grid,block>>>(dux,duy,duz,dHxdy,dHzdy);  return;
}// END FROM HX/HY/HZ TO HP AND HG........................................./


// COMPUTE THE LAPLACIAN TO ADD TO THE RHS OF U............................/
void compLap(float2 *Lap,float2 *u){
   // Derivative
   derivyy(Lap,u);
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x;   grid.y=NY;
   // Kernel 3: assemble the laplacian
   compLap_k<<<grid,block>>>(Lap,u); return;
}// END COMPUTE THE LAPLACIAN TO ADD TO THE RHS OF U....................../


// COMPUTE THE LAPLACIAN................................................../
void Laplacian(float2 *Lap,float2 *c){
   // Derivative
   derivyy(Lap,c);
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NX*NZ+block.x-1)/block.x;   grid.y=NY;
   // Kernel 3: assemble the laplacian
   Lap_k<<<grid,block>>>(Lap,c); return;
}// END COMPUTE THE LAPLACIAN............................................./

// COMPUTE THE TIME DERIVATIVE............................................/
void derivt(float2 *grad,float2 *u,float2 *ub){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;   grid.y=NY;
   // Kernel 3: assemble the laplacian
   derivt_k<<<grid,block>>>((float *)grad,(float *)u,(float *)ub); return;
}// END COMPUTE THE LAPLACIAN............................................./

// DESTROYER OF LUSOLVE..................................................../
void DerivDestroy(void){
   cudaFree(D);       return;
}// END DESTROYER OF LU..................................................../





