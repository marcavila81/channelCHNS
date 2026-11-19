#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES for correction to boundary conditions 
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 22/07/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++++++ KERNEL ++++++++++++++++++++++++++++++*/
// KERNEL TO COMPUTE THE INVERSE MATRIX FOR BOUNDARY CONDITIONS............/
static __global__ void get_invC_k(float2 *invC,float2 *dv1,float2 *dv2){
 // Indexes...............................................................  
 int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=NX*NZ){return;}
 // Initialize the data
 float2 a,b,c,d,det;  float mod;
 // Load the data at y=0 and y=1
 a=dv1[k]; b=dv2[k];         c=dv1[NX*NZ+k]; d=dv2[NX*NZ+k];
 // Compute the determinant^-1.......................................
 det.x=a.x*d.x - a.y*d.y - b.x*c.x + b.y*c.y;
 det.y=a.x*d.y + a.y*d.x - b.x*c.y - b.y*c.x;
 mod=det.x*det.x + det.y*det.y;   det.x=det.x/mod;  det.y=-det.y/mod;
 // Compute the element 1,1 of the inverse matrix:
 invC[0*NX*NZ+k].x= d.x*det.x - d.y*det.y;
 invC[0*NX*NZ+k].y= d.x*det.y + d.y*det.x;
 // Compute the element 1,2 of the inverse matrix:
 invC[1*NX*NZ+k].x=-b.x*det.x + b.y*det.y;
 invC[1*NX*NZ+k].y=-b.x*det.y - b.y*det.x;
 // Compute the element 2,1 of the inverse matrix:
 invC[2*NX*NZ+k].x=-c.x*det.x + c.y*det.y;
 invC[2*NX*NZ+k].y=-c.x*det.y - c.y*det.x;
 // Compute the element 2,2 of the inverse matrix:
 invC[3*NX*NZ+k].x= a.x*det.x - a.y*det.y;
 invC[3*NX*NZ+k].y= a.x*det.y + a.y*det.x;
}// END KERNEL TO COMPUTE THE INVERSE MATRIX FOR BOUNDARY CONDITION......../




// KERNEL TO COMPUTE THE COEFFICIENTS C..................................../
static __global__ void comp_c_k(float2 *c,float2 *invC,float2 *dvp){
  // Indexes...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; if(k>=NX*NZ){return;}  
  // Initialize the data
  float2 a11,a12,a21,a22,r1,r2; 
  // Read the data
  r1.x=-dvp[k].x;
  r1.y=-dvp[k].y;
  r2.x=-dvp[NX*NZ+k].x;
  r2.y=-dvp[NX*NZ+k].y;
  a11=invC[k]; a12=invC[NX*NZ+k]; a21=invC[2*NX*NZ+k]; a22=invC[3*NX*NZ+k];
  // Compute c(1)
  c[      k].x=a11.x*r1.x - a11.y*r1.y + a12.x*r2.x - a12.y*r2.y;
  c[      k].y=a11.x*r1.y + a11.y*r1.x + a12.x*r2.y + a12.y*r2.x;
  // Compute c(2)
  c[NX*NZ+k].x=a21.x*r1.x - a21.y*r1.y + a22.x*r2.x - a22.y*r2.y;
  c[NX*NZ+k].y=a21.x*r1.y + a21.y*r1.x + a22.x*r2.y + a22.y*r2.x;
}// END KERNEL TO COMPUTE THE COEFFICIENTS C.............................../


// KERNEL TO ADD TO V THE CORRECTION......................................./
static __global__ void add_k(float2 *v,float2 *v1,float2 *v2,float2 *c){
 // Indexes...............................................................
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 int i=blockIdx.y*blockDim.y+threadIdx.y;
 // Stopper and init 
 if(k>=NX*NZ || i>=NY){return;} int h=i*NX*NZ+k; float2 c1,c2,b1,b2;
 // Read the data
 c1=c[k]; c2=c[NX*NZ+k]; b1=v1[h]; b2=v2[h];
 // Compute the resultant addition
 v[h].x+=(c1.x*b1.x - c1.y*b1.y + c2.x*b2.x - c2.y*b2.y);
 v[h].y+=(c1.x*b1.y + c1.y*b1.x + c2.x*b2.y + c2.y*b2.x);
}// END KERNEL TO ADD TO V THE CORRECTION................................../







/*++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++*/
static float2 *v1,*v2,*c,*dvp,*p1,*p2;
static float2 *invC;
static void get_invC(float2 *invC,float2 *dv1,float2 *dv2);







/*++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++*/
// SETTER................................................................../
void setBCon(void){
  size_t size=NY*NX*NZ*sizeof(float2); float2 *rhs,*dv1,*dv2;
  // Allocate internal variables......................................
  CHECK_CUDART(cudaMalloc(&v1,size));  float2 uu; uu.x=0.0; uu.y=0.0;
  CHECK_CUDART(cudaMalloc(&v2,size));  double2 uni,bco; 
  CHECK_CUDART(cudaMalloc(&p1,size));  uni.x=0.0; uni.y=0.0;
  CHECK_CUDART(cudaMalloc(&p2,size));
  CHECK_CUDART(cudaMalloc(&rhs,size));
  CHECK_CUDART(cudaMalloc(&invC, 2*2*NX*NZ*sizeof(float2)));
  CHECK_CUDART(cudaMalloc(&c,      2*NX*NZ*sizeof(float2)));
  CHECK_CUDART(cudaMalloc(&dv1,    2*NX*NZ*sizeof(float2)));
  CHECK_CUDART(cudaMalloc(&dv2,    2*NX*NZ*sizeof(float2)));
  CHECK_CUDART(cudaMalloc(&dvp,    2*NX*NZ*sizeof(float2)));
  // CALCULATE·THE·HOMOGENEOUS·SOLUTION·OF·THE·VELOCITY·AND·PHI············
  // COMPUTE V1.......................................................
  uniformVec(rhs,uu); bco.x=1.0;  bco.y=1.0;
  // Solve for phi1                
  LU_pg(p1,rhs,bco,uni);
  // Copy from phi1 to rhs and multiply rhs by L2
  bco.x=0.0; bco.y=0.0;
  // Solve for v1                  
  LU_v(v1,p1,bco,uni);
  // COMPUTE V2.......................................................
  uniformVec(rhs,uu); bco.x=1.0; bco.y=1.0;
  // Solve for phi2
  LU_pg(p2,rhs,uni,bco);
  // Copy from phi2 to rhs and multiply rhs by L2
  bco.x=0.0; bco.y=0.0;
  // Solve for v2
  LU_v(v2,p2,bco,uni);
  // Compute the derivative at the wall and the matrix inverse
  derivyw(dv1,v1);  derivyw(dv2,v2); get_invC(invC,dv1,dv2);
  // OUTRO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  cudaFree(rhs); cudaFree(dv1); cudaFree(dv2); return;
}// END SETTER............................................................./




// CORRECT BCON............................................................/
void corr_bcon(float2 *p,float2 *v){
 // Compute the derivative at the wall
 derivyw(dvp,v);
 // Dimensions
 dim3 grid,block; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
 // Kernel to compute the coefficients
 comp_c_k<<<grid,block>>>(c,invC,dvp); grid.y=NY;
 // Add v1,v2 to the velocity
 add_k<<<grid,block>>>(v,v1,v2,c); 
 add_k<<<grid,block>>>(p,p1,p2,c);  return;
}// END CORRECT BCON......................................................./






// DESTROYER.............................................................../
void BConDestroy(void){
  cudaFree(invC);
  cudaFree(v1);  cudaFree(v2);   cudaFree(c); 
  cudaFree(p1);  cudaFree(p2);   return;
}// END DESTROYER........................................................../









//+++++++++++++++++++++++++++++++ WRAPPERS ++++++++++++++++++++++++++++++++/
// WRAPPER TO COMPUTE THE INVERSE MATRIX.................................../
static void get_invC(float2 *invC,float2 *dv1,float2 *dv2){
  // Dimensions
  dim3 grid,block; block.x=block_size; grid.x=(NX*NZ+block.x-1)/block.x;
  // Kernel
  get_invC_k<<<grid,block>>>(invC,dv1,dv2); return;
}// END WRAPPER TO COMPUTE THE INVERSE MATRIX............................../

