#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to compute mode zero-zero
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 07/08/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// MAIN KERNEL FOR MODE 0-0................................................/
static __global__ void main_ZZ_k(float2 *ux,float2 *uy,float2 *uz,
		                 float2 *p, float2 *g){
 // INDEX
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 // STOPPER & ACTUAL INDEX
 if(i>=NY){return;} int h=i*NX*NZ; float2 aux; aux.x=0.0; aux.y=0.0;
 // 1. Velocities 
 ux[h]=aux;   uy[h]=aux;   uz[h]=aux;  p[h]=aux;  g[h]=aux;
}// END MAIN KERNEL FOR MODE 0-0.........................................../


// KERNEL TO COPY TO UX EQUATION........................................../
static __global__ void copycorr_k(float2 *ru,float corr){
  // Index & Stopper
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NY){return;} 
  // Copy value
  if(i>0 && i<NY-1){ru[i].x+=corr; ru[i].y=0.0;}
}// END KERNEL TO COPY TO UX EQUATION...................................../

/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
static cublasHandle_t cublasHandle; static float *intdy, *Qa_h;




/*+++++++++++++++++++++++++++ MAIN FUNCTION ++++++++++++++++++++++++++++++*/
// CUBLAS CHECK------------------------------------------------------------/
void cublasCheck(cublasStatus_t error, const char* function ){
  if(error !=  CUBLAS_STATUS_SUCCESS){
    printf("\n error  %s : %d \n", function, error);  exit(1);
  }
  return;
}// END CUBLAS CHECK-------------------------------------------------------/


// SETTER CUBLAS AND FLUX FUNCTIONS----------------------------------------/
void setCublas(double *y_h){
    // Allocate intdy in the host
    float *dy_h=(float *)malloc(NY*sizeof(float)); 
    float *dy1_h=(float *)malloc(NY*sizeof(float)); 
    // Initialize dy_h to 0
    for(int i=0;i<NY;i++){dy_h[i]=0.0;  dy1_h[i]=0.0;}
    // Compute dy1_h
    for(int i=0;i<NY-1;i++){dy1_h[i]=0.5*float(y_h[i+1]-y_h[i]);} 
    // First element of the dy_h vector
    dy_h[0]=dy1_h[0];
    // Loop on i to add the two sides of the trapezoidal rule 
    for(int i=1;i<NY;i++){dy_h[i]=dy1_h[i]+dy1_h[i-1];}
    // Allocate intdy
    CHECK_CUDART(cudaMalloc(&intdy,NY*sizeof(float)));
    // CudaMemcpy
    CHECK_CUDART(cudaMemcpy(intdy,dy_h,NY*sizeof(float),cudaMemcpyHostToDevice));
    // Cublacreate
    cublasCheck(cublasCreate(&cublasHandle),"Cre");
    // Dot product auxiliary vector: where to save
    Qa_h=(float *)malloc(NY*sizeof(float)); free(dy_h); free(dy1_h); return;
}//END OF SETTER AND FLUX................................................../


// DESTROYER CUBLAS......................................................../
void CublasDestroy(void){cudaFree(intdy); free(Qa_h); 
                         cublasDestroy(cublasHandle); return;
}// END OF DESTROYER CUBLAS................................................/



// PERFORM DOT PRODUCT USING CUBLAS----------------------------------------/
void adj_flux(float2 *rhx){  float Qdes=2.0/3.0;
   // Perform dot product
 cublasCheck(cublasSdot(cublasHandle,NY,intdy,1,(float *)rhx,2,Qa_h),"Tr"); 
   // Correction to the bulk velocity. I multiply by 0.5 instead of 1/LY
   float corr=Qdes-0.5*Qa_h[0]; 
   // Set-up kernel to copy the correction to the rhs
   dim3 block,grid; block.x=block_size;
   block.x=block_size; grid.x=(NY+block.x-1)/block.x; grid.y=1;
   // Copy in kernel
   copycorr_k<<<grid,block>>>(rhx,corr);                return;
}// END OF DOT PRODUCT USING CUBLAS......................................../

// MAIN FUNCTION TO INTEGRATE MODE ZZ....................................../
void main_ZZ(vfield u,float2 *p,float2 *g){
  // Call the wrapper to set the mode zz to 0
  dim3 grid,block; block.x=block_size; grid.x=(NY+block.x-1)/block.x;
  // Call the kernel
  main_ZZ_k<<<grid,block>>>(u.x,u.y,u.z,p,g);
}// END MAIN FUNCTION TO INTEGRATE MODE ZZ................................./


// MAIN FUNCTION TO COMPUTE THE POSITION OF THE CENTER OF GRAVITY........../
void wrt_ycg(float2 *ycg,float2 *bbl,float *flag,float th_s,float Bo){
   float S_ycg,S_Vl;
   // Sum up the position
   cublasCheck(cublasSasum(cublasHandle,NY*NXP*2*NZP,(float*)ycg,1,&S_ycg),"Tr");
   cublasCheck(cublasSasum(cublasHandle,NY*NXP*2*NZP,(float*)bbl,1,&S_Vl),"Tr");
   // Normalize the two values
   S_ycg=S_ycg/S_Vl;   S_Vl=float(LX*LY*LZ)*S_Vl/float(NY*NXP*(2*NZP-2)); 
   // Initialize
   FILE *ptrf;  char Nme[256]={0};
   // Write
   sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/ycg_%03.2f_%02.2f.txt",th_s,Bo,th_s,Bo);
   // Open the file
   ptrf= fopen(Nme,"a");
   // Write the two values
   fprintf(ptrf,"%.8e  %.8e  %.8e\n",flag[1],S_ycg,S_Vl); fclose(ptrf); return;
}// END MAIN FUNCTION TO COMPUTE THE POSITION OF THE CENTER OF GRAVITY...../





