/* MAIN FILE OF NSRAYBEN IN CUDA ---------------------------------------- */
/* Purpose: It integrates the Navier-Stokes equations in 
            cartesian coordinates using a pseudo-spectral formulation and a 
            fractional step method.
            - This version doesnot include a dynamic time step
-  Authors: Daniel Morón Montesdeoca. 
-  Contact: daniel.moron@zarm.uni-bremen.de
-  Date   : 17/07/2024                                                    */
#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int main(int argc, const char* argv[]){
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 printf("\n++++++++++++++  Welcome to nsWallChan in CUDA  +++++++++++++++");
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 // SINGLE CASE: explicit contact angle and Bond number (defines in head.h)
 float th_s = TH_S, Bo = BO_VALUE;
 // Startup assertion: the case values must be finite (guards against bad
 // defines in head.h, e.g. BO_VALUE left at 0)
 if(!isfinite(th_s) || !isfinite(Bo)){
   fprintf(stderr,"\n ERROR: non-finite case values: TH_S=%f, BO_VALUE=%f -- check the defines in head.h\n",(double)th_s,(double)Bo);
   exit(1);
 }
 // SET DEVICE AND START CLOCK
 int dev=3;  clock_t begin = clock(); struct stat st = {0};
 printf("\nSetting device %d\n",dev); CHECK_CUDART(cudaSetDevice(dev));
 // INITIALIZATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 size_p sizes; sizes.Ny=NY; sizes.Nz=NZ; sizes.Nx=NX; char Nme[255]={0}; 
 // Initialize y grid and integration matrix
 double *y_h  =(double *)malloc(NY*sizeof(double));
 double *D_h  =(double *)malloc(NY*NY*sizeof(double));
 double *L_h  =(double *)malloc( 3*NY*sizeof(double));
 double *L2_h =(double *)malloc( 3*NY*sizeof(double));
 double *R_h  =(double *)malloc( 5*NY*sizeof(double));
 double *R2_h =(double *)malloc( 5*NY*sizeof(double));
 double *FD_h =(double *)malloc(    6*sizeof(double));
 // Compute the grid and the compact finite difference matrices
 init_cfd(y_h,L_h,R_h,D_h,L2_h,R2_h,FD_h);  
 // Allocate memory for u and rhs
 vfield u; float2 *ch;  size_t size_p=NY*NX*NZ*sizeof(float2);
 CHECK_CUDART(cudaMalloc(&ch, size_p));  
 CHECK_CUDART(cudaMalloc(&u.x,size_p));  
 CHECK_CUDART(cudaMalloc(&u.y,size_p));  
 CHECK_CUDART(cudaMalloc(&u.z,size_p));  
 // Setters
 setFft(sizes);  setIO(y_h);   setNonlinear(y_h);  setDeriv(D_h);
 setLinear(L_h,R_h,L2_h,R2_h,FD_h); setBCon(); setCublas(y_h);
 // Create the folder where to save
 if(stat("./Results/",&st)==-1){mkdir("./Results/",0700);}
 // Create the case folder where to save
 sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/",th_s,Bo);
 if(stat(Nme,&st)==-1){mkdir(Nme,0700);}
 // Initialize the velocity field and the scalar field
 initField(u);  initCh(ch,y_h,th_s); 
 // INTEGRATION+++++++++++++++++++++++++++++++++++++++++++++++
 integrate(u,ch,Bo,th_s);   
 // OUTRO ++++++++++++++++++++++++++++++++++++++++++++++++++++
 printf("\n FINISHED TH=%03.2f AND BO=%02.2f\n",th_s,Bo);
 // Last velocity field 
 sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/ch.bin",th_s,Bo);
 wBufferBinary((float*)ch,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/ux.bin",th_s,Bo);
 wBufferBinary((float*)u.x,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/uy.bin",th_s,Bo);
 wBufferBinary((float*)u.y,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"./Results/TH_%03.2f_BO_%02.2f/uz.bin",th_s,Bo); 
 wBufferBinary((float*)u.z,Nme,sizeof(float2),NY*NX*NZ);
 // Destroyers of the Universe
 fftDestroy(); IODestroy(); NonDestroy(); LinearDestroy();
 DerivDestroy(); BConDestroy(); CublasDestroy();
 // Free GPU memory
 cudaFree(ch);   cudaFree(u.x);   cudaFree(u.y);    cudaFree(u.z);  
 // Free CPU memory
 free(y_h); free(D_h); free(L_h); free(R_h); free(L2_h); free(R2_h);
 // Final time and print
 clock_t end=clock();      double tot_t=(double)(end-begin)/CLOCKS_PER_SEC;
 printf("\n Total Time = %.3fs\n",tot_t);        printf("\n");    return 0;
}// END THE CODE!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
