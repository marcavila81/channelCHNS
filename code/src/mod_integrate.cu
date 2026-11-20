#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate functions to integrate
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 01/08/2024
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// KERNEL TO COPY TO UX EQUATION........................................../
static __global__ void copyuxb_k(float2 *ux,float2 *u){  
  // Index & Stopper
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NY){return;} int h=i*NX*NZ;  // Copy value
  ux[h].x=u[i].x;  ux[h].y=0.0;   
}// END KERNEL TO COPY TO UX EQUATION...................................../


/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
static void main_int(float2 *p,float2 *g,float2 *ch,vfield u,float2 *ux,
                     vfield rhs,float2 *rhc,float2 *rux,float2 *th_BC,     
     		    float2 *thn_BC,float2 *ch_BC,float2 *chn_BC,float2 *th);
/*+++++++++++++++++++++++++++ MAIN FUNCTION ++++++++++++++++++++++++++++++*/
// MAIN FUNCTION TO INTEGRATE............................................../
void integrate(vfield u,float2 *ch,float Bo,float th_s){
  // Allocate variables
  float2 *th,*hp,*hg,*hc,*p,*g,*ux,*rhc,*rux,*hux;
  float2 *th_BC,*ch_BC,*thn_BC,*chn_BC;
  // Allocate data of the integration
  size_t size_p=NY*NX*NZ*sizeof(float2);  vfield rhs,ub;
  CHECK_CUDART(cudaMalloc(&rhc,size_p));
  CHECK_CUDART(cudaMalloc(&ub.x,size_p));
  CHECK_CUDART(cudaMalloc(&ub.y,size_p));
  CHECK_CUDART(cudaMalloc(&ub.z,size_p));
  CHECK_CUDART(cudaMalloc(&rhs.x,size_p));
  CHECK_CUDART(cudaMalloc(&rhs.y,size_p));
  CHECK_CUDART(cudaMalloc(&rhs.z,size_p));
  CHECK_CUDART(cudaMalloc(&g,size_p)); 
  CHECK_CUDART(cudaMalloc(&p,size_p)); 
  CHECK_CUDART(cudaMalloc(&hc,size_p));
  CHECK_CUDART(cudaMalloc(&hp,size_p)); 
  CHECK_CUDART(cudaMalloc(&hg,size_p));
  CHECK_CUDART(cudaMalloc(&th,size_p));
  // For BC of the CH equation
  size_p=2*NX*NZ*sizeof(float2); 
  CHECK_CUDART(cudaMalloc(&th_BC,size_p));
  CHECK_CUDART(cudaMalloc(&ch_BC,size_p));
  CHECK_CUDART(cudaMalloc(&thn_BC,size_p));
  CHECK_CUDART(cudaMalloc(&chn_BC,size_p));
  // For mean profile
  size_p=NY*sizeof(float2);
  CHECK_CUDART(cudaMalloc(&ux,size_p));
  CHECK_CUDART(cudaMalloc(&rux,size_p));
  CHECK_CUDART(cudaMalloc(&hux,size_p)); 
  // STEP 0  : INITIALIZATION - - - - - - - - - - - - - - - - - - - - -
  float *flag = (float *)malloc(3*sizeof(float)); 
  flag[0]=float(dt);flag[1]=float(dt);flag[2]=float(dt);char file[512];
  float t=0.0,t_mp=0.0,t_rms=0.0,t_bin=0.0,t_pln=0.0;
  // Step 0.1: Initial phi, g and old u
  v_to_phi(p,u.y);  u_to_g(g,u.x,u.z);  copyVfield(ub,u);
  // Step 0.2: Compute the initial hp(rhs.x) hg(rhs.z)
  init_nonlinear(rhs,rhc,rux,u,ch,ux,flag,th_BC,ch_BC,ub,Bo,th_s);     
  // Step 0.3: Copy to the previous to last rhs/BC
  copyBuffer(hp,rhs.x);       copyBuffer(hg,rhs.z);  
  copyBuffer(hc,rhc);         copyBufferux(hux,rux);
  copyBufferBC(thn_BC,th_BC); copyBufferBC(chn_BC,ch_BC); 
  flag[0]=0.0;  flag[1]=0.0;  flag[2]=0.0;
  // STEP 1  : LOOP ON TIME STEPS- - - - - - - - - - - - - - - - - - -
  while(t<=t_tot){
    // STEP 1.0.1: If needed prepare the output to mean profile file
    if(t_mp/dt_mp>=1){flag[0]=t;t_mp=0.0;printf("\n Time = %.3f\n",t);}
    // STEP 1.0.2: If needed prepare the output to rms profile file
    if(t_rms/dt_rms>=1){flag[1]=t;t_rms=0.0;}
    // STEP 1.0.3: If needed prepare the output to plot the plane
    if(t_pln/dt_pln>=1){flag[2]=t;t_pln=0.0;}
    // STEP 1.1: Compute the rhs save to u! (inside u also update ub)
    nonlinear(u,ch,ux,rhs,rhc,rux,p,g,hp,hg,hc,hux,flag,th_BC,ch_BC,ub,Bo,th_s);
    // STEP 1.2: Wrapper for the integration
    main_int(p,g,ch,u,ux,rhs,rhc,rux,th_BC,thn_BC,ch_BC,chn_BC,th);    
    // STEP 1.3: new time step
    t+=dt; t_mp+=dt; t_rms+=dt; t_bin+=dt; t_pln+=dt;
    flag[0]=0.0; flag[1]=0.0;  flag[2]=0.0;
    // Write to binary files
    if(t_bin/dt_bin>=1){t_bin=0.0;
      sprintf(file,"./Results/TH_%03.2f_BO_%02.2f/ch_%02.2f.bin",th_s,Bo,t);
      wBufferBinary((float*)ch, file,sizeof(float2),NY*NX*NZ);
    }// end write to binary files
  }// end loop on time steps _ - _ - _ - _ - _ - _ - _ - _ - _ - _ - _
  // OUTRO : - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 sprintf(file,"./Results/TH_%03.2f_BO_%02.2f/ch_%02.2f.bin",th_s,Bo,t);
 wBufferBinary((float*)ch, file,sizeof(float2),NY*NX*NZ);
 cudaFree(rhs.x); cudaFree(rhs.y); cudaFree(rhs.z); free(flag); 
 cudaFree(g);     cudaFree(p);     cudaFree(hp);    cudaFree(hg); 
 cudaFree(ux);    cudaFree(rux);   cudaFree(hux);   cudaFree(th); 
 cudaFree(rhc);   cudaFree(hc);    cudaFree(th_BC); cudaFree(thn_BC);
 cudaFree(ub.x);  cudaFree(ub.y);  cudaFree(ub.z);
 cudaFree(ch_BC); cudaFree(chn_BC);        return;
}// END MAIN FUNCTION TO INTEGRATE........................................./



/*+++++++++++++++++++++ WRAPPER OF INTEGRATION FUNCTION ++++++++++++++++++*/
// FUNCTION TO PERFORM THE INTEGRATION...................................../
static void main_int(float2 *p,float2 *g,float2 *ch,vfield u,float2 *ux,
                     vfield rhs,float2 *rhc,float2 *rux,float2 *th_BC,
	float2 *thn_BC,float2 *ch_BC,float2 *chn_BC,float2 *th){
 // STEP 0: INITIALIZATION
 double2 bco,bc2; bco.x=0.0; bco.y=0.0; bc2.x=0.0; bc2.y=0.0;
 // STEP 1: SOLVE FOR THE EQUATION OF G
 LU_pg(g,u.z,bco,bc2); 
 // STEP 2: SOLVE FOR THE EQUATION OF P
 LU_pg(p,u.x,bco,bc2); 
 // STEP 3: SOLVE FOR THE EQUATION OF V
 LU_v(u.y,p,bco,bc2); 
 // STEP 4: CORRECT THE BOUNDARY CONDITION. AND MODE ZZ
 corr_bcon(p,u.y);  vg_to_u(u.x,u.z,g,u.y);  main_ZZ(u,p,g);
 // STEP 5: INTEGRATE THE EQUATION OF THE MEAN PROFILE save to rux/adj flux
 // bco.x=0.0; bc2.x=0.0; LU_x(ux,rux,bco,bc2);          
 // STEP 6: WRITE THE MEAN PROFILE TO U.X
 // dim3 block,grid; block.x=block_size;
 // block.x=block_size; grid.x=(NY+block.x-1)/block.x; grid.y=1;  
 // copyuxb_k<<<grid,block>>>(u.x,ux);  
 // STEP 7: SOLVE FOR THE AUXILIARY TH VARIABLE
 LU_th(th,rhc,th_BC,thn_BC);   
 // STEP 8: SOLVE FOR THE CH VARIABLE
 LU_ch(ch, th,ch_BC,chn_BC);      
 // STEP 9: SAVE THE BCon FOR THE NEXT STEP
 copyBufferBC(thn_BC,th_BC); copyBufferBC(chn_BC,ch_BC); return;
}// END FUNCTION TO PERFORM THE INTEGRATION................................/

