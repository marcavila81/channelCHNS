#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform input/output operations
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 28/12/2023
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++*/
// KERNEL FOR MEAN PROFILE INITIALIZATION................................../
static __global__ void initux_k(float2 *ux,double *y){
  // Index & Stopper
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NY){return;} int h=i*NX*NZ;
  // Set Values
  ux[h].x=float(y[i]);     ux[h].y=0.0;
}// KERNEL FOR MEAN PROFILE INITIALIZATION................................./


// KERNEL TO WRITE TO VEC_MP.............................................../
static __global__ void wrt_mp_k(float *vec_mp,float2 *ux,float flag,
		                                               int cnt_mp){
 // Index of kernel
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  // Stopper & index 
  if(k>NY){return;} int i=cnt_mp*(NY+1)+k;
  // Time thread
  if(k<1){vec_mp[i]=flag;   return;}
  // Other threads
  else{vec_mp[i]=ux[k-1].x; return;}
}// END KERNEL TO WRITE TO VEC_MP........................................../

// KERNEL TO WRITE TO VEC_RMS............................................../
static __global__ void wrt_rms_k(float *vc,float2 *u,float t,int cnt,int h){
  // Index of kernel
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  // Stopper & index in the vector;                index in the velocity
  if(k>NY){return;} int i=cnt*3*(NY+1)+h*(NY+1)+k, j=(k-1)*NXP*NZP;
  // Time thread
  if(k<1){vc[i]=t;} else{vc[i]=u[j].x;}
}// END KERNEL TO WRITE TO VEC_RMS........................................./


/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
// Variables for all files
static double *y;           FILE *ptr_mp,*ptr_rms;
static float *vec_mp;      static int cnt_mp;
static float *vec_rms;     static int cnt_rms;


/*+++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++*/
// SETTER OF IN OUT VARIABLES............................................../
void setIO(double *y_h){
  // Allocate y
  CHECK_CUDART(cudaMalloc(&y ,NY*sizeof(double)));
  CHECK_CUDART(cudaMemcpy(y,y_h,NY*sizeof(double),cudaMemcpyHostToDevice));
  // Open file of the Mean Profile·············
  ptr_mp = fopen("./io_mp.txt","w");
  // Write the first time step
  fprintf(ptr_mp,"%.8e  ",0.0); cnt_mp=0;
  // Write the y grid in the two files
  for(int i=0;i<NY;i++){fprintf(ptr_mp,"%.8e  ",y_h[i]);}
  fprintf(ptr_mp,"\n");
  // Allocate the vectors to save the intermediate steps of Nu and Re
  size_t size=(NY+1)*dc_mp*sizeof(float);
  CHECK_CUDART(cudaMalloc(&vec_mp,size));
  // Open file of the rms······················
  ptr_rms = fopen("./io_rms.txt","w");
  // Write the first time step
  fprintf(ptr_rms,"%.8e  ",0.0); cnt_rms=0;
  // Write the y grid in the two files
  for(int i=0;i<NY;i++){fprintf(ptr_rms,"%.8e  ",y_h[i]);}
  fprintf(ptr_rms,"\n");
  // Allocate the vectors to save the intermediate steps of Nu and Re
  size=3*(NY+1)*dc_rms*sizeof(float); 
  CHECK_CUDART(cudaMalloc(&vec_rms,size));        return;
}// END SETTER OF IN OUT VARIABLES........................................./



// DESTROY IN OUT........................................................../
void IODestroy(void){cudaFree(y); cudaFree(vec_mp); fclose(ptr_mp); 
                                  cudaFree(vec_rms);fclose(ptr_rms); return;
}// END DESTROYER IN OUT.................................................../




/*+++++++++++++++++++++++++++ INITIAL FIELD ++++++++++++++++++++++++++++++*/
// INIT FIELD............................................................../
void initField(vfield u){
   // Read .bin files Careful they must have the same length u input
   if(restart>0){ printf("Reading initial condition\n");
    rBufferBinary((float*)u.x,"ux_s.bin",sizeof(float2),NY*NX*NZ);
    rBufferBinary((float*)u.y,"uy_s.bin",sizeof(float2),NY*NX*NZ);
    rBufferBinary((float*)u.z,"uz_s.bin",sizeof(float2),NY*NX*NZ);}
   // Usually u initialize with the optimum perturbation
   else{ printf("Preparing initial condition\n");
   // Auxiliary variables
    float2 uni; uni.x=0.0; uni.y=0.0;
   // Set current velocity to 0
    uniformVec(u.x,uni); uniformVec(u.y,uni); uniformVec(u.z,uni); 
   // Call function to initialize the Temperature field
   dim3 block,grid; block.x=block_size; grid.x=(NY+block.x-1)/block.x;
   // Kernel
   initux_k<<<grid,block>>>(u.x,y);
   // Add perturbation and noise
   add_pert(u,y);
   }
   // Return
   return;
}// END INIT FIELD........................................................./







/*+++++++++++++++++++++++++++++ WRITE BUFFER +++++++++++++++++++++++++++++*/
// WRITE BUFFER............................................................/
void wBufferBinary(float* w,const char*file,size_t elsize,size_t elements){
    // Allocate memory in host create sizes
    float* t_host=(float*)malloc(elsize*elements); size_t size=elements;
    // Copy w to host
 CHECK_CUDART(cudaMemcpy(t_host,w,elsize*elements, cudaMemcpyDeviceToHost));
    // Open file
    FILE *fp=fopen(file,"wb");
    if(fp==NULL){printf("\nerror escritura: %s",file); exit(1);}
    // Write to file
    size_t fsize =fwrite( (unsigned char*)t_host,elsize,size,fp);
    if(fsize!=size){ printf("\nwriting error: %s",file); exit(1);}
    // Close file and free host
    fclose(fp);    free(t_host); return;
}// END BUFFER WRITER....................................................../


/*++++++++++++++++++++++++++++++ READ BUFFER +++++++++++++++++++++++++++++*/
// READ BUFFER............................................................./
void rBufferBinary(float* w,const char*file,size_t elsize,size_t elements){
    // Allocate memory in host create sizes
    float* t_host=(float*)malloc(elsize*elements); size_t size=elements;
    // Open file
    FILE *fp=fopen(file,"rb");
    if(fp==NULL){printf("\nerror lectura: %s",file); exit(1);}
    // Read file
    size_t fsize =fread( (unsigned char*)t_host,elsize,size,fp);
    if(fsize!=size){ printf("\nreading error: %s",file); exit(1);}
    // Close file
    fclose(fp);
    // Copy to device
  CHECK_CUDART(cudaMemcpy(w,t_host,elsize*elements,cudaMemcpyHostToDevice));
    // Free host and return
    free(t_host); return;
}// END BUFFER READER....................................................../




/*++++++++++++++++++++++++++++++ TO CHECK STUFF ++++++++++++++++++++++++++*/
// WRITE BUFFER H........................................................../
void wBufferBinary_h(float* w,const char* file,int sz){
    FILE *fp=fopen(file,"w");
    // Write to file
    for(int i=0;i<sz;i++){fprintf(fp,"%.8e\n",w[i]);}
    // Close file and free host
    fclose(fp);     return;
}// END BUFFER WRITER H..................................................../


// WRITE MEAN PROFILE TO THE FILE........................................../
void wrt_mp(float2 *ux,float *flag){
  // Read time
  float t=flag[0];
  // Save the current mp and Re1 to the corresponding vectors
  dim3 grid,block;block.x=block_size;grid.x=(NY+block.x)/block.x;grid.y=1;
  wrt_mp_k<<<grid,block>>>(vec_mp,ux,t,cnt_mp);              cnt_mp+=1;
  // If cnt_mp==dc_mp write
  if(cnt_mp==dc_mp){
  // Copy to host
  size_t size=(NY+1)*dc_mp*sizeof(float);float *v_m_h=(float *)malloc(size);
  CHECK_CUDART(cudaMemcpy(v_m_h,vec_mp,size,cudaMemcpyDeviceToHost));
  // Write
  for(int i=0;i<dc_mp;i++){
    for(int j=0;j<NY+1;j++){
      fprintf(ptr_mp,"%.8e  ",v_m_h[i*(NY+1)+j]);
    }// end loop in j
    fprintf(ptr_mp,"\n");  
  }// end loop in i
  // Reset counter
  cnt_mp=0;       free(v_m_h);   
  }return;
}// END WRITE MEAN PROFILE TO THE FILE...................................../


// WRITE RMS TO THE FILE.................................................../
void wrt_rms(vfield d2u,float *flag){
  // Read time
  float t=flag[1];
  // Prepare the kernel
  dim3 grid,block;block.x=block_size;grid.x=(NY+block.x)/block.x;grid.y=1;
  // Write urms
  wrt_rms_k<<<grid,block>>>(vec_rms,(float2 *)d2u.x,t,cnt_rms,0);
  // Write vrms
  wrt_rms_k<<<grid,block>>>(vec_rms,(float2 *)d2u.y,t,cnt_rms,1);
  // Write wrms
  wrt_rms_k<<<grid,block>>>(vec_rms,(float2 *)d2u.z,t,cnt_rms,2);
  // Increase the counter by 1
  cnt_rms+=1;
  // If cnt_rms==dc_rms write to file
  if(cnt_rms==dc_rms){
  // Copy to host
  size_t size=3*(NY+1)*dc_rms*sizeof(float);
  float *v_m_h=(float *)malloc(size);
  CHECK_CUDART(cudaMemcpy(v_m_h,vec_rms,size,cudaMemcpyDeviceToHost));
  // Write
  for(int i=0;i<dc_rms;i++){
   for(int h=0;h<3;h++){
    for(int j=0;j<NY+1;j++){
      fprintf(ptr_rms,"%.8e  ",v_m_h[i*3*(NY+1)+h*(NY+1)+j]);
    }// end loop in j
    fprintf(ptr_rms,"\n");
   }// end loop in h
  }// end loop in i
  // Reset counter
  cnt_rms=0;       free(v_m_h);
  }return;
}// END WRITE RMS TO THE FILE............................................../





