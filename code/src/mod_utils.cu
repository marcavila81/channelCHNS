#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES Alberto's code that complement others
// AUTHOR: ----------- Alberto Vela Martín & Daniel Morón Montesdeoca
// DATE:   ----------- 27/03/2022
// MODIFIED:---------- 29/12/2023
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++*/
// KERNEL TO NORMALIZE......................................................
static __global__ void normalize_k(float2 *u,float norm,size_t elements){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    // Stopper
    if(h>=elements){return;}
    // Normalize
    float2 uh=u[h]; uh.x=uh.x/norm; uh.y=uh.y/norm; u[h]=uh;
}// END KERNEL TO NORMALIZE................................................

// KERNEL TO WRITE BCON TO RHS............................................./
static __global__ void write_bcon_k(double2 *rhs,double2 bcon,double2 bco2){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // Stopper
    if(h>=NX*NZ || i>=2){return;}
    // Index
    int id=i*(NY-1)*NX*NZ + h;
    // Write the Bcon
    rhs[id].x=double(1-i)*bcon.x + double(i)*bco2.x;
    rhs[id].y=double(1-i)*bcon.y + double(i)*bco2.y;
}// END KERNEL TO WRITE BCON.............................................../

// KERNEL TO WRITE BCON TO THE RHS OF CH EQUATION........................../
static __global__ void write_bconCH_k(double2 *rhs,float2 *cB,float2 *cnB){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    // Stopper
    if(h>=NX*NZ || i>=2){return;}
    // Index
    int id=i*(NY-1)*NX*NZ + h;
    // read the bcon
    float2 BC, BCn; BC=cB[h+i*NX*NZ]; BCn=cnB[h+i*NX*NZ]; 
    // write to rhs
    rhs[id].x=double(0.5*(3.0*BC.x - BCn.x));
    rhs[id].y=double(0.5*(3.0*BC.y - BCn.y));   return;
}// END KERNEL TO WRITE BCON TO THE RHS OF CH EQUATION...................../



/*++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++*/
// WRAPPER TO NORMALIZE BY A GIVEN NORM.....................................
void normalize(float2 *u,float norm,size_t elements){
    // Dimension of the block and grid
    dim3 grid,block;block.x=block_size; grid.x=(elements+block.x-1)/block.x;
    // Call the kernel
    normalize_k<<<grid,block>>>(u, norm, elements);  return;
}// END WRAPPER TO NORMALIZE A GIVEN VECTOR.................................

// WRAPPER TO INTRODUCE BCON TO A GIVEN  RHS VECTOR......................../
void write_bcon(double2 *rhs,double2 bcon,double2 bco2){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; 
    grid.x=(NX*NZ+block.x-1)/block.x; grid.y=2;
    write_bcon_k<<<grid,block>>>(rhs,bcon,bco2);     return;
}// END WRAPPER TO BCON TO A GIVEN RHS VECTOR............................../

// WRAPPER TO WRITE THE BCON OF THE CH EQUATIONS.........................../
void write_bconCH(double2 *rhs,float2 *ch_BC,float2 *chn_BC){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size;
    grid.x=(NX*NZ+block.x-1)/block.x; grid.y=2;
    write_bconCH_k<<<grid,block>>>(rhs,ch_BC,chn_BC);     return;    
}// END WRAPPER TO WRITE THE BCON OF THE CH EQUATIONS....................../

// UNIFORM VECTOR DEFINITION................................................
void uniformVec(float2 *u,float2 uni){
    // Initialize vector
    float2* vec=(float2*)malloc(NY*NX*NZ*sizeof(float2));
    // Loop
    for(int i=0;i<NY*NX*NZ;i++){vec[i].x=uni.x; vec[i].y=uni.y;}
    // Copy back to CUDA
CHECK_CUDART(cudaMemcpy(u,vec,NY*NX*NZ*sizeof(float2),cudaMemcpyHostToDevice)); 
    free(vec);  return;
}// END UNIFORM VECTOR DEFINITION...........................................

// WRAPPER TO COPY ONE VECTOR (u1->u2)......................................
void copyBuffer(float2* u2, float2* u1){
    size_t size=NY*NX*NZ*sizeof(float2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2,u1,size,cudaMemcpyDeviceToDevice)); return;
}// END WRAPPER TO COPY ONE VECTOR..........................................


// WRAPPER TO COPY ONE PHYSICAL VECTOR (u1->u2).............................
void copyBufferP(float2* u2, float2* u1){
    size_t size=NY*NXP*NZP*sizeof(float2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2,u1,size,cudaMemcpyDeviceToDevice)); return;
}// END WRAPPER TO COPY ONE PHYSICAL VECTOR.................................


// WRAPPER TO COPY ONE VECTOR (ux->ux)......................................
void copyBufferux(float2* u2, float2* u1){
    size_t size=NY*sizeof(float2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2,u1,size,cudaMemcpyDeviceToDevice)); return;
}// END WRAPPER TO COPY ONE VECTOR..........................................


// WRAPPER TO COPY ONE VECTOR (ux->ux)......................................
void copyBufferBC(float2* u2, float2* u1){
    size_t size=2*NX*NZ*sizeof(float2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2,u1,size,cudaMemcpyDeviceToDevice)); return;
}// END WRAPPER TO COPY ONE VECTOR..........................................




// WRAPPER TO COPY THE VELOCITY FIELD (u1->u2).............................
void copyVfield(vfield u2, vfield u1){
    size_t size=NY*NX*NZ*sizeof(float2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2.x,u1.x,size,cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaMemcpy(u2.y,u1.y,size,cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaMemcpy(u2.z,u1.z,size,cudaMemcpyDeviceToDevice));  
    return;
}// END WRAPPER TO COPY THE VELOCITY FIELD..................................





