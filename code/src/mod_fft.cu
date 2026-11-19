#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform ffts
// AUTHOR: ----------- Alberto Vela Martin & Daniel Morón Montesdeoca
// DATE:   ----------- 27/03/2022
// MODIFIED: --------- 29/12/2023
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++*/
static cufftHandle fft2_d2z_zp;
static cufftHandle fft2_d2z_BC;
static cufftHandle fft2_z2d_zp;
// Checker of fft in cuda...................................................
void cufftCheck(cufftResult error, const char* function ){
    if(error != CUFFT_SUCCESS){
       printf("\n error  %s : %d \n", function, error); exit(1);
    }
    return;
}// END checker of fft in cuda..............................................

/*+++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++*/
// INITIALIZE THE FFT.......................................................
void setFft(size_p sizes){
    // Size of the points after aliasing
    int nzp[2]={NXP,2*NZP-2};
    // Planner of the fft 
cufftCheck(cufftPlanMany(&fft2_d2z_zp,2,nzp,NULL,1,0,NULL,1,0,CUFFT_R2C,NY),"ALLOCATE_FFT3_R2C_ZP");
cufftCheck(cufftPlanMany(&fft2_d2z_BC,2,nzp,NULL,1,0,NULL,1,0,CUFFT_R2C,4),"ALLOCATE_FFT3_R2C_ZP");
cufftCheck(cufftPlanMany(&fft2_z2d_zp,2,nzp,NULL,1,0,NULL,1,0,CUFFT_C2R,NY),"ALLOCATE_FFT3_C2R_ZP");
    // Allocate the auxiliary vector
    return;
}//END SETTER...............................................................

// DESTROY THE FFT..........................................................
void fftDestroy(void){
    cufftDestroy(fft2_d2z_zp);  cufftDestroy(fft2_z2d_zp); 
    cufftDestroy(fft2_d2z_BC);  return;
}//END DESTROYER............................................................

// FORWARD FFT!.............................................................
void fftForward(float2* buffer){
    // Execute the transform
    cufftCheck(cufftExecR2C(fft2_d2z_zp,(float*)buffer,(float2*)buffer),"forward transform_zp");
    // Normalize 
    normalize(buffer,(float)NXP*(2*NZP-2),NY*NXP*NZP);    return;
}// END FORWARD FFT.........................................................

// FORWARD FFT BC...........................................................
void fftForwardBC(float2* buffer){
    // Execute the transform
    cufftCheck(cufftExecR2C(fft2_d2z_BC,(float*)buffer,(float2*)buffer),"forward transform_zp");
    // Normalize
    normalize(buffer,(float)NXP*(2*NZP-2),4*NXP*NZP);    return;
}// END FORWARD FFT BC......................................................

// BACKWARD FFT!............................................................
void fftBackward(float2* buffer){
    // Execute the transform
    cufftCheck(cufftExecC2R(fft2_z2d_zp,(float2*)buffer,(float*)buffer),"backward transform_zp");
    return;
}// END BACKWARD FFT.......................................................
