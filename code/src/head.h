/* INTEGRATE NAVIER-STOKES EQUATIONS IN CARTESIAN COORDINATES USING A PSEUDO-SPECTRAL METHOD */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

/* code version, injected by the Makefile (-DGIT_COMMIT=...) */
#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

/*++++++++++++++++++++++++++++++ CONSTANTS ++++++++++++++++++++++++++++*/
#define PI2 6.283185307179586 // two times pi

/*++++++++++++++++++++++++ SIMULATION PARAMETERS ++++++++++++++++++++++*/
// Grid points and modes
// overridable from the Makefile via -D (e.g. sanity runs: make FLAGS="-DNY=65 ...")
#ifndef NY
#define NY   257            // Wall-normal direction no. points
#endif
#ifndef NX
#define NX   256            // 2*m_x streamwise modes (must be even!)
#endif
#ifndef NZ
#define NZ   129            // m_z+1 spanwise   modes (must be odd! )
#endif
// Length of the domain
#define LX   (5.0)         // Length in x direction (in initial radius)
#define LY   (5.0)         // Length in y direction (in initial radius)
#define LZ   (5.0)         // Length in z direction (in initial radius)
// Initial conditions
#ifndef Rad
#define Rad  1.0            // radius of the drop w.r.t. half height
#endif
#define A_P  0.0            // magnitude of the streamwise perturbation
#define eps  0.0            // magnitude of the random perturbation
/*++++++++++++++++++++++++++++ FLOW PARAMETERS ++++++++++++++++++++++++*/
#ifndef Cn
#define Cn   0.025           // Cahn number
#endif
#ifndef Re
#define Re   50.0            // Reynolds number
#endif
#define Fr   1.0             // Froude number
#define WE_REF 1.0           // reference Weber number (capillary slot 1/(Cn*We)) used if rho_r==1
/* CASE INPUTS ARE READ AT RUNTIME from an input file (mandatory, no defaults):
   usage: ./wnCH_NS [input_file]      (default input file: ./case.in)
   Format: one 'key value' per line, '#' starts a comment. Required keys:
       th_s    case contact angle (degrees)
       Bo      physical Bond number of the case
       t_tot   total simulation time
       dt_mp   console progress-print interval (mp file output disabled)
       dt_rms  interval for the time series ycg.txt (cg+volume) and
               diag_v_int.txt (spurious currents)
       dt_pln  interval for pln_*.bin plane cuts (x-y plane at drop center)
       dt_bin  interval for ch_*.bin full-field dumps (final ch/ux/uy/uz.bin
               are always written)
       (these four are runtime inputs, declared extern below)
   Optional key:
       outdir  output directory (default "output"); results go directly into
               this directory, no TH/Bo subfolder
   If the file cannot be opened or a required key is missing, the code prints
   an error and exits. Compile-time setup (grid, dt, Re, Cn, rho_r, ...) stays
   in the #defines of this file.
   Bo (physical) vs We (internal Weber/capillary slot), up to O(1) consts.
   Lengths are in units of the initial drop volume-radius (Rad=1 default):
       Bo_input = (1-rho_r)*Rad^2*We/Fr^2  is the physical Bond number based
       on the drop radius; the Bond number in code units is Bo/Rad^2.
       =>  We = Bo*Fr^2/((1-rho_r)*Rad^2)
   Convention: bubble = c=+1 phase, continuous liquid = reference (density 1,
   base viscosity), so rho_r = rho_bubble/rho_liquid <= 1 and mu_r <= 1.
   If rho_r == 1 (matched densities) there is no buoyancy: Bo is forced to 0
   at runtime (with a printf) and We = WE_REF. Output file names keep BO but
   print the physical Bo. The old Fritz-based critical estimate
       Boc = 0.564*(th_s*pi/360)^2/(1-rho_r) - 0.1
   is kept for reference only: it must NOT be computed anywhere in the code
   (it is singular for rho_r = 1). */
extern float t_tot, dt_mp, dt_rms, dt_pln, dt_bin;  // set from input file in main
extern char OUT[300];   // output directory (trailing slash), default "output/"
// Ratios
#ifndef rho_r
#define rho_r 1.0           // density   ratio
#endif
#ifndef mu_r
#define mu_r  1.0           // viscosity ratio			     
#endif

/*++++++++++++++++++++++++++++ GRID PARAMETERS ++++++++++++++++++++++++*/
#define p_y  0.0            // p_y==1-> uniform grid, p_y=0->Chebyshev
			     
/*+++++++++++++++++++++++++++ OUTPUT PARAMETERS +++++++++++++++++++++++*/
// Write intervals dt_mp, dt_rms, dt_pln, dt_bin are RUNTIME inputs from the
// input file (see extern declaration above), not defines.
#define dc_mp 1             // Rows to store before saving (mean profile)
#define dc_rms 1            // Rows to store before saving (RMS)

/*+++++++++++++++++++++++++++ IN&OUT PARAMETERS +++++++++++++++++++++++*/
// Restart 
#define restart 0           // >0 will try to read *_s.bin files

/*++++++++++++++++++++++++ INTEGRATOR PARAMETERS ++++++++++++++++++++++*/
#define d_im  1.0           // implicit parameter
// Time-Step size
#ifndef dt
#define dt    0.00005        // time step size (constant)
#endif
// t_tot (total simulation time) is a RUNTIME input from the input file
// Points after aliasing
#define NXP (3*NX/2)        // Total physical streamwise points
#define NZP (3*(NZ-1)/2+1)  // half of physical spanwise points
			    
/*+++++++++++++++++++++++++ DERIVED PARAMETERS ++++++++++++++++++++++++*/
#define Pe (3.0/(1.0*Cn))   // Peclet number following Magaletti
#define s_c (2.0*Cn*sqrt(Pe/(d_im*dt)))

/*++++++++++++++++++++++ ARCHITECTURE PARAMETERS ++++++++++++++++++++++*/
// Prepare block dimensions
#define block_size  256      // Number of threads per block
#define grid_size 65535      // Number of blocks in direction y and z 

// Check Cudart in case u want to troubleshoot errors
#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, " CUDART: %s = %d (%s) at (%s:%d)\n",  #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

/*+++++++++++++++    STRUCTURES INHERITED FROM ALBERTO   ++++++++++++++*/
typedef struct {float2* x;float2* y;float2* z;} vfield;
typedef struct {int Ny; int Nz; int Nx; float Lz ; float Lx;} size_p;

/*+++++++++++++++++++++++++++    FUNCTIONS    +++++++++++++++++++++++++*/
// BOUNDARY CONDITION
void setBCon(void);
void corr_bcon(float2 *p,float2 *v);
void BConDestroy(void);

// COMPACT FINITE DIFFERENCES
void init_cfd(double *y_h,double *LH_1,double *RH_1,double *DC1,
                       double *LH_2,double *RH_2,double *FD_h);

// CUBLAS 
void setCublas(double *y_h);
void cublasCheck(cublasStatus_t error,const char* function);
cublasHandle_t getCublasHandle(void);
void CublasDestroy(void);
void adj_flux(float2 *rhx);

// DERIVATIVES
void setDeriv(double *D_h);
void derivx(float2 *grad,float2 *p);
void derivz(float2 *grad,float2 *p);
void derivyw(float2 *grad,float2 *p);
void v_to_phi(float2 *phi,float2 *v);
void u_to_g(float2 *g,float2 *ux,float2 *uz);
void vg_to_u(float2 *ux,float2 *uz,float2 *g,float2 *v);
void get_hp_hg(float2 *dux,float2 *duy,float2 *duz,
	       float2 *dHxdy,float2 *dHzdy);
void compLap(float2 *Lap,float2 *u);
void Laplacian(float2 *Lap,float2 *c);
void derivt(float2 *grad,float2 *u,float2 *ub);
void DerivDestroy(void);

// FAST FOURIER TRANSFORM
void setFft(size_p sizes);
void fftDestroy(void);
void fftForward(float2* buffer);
void fftForwardBC(float2* buffer);
void fftBackward(float2* buffer);

// LINEAR 
void setLinear(double*L_h,double*R_h,double*L2_h,double*R2_h,double*FD_h);
void LU_v(float2 *u, float2 *rhs,double2 bcon,double2 bco2);
void LU_x(float2 *rux,float2 *ux,double2 bcon,double2 bco2);
void LU_pg(float2 *u,float2 *rhs,double2 bcon,double2 bco2);
void LU_ch(float2*ch,float2  *th,float2 *ch_BC,float2 *chn_BC);
void LU_th(float2*th,float2 *rhs,float2 *th_BC,float2 *thn_BC);
void derivy(float2 *grad,float2 *p);
void derivyy(float2 *grad,float2 *p);
void rhsux(float2 *rux,float2 *hux,float2 *ux);
void LinearDestroy(void);

// IN/OUT                         
void setIO(double *y_h);   
void IODestroy(void);
void wBufferBinary(float*w,const char* file,size_t elsize,size_t elements);
void rBufferBinary(float*w,const char* file,size_t elsize,size_t elements);
void wBufferBinary_h(float* w,const char* file,int sz);
void initField(vfield u);
void wrt_mp(float2 *ux,float *flag);
void wrt_rms(vfield d2u,float *flag);
void wrt_pln(float *c_pln,float *flag,float th_s,float Bo);
void wrt_diag(float *flag,float We);          // spurious-current diagnostics (file write only)
void setDiagDir(const char* nme);             // case results folder, set once from main
void setDiagVals(float umax,float urms,float I0); // reduced values for the next wrt_diag row

// INTEGRATE
void integrate(vfield u,float2 *ch,float Bo,float We,float th_s);

// MODE ZZ
void main_ZZ(vfield u,float2 *p,float2 *g);
void wrt_ycg(float2 *ycg,float2 *bbl,float *flag,float th_s,float Bo);

// NONLINEAR
void setNonlinear(double *y_h);
void NonDestroy(void);
void padForward(float2 *pad,float2 *u);
void padBackward(float2 *u,float2 *pad);
void padBackwardBC(float2 *u,float2 *pad);
void add_pert(vfield u,double *y);
void add_drop(float2* ch,double *y);
void init_nonlinear(vfield rhs,float2 *rhc,float2 *rux,vfield u,float2 *c,
     float2 *ux,float* flag,float2 *th_BC,float2 *ch_BC,vfield ub,float Bo,float We,float th_s);
void nonlinear(vfield u,float2*c,float2*ux,vfield rhs,float2*rhc,
    float2 *rux,float2*p,float2*g,float2*hp,float2*hg,float2*hc,
     float2*hux,float* flag,float2 *th_BC,float2 *ch_BC,vfield ub,float Bo,float We,float th_s);
void initCh(float2 *ch,double *y_h,float th_s);

// UTILS
void normalize(float2* u,float norm,size_t elements);
void uniformVec(float2 *u,float2 uni);
void copyBuffer(float2 *u2, float2 *u1);
void copyBufferP(float2* u2, float2* u1);
void copyBufferux(float2 *u2, float2 *u1);
void copyBufferBC(float2* u2, float2* u1);
void copyVfield(vfield u2, vfield u1);
void write_bcon(double2 *rhs,double2 bcon,double2 bco2);
void write_bconCH(double2 *rhs,float2 *ch_BC,float2 *chn_BC);
