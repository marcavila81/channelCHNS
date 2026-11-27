#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate the nonlinear terms in NSeq
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 11/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// GUIDE
// d1ux -> du/dy;     d1uy -> du/dz;     d1uz -> dv/dx
// d2ux -> dv/dz;     d2uy -> dw/dx;     d2uz -> dw/dy
/*++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++*/
// KERNEL TO CALC NONLINEAR.............................................../
static __global__ void calcNonlinear_k(float*uxK,float *uyK,float *uzK,
  float *d1ux,float *d1uy,float *d1uz,float *d2ux,float *d2uy,float *d2uz,
  float *d3ux,float *d3uy,float *d3uz,float *dcx,float *dcy,float *dcz,float *cK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Write velocity
  float ux=uxK[h], uy=uyK[h], uz=uzK[h], c=cK[h];
  // Write dudx
  float dxux=d3ux[h],  dxuy=d1uz[h], dxuz=d2uy[h];
  // Write dudy
  float dyux=d1ux[h],  dyuy=d3uy[h], dyuz=d2uz[h];
  // Write dudz
  float dzux=d1uy[h],  dzuy=d2ux[h], dzuz=d3uz[h];
  // Write dc
  float dcdx=dcx[h],   dcdy=dcy[h],  dcdz=dcz[h];
  // Write Hx, Hy, Hz, HT
  float omx, omy, omz, Hx, Hy, Hz, Mx, My, Mz;
  // Rho factor
  float rh_fc=1.0+0.5*(rho_r-1.0)*(c+1);
  // Buoyancy force! No gravity
  float buoy =-0.5*(rho_r-1.0)*(c+1)/(float(Fr)*float(Fr));
  // Mu factor
  float mu_fc=0.5*(mu_r-1.0)*(c+1)/float(Re); 
  // 0.Compute Omegas
  omx=dyuz-dzuy;    omy=dzux-dxuz;                     omz=dxuy-dyux;
  // 1.Compute the Hx, Hy, Hz for the matched case
  Hx=uy*omz-uz*omy; Hy=uz*omx-ux*omz;  Hz=ux*omy-uy*omx;
  // 2.Compute the Hx, Hy, Hz for the variable density case
  Hx=rh_fc*Hx;      Hy=rh_fc*Hy+buoy;  Hz=rh_fc*Hz; 
  // 3. Compute the variable viscosity terms (donot multiply by mu_fc)
  Mx=    dcdx*2*dxux   + dcdy*(dyux+dxuy) + dcdz*(dzux+dxuz); 
  My= dcdx*(dyux+dxuy) +   dcdy*2*dyuy    + dcdz*(dyuz+dzuy);
  Mz= dcdx*(dzux+dxuz) + dcdy*(dyuz+dzuy) +   dcdz*2*dzuz   ;
  // Save to dx.. structure
  d1ux[h]=Hx+mu_fc*Mx; d1uy[h]=Hy+mu_fc*My; d1uz[h]=Hz+mu_fc*Mz;   return;
}// END KERNEL TO CALC NONLINEAR........................................../

// KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH CH TERM................../
static __global__ void calcNL_NSE_CH_k(float *rhxK,float *rhyK,float *rhzK,
		float *cK,float *dvxK,float *dvyK,float *dvzK,float We){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index 
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float c=cK[h], dvx=dvxK[h], dvy=dvyK[h], dvz=dvzK[h];
  // Parameter of the forcing
  float Up=0.25*3.0*sqrtf(2.0)*(1.0/float(Cn*We)); 
  // Compute and save
  rhxK[h]+=-Up*c*dvx;   
  rhyK[h]+=-Up*c*dvy;   
  rhzK[h]+=-Up*c*dvz; return; 
}// END KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH CH TERM............./

// KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH Vc TERM................../
static __global__ void calcNL_NSE_Vc_k(float *rhxK,float *rhyK,float *rhzK,
                            float *cK,float *LxK,float *LyK,float *LzK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float c=cK[h], dvx=LxK[h], dvy=LyK[h], dvz=LzK[h];
  // Parameter of the forcing
  float mu_fc=0.5*(mu_r-1.0)*(c+1)/float(Re);
  // Compute and save
  rhxK[h]+=mu_fc*dvx;
  rhyK[h]+=mu_fc*dvy;
  rhzK[h]+=mu_fc*dvz; return;
}// END KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH Vc TERM............./

// KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH dt TERM................../
static __global__ void calcNL_NSE_dt_k(float *rhxK,float *rhyK,float *rhzK,
                            float *cK,float *dtxK,float *dtyK,float *dtzK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float c=cK[h], dvx=dtxK[h], dvy=dtyK[h], dvz=dtzK[h];
  // Parameter of the forcing
  float rh_fc=-0.5*(rho_r-1.0)*(c+1);
  // Compute and save
  rhxK[h]+=rh_fc*dvx;
  rhyK[h]+=rh_fc*dvy;
  rhzK[h]+=rh_fc*dvz; return;
}// END KERNEL TO CALC NONLINEAR TERM OF THE NSE WITH dt TERM............./

// KERNEL TO COMPUTE THE CUBE OF C......................................../
static __global__ void calc_c3_k(float *c3K,float *cK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float c=cK[h]; float c3=c*c*c; c3K[h]=c3; return; 
}// END KERNEL TO COMPUTE THE CUBE OF C.................................../

// KERNEL TO COMPUTE c3+=-(1.0+s_c)c....................................../
static __global__ void calc_vphiCH_k(float *c3K,float *cK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float c=cK[h];  c3K[h]+=-(1.0+s_c)*c;  return; 
}// END KERNEL TO COMPUTE c3+=-(1.0+s_c)c................................./

// KERNEL TO COMPUTE -u·nabla(c)........................................../
static __global__ void calc_conv_CH_k(float *uNc,float *uKx,float *uKy,  
		float *uKz,float *dcKx,float *dcKy,float *dcKz){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k;
  // Read
  float ux=uKx[h],uy=uKy[h],uz=uKz[h],dcx=dcKx[h],dcy=dcKy[h],dcz=dcKz[h];
  // Compute and save
  uNc[h]=(-ux*dcx-uy*dcy-uz*dcz);   return;
}// END KERNEL TO COMPUTE -u·nabla(c)...................................../

// KERNEL TO COMPUTE vphi................................................./
static __global__ void calc_vphi_k(float *c3K,float *LcK,float *cK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data  
  float Lc=LcK[h], c=cK[h], c3=c3K[h], vphi;
  // Compute
  vphi=c3-c-float(Cn)*float(Cn)*Lc; c3K[h]=vphi; return;
}// END KERNEL TO COMPUTE vphi............................................/

// KERNEL TO COMPUTE THE NL TERM OF THE CH EQUATION......................./
static __global__ void calc_rhc_k(float *rhc,float *aux1){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data
  float Lc=rhc[h], uNc=aux1[h], rc;  
  // COMPUTE!
  rc=uNc+float(1.0/Pe)*Lc;
  // Write to vector
  rhc[h]=rc;   return;
}// END KERNEL TO COMPUTE THE NL TERM OF THE CH EQUATION................../


// KERNEL TO COMPUTE THE BC OF THE CH EQUATION............................/
static __global__ void calc_BC_CH_k(float *BC_p,float *c_p,float th_s){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=4){return;} 
  // Auxiliary indexes
  int iy=(NY-1)*(i%2), iBC=i/2; int h=iy*(2*NXP*NZP)+k;
  // Related parameters
  float g_c=(-0.5*s_c/float(Cn*Cn));
  //g_c = g_c*(1.0+sqrtf(1.0-((4.0*Pe*Cn*Cn)/(d_im*dt*s_c*s_c))));
  // Read c prepare floats 
  float c=c_p[h],bc;
  // Angle in radians and the cosine
  double an_rad=double(th_s)*double(PI2)/double(360), csths=cos(an_rad);
  // Correct error because I donot trust myself 
  if(abs(csths)<1e-4){csths=0.0;} 
  // BCon for ch
  bc=0.5*sqrtf(2.0)*(1.0-c*c)*float(csths/double(Cn));
  // BCon for th
  if(iBC<1){ bc=bc*(g_c+(3.0*c*c-1)/float(Cn*Cn)); }  
  // Register
  BC_p[i*(2*NXP*NZP)+k]=bc;
}// END KERNEL TO COMPUTE THE BC OF THE CH EQUATION......................./



// KERNEL TO COMPUTE Cn*Cn*nabla2(c) - s_c*c............................../
static __global__ void wraprhc1_k(float *LapcK,float *cK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data
  float Lc=LapcK[h], c=cK[h]; 
  // Compute and save
  LapcK[h]=float(Cn)*float(Cn)*Lc-s_c*c;   return;
}// END KERNEL TO COMPUTE Cn*Cn*nabla2(c) - s_c*c........................./


// KERNEL TO ASSEMBLE THE FULL RHS......................................../
static __global__ void wraprhs_k(float *LaK,float *nK,float *nn_1K){
  // Indexes of thread 
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;  
  // Stopper % index   
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data
  float Lap=LaK[h], nn=nK[h], n_1=nn_1K[h]; 
  // Assemble data
  LaK[h]=float(dt/2.0)*(3.0*nn - n_1) + Lap;  return;
}// END KERNEL TO ASSEMBLE THE FULL RHS.................................../


// KERNEL TO ASSEMBLE THE FULL RHS OF CH................................../
static __global__ void wraprhc_k(float *rcK,float *cK,float *lcK,
                                                  float *ncK,float *ncK_1){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NX*NZ || i>=NY){return;}  int h=i*2*NX*NZ+k;
  // Read data
  float c=cK[h], lc=lcK[h], nc=ncK[h], nc_1=ncK_1[h], rc; 
  // Compute the rhs
  rc=float(1.0/float(dt))*c-float(1.0-d_im)*lc/float(Pe)+0.5*(3*nc-nc_1);
  // Record to vector
  rcK[h]=rc;   return;
}// END KERNEL TO ASSEMBLE THE FULL RHS OF CH............................./



// KERNEL TO COMPUTE RMS................................................../
static __global__ void wrms_k(float *dux,float *duy,float *duz,float *uxK,
                              float *uyK,float *uzK,float2 *u){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % index
  if(k>=2*NXP*NZP || i>=NY){return;}  int h=i*(2*NXP*NZP)+k; 
  // Read stuff
  float um=0*u[i].x, ux=uxK[h], uy=uyK[h], uz=uzK[h];
  // Write stuff
  dux[h]=(ux)*(ux); duy[h]=uy*uy; duz[h]=uz*uz;
}// END KERNEL TO COMPUTE THE RHS........................................./


// KERNEL TO COPY TO UX EQUATION........................................../
static __global__ void copyux_k(float2 *ru,float2 *du,float2 *ux,float2 *u){
  // Index & Stopper
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NY){return;} int h=i*NX*NZ;
  // Copy value
  ru[i].x=du[h].x; ru[i].y=0.0; u[i].x=ux[h].x; u[i].y=0.0;
}// END KERNEL TO COPY TO UX EQUATION...................................../

// KERNEL TO COMPUTE THE CG OF THE BUBBLE................................./
static __global__ void wycg_k(float* ycg,float* Vl,float *cK,double *yK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Index of interest
  int ix=k/(2*NZP), l=k%(2*NZP), h=i*(2*NXP*NZP)+k;
  // STOPPER AND INDEX
  if(l>=2*NZP || ix>=NXP || i>=NY){return;}
  // Case of l>=2*NZP-2
  if(l>=2*NZP-2){ycg[h]=0.0; Vl[h]=0.0; return;}
  // Read stuff
  float y=float(yK[i]); float c=cK[h]; float bbl=0.0;
  // Bubble?
  if(c>=0.0){bbl=1.0;}
  // Chebyshev cell width (trapezoidal weights on the wall-normal grid):
  // real volumes/centroids must weight each cell by its own dy, not LY/NY
  int im = i>0 ? i-1 : 0, ip = i<NY-1 ? i+1 : NY-1;
  float dy = 0.5f*float(yK[ip]-yK[im]);
  // CG and bubble (dy-weighted)
  Vl[h]=bbl*dy;    ycg[h]=y*bbl*dy;
}// END KERNEL TO COMPUTE THE CG OF THE BUBBLE............................/


// KERNEL TO COPY FROM THE 3D FIELD TO THE 2D PLANE......................./
static __global__ void wpln_k(float *c_pln,float *cch){
 // Indexes of thread
 int ix=blockIdx.x*blockDim.x+threadIdx.x;
 int i =blockIdx.y*blockDim.y+threadIdx.y;
 // Stopper
 if(ix>=NXP || i>=NY){return;}
 // Index of interest
 int l=NZP-1, h=i*(2*NXP*NZP)+ix*2*NZP+l, hpl=i*NXP + ix; 
 // Copy
 c_pln[hpl]=cch[h]; return;
}// END KERNEL TO COPY FROM THE 3D FIELD TO THE 2D PLANE................../


// KERNEL TO ADD THE NOISE................................................/
static __global__ void add_pert_k(float* uKx,float* uKy,
		                          float* uKz,float *rv,double *yK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Index of interest
  int ix=k/(2*NZP), l=k%(2*NZP), h=i*(2*NXP*NZP)+k;
  // STOPPER AND INDEX
  if(l>=2*NZP-2 || ix>=NXP || i>=NY){return;}
  // Initialize the auxiliary variables
  float ran, y=float(yK[i]), z, x, ux, fy, hy, kz, gz; 
  // Read the position
  x=(float(ix)/float(NXP))*LX;  z=PI2*float(l)/float(2*NZP-2);
  // Exponential function in streamwise direction
  ux=sin(0.5*PI2*x/float(LX));  ux=ux*ux;     ux=exp(-10.0*ux);
  // Cross-Section Localization functions
  fy=(1.0-y*y)*(1.0-y*y);    hy=-4.0*y*(1.0-y*y);
  kz=cos(z);                 gz=(PI2/LZ)*sin(z);
  // Additive noise ux
  ran=rv[3*h]*eps;       uKx[h]+=ran;
  // Additive noise uy
  ran=rv[3*h+1]*eps;     uKy[h]+=ran + A_P*fy*gz*ux;
  // Additive noise uz
  ran=rv[3*h+2]*eps;     uKz[h]+=ran + A_P*hy*kz*ux;
}// END KERNEL TO ADD THE NOISE........................................../


// KERNEL TO ADD THE NOISE................................................/
static __global__ void add_drop_k(float* chK,double *yK,float th_s){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Index of interest
  int ix=k/(2*NZP), l=k%(2*NZP), h=i*(2*NXP*NZP)+k;
  // STOPPER AND INDEX
  if(l>=2*NZP-2 || ix>=NXP || i>=NY){return;}
  // Angle
  double an_rd=double(th_s-90.0)*double(PI2)/double(360.0);
  double sn=sin(an_rd); float fc=1.0-float(sn); 
  // Radius
  float Rd=Rad*pow(2.0/(fc*fc*(3-fc)),1.0/3.0); float hg=-Rd*float(sn);
  // Initialize the auxiliary variables
  float y=(float(yK[i])-hg), z, x, cc;
  // Read the position
  x=float(ix)/float(NXP);  z=float(l)/float(2*NZP-2);
  // Compute the distance from the center
  x=float(LX)*(x-0.5);    z=float(LZ)*(z-0.5);   cc=x*x + y*y + z*z;
  // Radius distance
  cc=sqrtf(cc);  cc=(Rd-cc)/(sqrtf(2.0)*float(Cn)); 
  // Register to vector
  chK[h]=tanh(cc);   return;
}// END KERNEL TO ADD THE NOISE........................................../



// KERNEL OF PADDING FORWARD............................................./
static __global__ void padForward_k(float2 *pad,float2 *u){
    // Indexes
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int k = h%NZP; int j = h/(NXP*NZP);  int i = (h-j*NXP*NZP)/NZP;
    // Stopper
    if(h<NY*NXP*NZP){
        // Auxiliary functions
        float factor=1.0; float2 aux;   aux.x=0.0;    aux.y=0.0;
        if((i<NX/2 || i>NX-1) && k<NZ){
            // X indices                
            int ix=i<NX/2 ? i : i-NX/2;
            // Z indices
            int kx=k; int h =j*NX*NZ+ix*NZ+kx;
            // Write to aux
            aux=u[h]; aux.x*=factor; aux.y*=factor;
        }
    // Write to pad
    int hp=j*NXP*NZP+i*NZP+k;  pad[hp]=aux;
    }
}// END KERNEL OF FORWARD PADDING......................................../


// KERNEL OF PADDING BACKWARD............................................/
static __global__ void padBackward_k(float2 *u,float2 *pad){
    // Index of interest
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int k = h%NZ; int j = h/(NX*NZ); int i = (h-j*NX*NZ)/NZ;
    // Continue only if you comply with conditions
    if(j<NY & i<NX & k<NZ){
        // Auxiliary
        float factor=1.0;  float2 aux;
        // X indices            
        int ip=i<NX/2 ? i : NXP + (i-NX);
        // Z indices
        int kp=k;  int hp=j*NXP*NZP+ip*NZP+kp; int h =j*NX*NZ+i*NZ+k;
        // Write to aux
        aux=pad[hp]; aux.x*=factor; aux.y*=factor;
        // Auxiliary function
        if(i==NX/2 || k==NZ-1){ aux.x=0.0; aux.y=0.0;}
        // Write to velocity 
        u[h]=aux;
    }
}// END KERNEL OF BACKWARD PADDING......................................./

// KERNEL OF PADDING BACKWARD BC........................................./
static __global__ void padBackward_BC_k(float2 *u,float2 *pad){
    // Index of interest
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int k = h%NZ; int j = h/(NX*NZ); int i = (h-j*NX*NZ)/NZ;
    // Continue only if you comply with conditions
    if(j<4 & i<NX & k<NZ){
        // Auxiliary
        float factor=1.0;  float2 aux;
        // X indices
        int ip=i<NX/2 ? i : NXP + (i-NX);
        // Z indices
        int kp=k;  int hp=j*NXP*NZP+ip*NZP+kp; int h =j*NX*NZ+i*NZ+k;
        // Write to aux
        aux=pad[hp]; aux.x*=factor; aux.y*=factor;
        // Auxiliary function
        if(i==NX/2 || k==NZ-1){ aux.x=0.0; aux.y=0.0;}
        // Write to velocity
        u[h]=aux;
    }
}// END KERNEL OF BACKWARD PADDING BC..................................../




/*++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++*/
static vfield u_p, d1u_p, d2u_p, d3u_p, dc_p;
static float2 *aux1, *aux2, *aux3, *auxBC;
static float2 *c_p, *BC_p;
static float *wint;   // dedicated scratch for the M_int mass diagnostic (c+1)
static float *c_pln;
static double *y;
static void calcNonlinear(vfield u,vfield d1u,vfield d2u,vfield d3u,float2 *cK,vfield dc);
static void calcNL_NSE_CH(vfield d1u,float2 *c,vfield dvp,float We);
static void calcNL_NSE_Vc(vfield d1u,float2 *c,vfield dvp);
static void calcNL_NSE_dt(vfield d1u,float2 *c,vfield dvp);
static void calc_c3(float2 *c3_p,float2 *c);
static void calc_vphiCH(float2 *c3_p,float2 *c);
static void calc_vphi(float2 *c3,float2 *Lc,float2 *c);
static void calc_conv_CH(float2 *c3_p,vfield u,vfield dc);
static void calc_rhc(float2 *rhc,float2 *aux1);
static void calc_BC_CH(float2 *BC_p,float2 *c_p,float th_s);
static void wraprhs(float2 *Lap,float2 *nn,float2 *nn_1);
static void wraprhc1(float2 *aux2,float2 *c);
static void wraprhc(float2 *rc,float2*c,float2 *lc,float2 *nc,float2 *nc_1);
static void wraprms(vfield d2u,vfield u,float2 *ux);
static void wrapycg(float2 *ycg,float2 *bbl,float2 *c_p);
static void wrappln(float *c_pln,float2 *c_p);

/*+++++++++++++++++++++++++++++ WRAPPERS +++++++++++++++++++++++++++++++++*/
// WRAPPER OF CALC NONLINEAR.............................................../
static void calcNonlinear(vfield u,vfield d1u,vfield d2u,vfield d3u,
		float2 *cK,vfield dc){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call Kernel
    calcNonlinear_k<<<grid,block>>>
    ((float*)u.x,  (float*)u.y,  (float*)u.z  
    ,(float*)d1u.x,(float*)d1u.y,(float*)d1u.z
    ,(float*)d2u.x,(float*)d2u.y,(float*)d2u.z
    ,(float*)d3u.x,(float*)d3u.y,(float*)d3u.z
    ,(float*)dc.x, (float*)dc.y, (float*)dc.z,(float *)cK);          return;
}// END WRAPPER OF CALC NONLINEAR........................................../

// WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE CH TERM................/
static void calcNL_NSE_CH(vfield d1u,float2 *c,vfield dvp,float We){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call Kernel
    calcNL_NSE_CH_k<<<grid,block>>>
    ((float *)d1u.x,(float *)d1u.y,(float *)d1u.z,(float *)c
    ,(float *)dvp.x,(float *)dvp.y,(float *)dvp.z,We);        return;
}// END WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE CH TERM.........../

// WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE Vc TERM................/
static void calcNL_NSE_Vc(vfield d1u,float2 *c,vfield dvp){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call Kernel
    calcNL_NSE_Vc_k<<<grid,block>>>
    ((float *)d1u.x,(float *)d1u.y,(float *)d1u.z,(float *)c
    ,(float *)dvp.x,(float *)dvp.y,(float *)dvp.z);        return;
}// END WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE Vc TERM.........../

// WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE dt TERM................/
static void calcNL_NSE_dt(vfield d1u,float2 *c,vfield dvp){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call Kernel
    calcNL_NSE_dt_k<<<grid,block>>>
    ((float *)d1u.x,(float *)d1u.y,(float *)d1u.z,(float *)c
    ,(float *)dvp.x,(float *)dvp.y,(float *)dvp.z);        return;
}// END WRAPPER TO CALC NONLINEAR TERM OF THE NSE OF THE dt TERM.........../

// WRAPPER TO CALCULATE c3................................................./
static void calc_c3(float2 *c3_p,float2 *c){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call kernel
    calc_c3_k<<<grid,block>>>((float *)c3_p,(float *)c); return;
}// END WRAPPER TO CALCULATE c3............................................/

// WRAPPER TO CALCULATE c3+=-(1.0+s_c)*c.................................../
static void calc_vphiCH(float2 *c3_p,float2 *c){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call kernel
    calc_vphiCH_k<<<grid,block>>>((float *)c3_p,(float *)c); return;
}// END WRAPPER TO CACULATE c3+=-(1.0+s_c)*c.............................../

// WRAPPER TO CALCULATE -u·nabla(c)......................................../
static void calc_conv_CH(float2 *c3_p,vfield u,vfield dc){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
    // Call kernel
    calc_conv_CH_k<<<grid,block>>>((float *)c3_p,
		(float*)u.x,  (float*)u.y,  (float*)u.z,
		(float*)dc.x, (float*)dc.y, (float*)dc.z); return;
}// END WRAPPER TO CALCULATE -u·nabla(c).................................../

// WRAPPER TO CALCULATE vphi and save to first vector....................../
static void calc_vphi(float2 *c3,float2 *Lc,float2 *c){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;  grid.y=NY;    	
   // Call kernel
   calc_vphi_k<<<grid,block>>>((float *)c3,(float *)Lc,(float *)c); return;
}// END WRAPPER TO CALCULATE vphi and save to first vector................./

// WRAPPER TO CALCULATE rhc=aux1+(1/Pe)*rhc................................/
static void calc_rhc(float2 *rhc,float2 *aux1){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
   calc_rhc_k<<<grid,block>>>((float *)rhc,(float *)aux1); return;
}// END WRAPPER TO CALCULATE rhc=aux1+(1/Pe)*rhc.........................../

// WRAPPER TO CALCULATE BC................................................./
static void calc_BC_CH(float2 *BC_p,float2 *c_p,float th_s){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=4;
   // Call kernel
   calc_BC_CH_k<<<grid,block>>>((float *)BC_p,(float *)c_p,th_s); return;
}// END WRAPPER TO CALCULATE BC............................................/


// WRAPPER TO CALCULATE Cn*Cn*nabla2(c) - s_c*c............................/
static void wraprhc1(float2 *aux1,float2 *c){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
   wraprhc1_k<<<grid,block>>>((float*)aux1,(float*)c); return;
}// END WRAPPER TO CALCULATE Cn*Cn*nabla2(c) - s_c*c......................./

// WRAPPER TO ADD UP THE RHS OF C........................................../
static void wraprhc(float2 *rc,float2*c,float2 *lc,float2 *nc,float2 *nc_1){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
   wraprhc_k<<<grid,block>>>((float*)rc,(float*)c,(float*)lc,
		             (float*)nc,(float*)nc_1); return;
}// END WRAPPER TO ADD UP THE RHS OF C...................................../

// WRAPPER OF ASEMBLE OF RHS.............................................../
static void wraprhs(float2 *Lap,float2 *nn,float2 *nn_1){
   // DIMENSIONS
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NX*NZ+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
  wraprhs_k<<<grid,block>>>((float*)Lap,(float*)nn,(float*)nn_1); return;
}// END WRAPPER OF ASEMBLE OF RHS........................................../

// WRAPPER TO COMPUTE RMS................................................../
static void wraprms(vfield d2u,vfield u,float2 *ux){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
   wrms_k<<<grid,block>>>((float*)d2u.x,(float*)d2u.y,(float*)d2u.z,
            (float*)u.x,  (float*)u.y,  (float*)u.z,ux); return;
}// END WRAPPER TO COMPUTE RMS............................................./

// WRAPPER TO COMPUTE THE POSITION OF THE CG OF THE BUBBLE................./
static void wrapycg(float2 *ycg,float2 *bbl,float2 *c_p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(2*NXP*NZP+block.x-1)/block.x;  grid.y=NY;
   // Call kernel
   wycg_k<<<grid,block>>>((float*)ycg,(float*)bbl,(float*)c_p,y);return;
}// END WRAPPER TO COMPUTE THE POSITION OF THE CG OF THE BUBBLE............/

// KERNEL FOR SPURIOUS-CURRENT DIAGNOSTICS (measurement only, no math change) /
// Physical padded space, same grid/indexing as wycg_k:
//   A = |u|^2,  B = (1-c^2)|u|^2,  C = (1-c^2)
static __global__ void wdiag_k(float* A,float* B,float* C,float* D,
		float* uxK,float* uyK,float* uzK,float* cK){
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  int ix=k/(2*NZP), l=k%(2*NZP), h=i*(2*NXP*NZP)+k;
  if(l>=2*NZP || ix>=NXP || i>=NY){return;}
  if(l>=2*NZP-2){A[h]=0.0f; B[h]=0.0f; C[h]=0.0f; D[h]=0.0f; return;}
  float ux=uxK[h], uy=uyK[h], uz=uzK[h], c=cK[h];
  float u2=ux*ux+uy*uy+uz*uz;
  float w=1.0f-c*c;
  A[h]=u2; B[h]=w*u2; C[h]=w; D[h]=c+1.0f; return;
}
// Wrapper: launch kernel + cublas reductions on the EXISTING handle, then
// hand the reduced values to mod_io (wrt_diag owns only the file write)
static void wrapdiag(float* A,float* B,float* C,float* D,vfield u_p,float2* c_p){
  dim3 grid,block; block.x=block_size;
  grid.x=(2*NXP*NZP+block.x-1)/block.x; grid.y=NY;
  wdiag_k<<<grid,block>>>(A,B,C,D,(float*)u_p.x,(float*)u_p.y,(float*)u_p.z,(float*)c_p);
  cublasHandle_t h=getCublasHandle();
  int N=NY*NXP*2*NZP, imax=0; float amax=0.0f, I1=0.0f, I0=0.0f, Mplus=0.0f;
  cublasCheck(cublasIsamax(h,N,A,1,&imax),"Isamax");
  CHECK_CUDART(cudaMemcpy(&amax,A+(imax-1),sizeof(float),cudaMemcpyDeviceToHost));
  if(!(amax>=0.0f) || isnan(amax)){
    fprintf(stderr,"\n ERROR: diag umax invalid (max |u|^2 entry = %g) -- aborting\n",(double)amax);
    exit(1);
  }
  float umax=sqrtf(amax);
  cublasCheck(cublasSasum(h,N,B,1,&I1),"SasumB");
  cublasCheck(cublasSasum(h,N,C,1,&I0),"SasumC");
  cublasCheck(cublasSasum(h,N,D,1,&Mplus),"SasumD");
  if(!(I0>0.0f) || isnan(I0)){
    fprintf(stderr,"\n ERROR: diag I0 invalid (%g) -- aborting\n",(double)I0);
    exit(1);
  }
  setDiagVals(umax,sqrtf(I1/I0),I0,Mplus);
  return;
}

// WRAPPER TO COPY FROM THE WHOLE 3D FIELD TO THE PLANE VECTOR............./
static void wrappln(float *c_pln,float2 *c_p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NXP+block.x-1)/block.x; grid.y=NY;
   // Call kernel
   wpln_k<<<grid,block>>>(c_pln,(float *)c_p); return;
}// END WRAPPER TO COPY FROM THE WHOLE 3D FIELD TO THE PLANE VECTOR......../


// GUIDE
// d1u_p.x -> du/dy;     d1u_p.y -> du/dz;     d1u_p.z -> dv/dx
// d2u_p.x -> dv/dz;     d2u_p.y -> dw/dx;     d2u_p.z -> dw/dy



/*+++++++++++++++++++++++ MAIN FUNCTIONS NONLINEAR +++++++++++++++++++++++*/
// INITIALIZE NONLINEAR TERMS............................................../
void init_nonlinear(vfield du,float2 *rhc,float2 *rux,vfield u,float2 *c,
 float2 *ux,float *flag,float2 *th_BC,float2 *ch_BC,vfield ub,float Bo,float We,float th_s){
 // OLD CALCULATION OF THE NON-LINEAR TERM································
 // 1. Compute the wall-normal derivative 
 derivy(du.x,u.x);         derivy(du.y,u.y);          derivy(du.z,u.z); 
 // 2. Pad forward the velocity and c
 padForward(u_p.x,u.x);    padForward(u_p.y,u.y);    
 padForward(u_p.z,u.z);    padForward(c_p,c);
 // 3. Pad forward the wall-normal derivative of the velocity 
 padForward(d1u_p.x,du.x); padForward(d2u_p.z,du.z);
 padForward(d3u_p.y,du.y);
 // 4. Calculate the streamwise derivative
 derivx(du.x,u.x);         derivx(du.y,u.y);         derivx(du.z,u.z); 
 padForward(d1u_p.z,du.y); padForward(d2u_p.y,du.z);  
 padForward(d3u_p.x,du.x); 
 // 5. Calculate the spanwise   derivative
 derivz(du.x,u.x);         derivz(du.y,u.y);         derivz(du.z,u.z);
 padForward(d1u_p.y,du.x); padForward(d2u_p.x,du.y);  
 padForward(d3u_p.z,du.z);
 // 6. Derivative of c
 derivx(du.x,c);           derivy(du.y,c);          derivz(du.z,c);
 padForward(dc_p.x,du.x);  padForward(dc_p.y,du.y); padForward(dc_p.z,du.z);
 fftBackward(dc_p.x);      fftBackward(dc_p.y);     fftBackward(dc_p.z);
 // 7. fftBackward of u & c
 fftBackward(u_p.x);       fftBackward(u_p.y);      
 fftBackward(u_p.z);       fftBackward(c_p);   
 // 8. fftBackward of d1
 fftBackward(d1u_p.x);     fftBackward(d1u_p.y);    fftBackward(d1u_p.z);   
 // 9. fftBackward of d2
 fftBackward(d2u_p.x);     fftBackward(d2u_p.y);    fftBackward(d2u_p.z);
 // 10.fftBackward of d3
 fftBackward(d3u_p.x);     fftBackward(d3u_p.y);    fftBackward(d3u_p.z);
 // 11.Calc Nonlinear (u x w only) and save it in d1u_p (d2u_p&d3u_p r free)
 calcNonlinear(u_p,d1u_p,d2u_p,d3u_p,c_p,dc_p);
 // NEW CALCULATION OF mu(c)·nabla(u)········································
 // 12.Compute the laplacian of each velocity component
 Laplacian(du.x,u.x);     Laplacian(du.y,u.y);     Laplacian(du.z,u.z);
 padForward(d2u_p.x,du.x);padForward(d2u_p.y,du.y);padForward(d2u_p.z,du.z);
 fftBackward(d2u_p.x);    fftBackward(d2u_p.y);    fftBackward(d2u_p.z);
 // 13.Add to the rhs of the NSE + mu_fc·c·nabla(u)
 calcNL_NSE_Vc(d1u_p,c_p,d2u_p);
 // NEW CALCULATION OF rho(c)*dudt··········································
 derivt(du.x,u.x,ub.x);   derivt(du.y,u.y,ub.y);   derivt(du.z,u.z,ub.z);
 padForward(d2u_p.x,du.x);padForward(d2u_p.y,du.y);padForward(d2u_p.z,du.z);
 fftBackward(d2u_p.x);    fftBackward(d2u_p.y);    fftBackward(d2u_p.z);
 // 13.Add to the rhs of the NSE + rho_fc·c·dudt
 calcNL_NSE_dt(d1u_p,c_p,d2u_p);
 // NEW CALCULATION OF THE c·nabla(phi)··································
 // 16.Calculate c3=c*c*c, pass it to spectral space and save to aux1
 calc_c3(d3u_p.x,c_p);   copyBufferP(d2u_p.x,d3u_p.x); 
 fftForward(d2u_p.x);    padBackward(aux1,d2u_p.x);
 // 17.Calculate Lap(c), save it to aux2. Compute vphi and save to aux1
 Laplacian(aux2,c);       calc_vphi(aux1,aux2,c); 
 // 18.Compute the derivatives of vphi and save to du
 derivx(du.x,aux1);       derivy(du.y,aux1);       derivz(du.z,aux1); 
 // 19.Padd and pass it to physical space
 padForward(d2u_p.x,du.x);padForward(d2u_p.y,du.y);padForward(d2u_p.z,du.z);
 fftBackward(d2u_p.x);    fftBackward(d2u_p.y);    fftBackward(d2u_p.z);
 // 20.Add to the rhs of the NSE - Up·c·nabla(phi); capillary slot = We
 calcNL_NSE_CH(d1u_p,c_p,d2u_p,We);  
 // NONLINEAR TERM FOR THE NSE··········································
 // 21.fftForward  of rhs
 fftForward(d1u_p.x);      fftForward(d1u_p.y);     fftForward(d1u_p.z);    
 // 22.Pad Backward the rhs
 padBackward(du.x,d1u_p.x);padBackward(du.y,d1u_p.y);
 padBackward(du.z,d1u_p.z);
 // 23.Copy the 0,0 rhs.x to rux, and introduce the constant press grad
 // dim3 block,grid; block.x=block_size; 
 // block.x=block_size; grid.x=(NY+block.x-1)/block.x; grid.y=1;
 // copyux_k<<<grid,block>>>(rux,du.x,u.x,ux);  
 // NONLINEAR TERM OF THE CH EQUATION···································
 // 24.Add to c3_p += -(1+s_c)*c_p, pass it to spectral
 calc_vphiCH(d3u_p.x,c_p); fftForward(d3u_p.x); padBackward(aux1,d3u_p.x);
 // 25.Compute the Laplacian of Lap[c3-(1.0+s_c)c] save to rhc
 Laplacian(rhc,aux1); 
 // 26.Compute the convective term -u·nabla(c). Save to c3_p & aux1
 calc_conv_CH(d3u_p.x,u_p,dc_p); fftForward(d3u_p.x); 
 padBackward(aux1,d3u_p.x);
 // 27.Construct the final hc as rhc=-u·nabla(c)+(1/Pe)*rhc
 calc_rhc(rhc,aux1); 
 // NONLINEAR BCON FOR THE CH EQUATIONS.................................
 // 28.Compute the Bcon and save to auxBC
 calc_BC_CH(BC_p,c_p,th_s); fftForwardBC(BC_p); padBackwardBC(auxBC,BC_p);
 // 29.Copy to the vectors the corresponding BCon
 CHECK_CUDART(cudaMemcpy(th_BC,auxBC,        2*NX*NZ*sizeof(float2),cudaMemcpyDeviceToDevice));
 CHECK_CUDART(cudaMemcpy(ch_BC,auxBC+2*NX*NZ,2*NX*NZ*sizeof(float2),cudaMemcpyDeviceToDevice));
 // OUTPUTS·····························································
 // 30.Output to the mean profile file (io_mp.txt) if flag[0]>0  
 if(flag[2]>0.0){wrappln(c_pln,c_p); wrt_pln(c_pln,flag,th_s,Bo);}
 if(flag[1]>0.0){wrapycg(d2u_p.x,d2u_p.y,c_p); wrt_ycg(d2u_p.x,d2u_p.y,flag,th_s,Bo);
   // spurious-current diagnostics (measurement only): reuse the d2u_p pads
   // as scratch A,B,C (free after wrt_ycg); u_p and c_p are physical here
   wrapdiag((float*)d2u_p.x,(float*)d2u_p.y,(float*)d2u_p.z,wint,u_p,c_p);
   wrt_diag(flag,We);
 }
 //if(flag[0]>0.0){
  // 30.1 Call kernel to write to io_mpr.txt file
  // wrt_mp(ux,flag);  
 //}// end if
 // 31.Output to the rms profile file (io_rms.txt) if flag[1]>0
 //if(flag[1]>0.0){
  // 31.0 Compute the (u-mean(u))^2 and save to dyu_p
  //wraprms(d2u_p,u_p,ux); 
  // 31.1 Perform the Forward fft
  //fftForward(d2u_p.x);      fftForward(d2u_p.y);     fftForward(d2u_p.z);
  // 31.2 Call kernel to write to io_rms.txt file
  //wrt_rms(d2u_p,flag); 
 //}// end if
 // ADAPT TO WALL-NORMAL FORMULATION···································
 // 32.Compute the derivative of hx and hz wrt y. Save to aux1, aux2
 derivy(aux1,du.x);         derivy(aux2,du.z);
 // 33.Compute the final rhs for phi and g: dxu_p.x, dxu_p.z
 get_hp_hg(du.x,du.y,du.z,aux1,aux2);
}// END INITIALIZE NONLINEAR TERMS........................................./



// NONLINEAR FUNCTION....................................................../
void nonlinear(vfield u,float2*c,float2*ux,vfield rhs,float2*rhc,float2 *rux
   ,float2*p,float2*g,float2*hp,float2*hg,float2*hc,float2*hux,float* flag
   ,float2 *th_BC,float2 *ch_BC,vfield ub,float Bo,float We,float th_s){
 // STEP 0. Compute the rhs and u at this time step      save old u !!!
 init_nonlinear(rhs,rhc,rux,u,c,ux,flag,th_BC,ch_BC,ub,Bo,We,th_s); 
 copyVfield(ub,u);    
 // STEP 1. Compute the laplacian of phi, g, Save to u.x,u.z
 compLap(u.x,p);        compLap(u.z,g);             
 // STEP 2. Asemble the rhs of the phi, g, and save to u!
 wraprhs(u.x,rhs.x,hp); wraprhs(u.z,rhs.z,hg); 
 // STEP 4. Copy the new non-linear terms to hp, hg
 copyBuffer(hp,rhs.x);  copyBuffer(hg,rhs.z); 
 // STEP 5. Assemble rhs of ux and save it to rux, save to hux old rux
 // rhsux(rux,hux,ux);    
 // STEP 6. Compute nabla2(c) -> aux1 (only if d_im<1)
 if(d_im<0.999999999999){
 Laplacian(aux1,c); 
 // Step 7. Compute Cn*Cn*nabla2(c)-s_c*c -> aux1
 wraprhc1(aux1,c); 
 // Step 8. Compute nabla2(aux1)
 Laplacian(aux2,aux1);}
 else{float2 uu; uu.x=0.0; uu.y=0.0;  uniformVec(aux2,uu);}
 // Step 9. Assemble the rhc= c/dt - aux2*(1-d_im)/Pe+0.5*(3*rhc-hc). 
 // Save it to aux3
 wraprhc(aux3,c,aux2,rhc,hc);
 // Step 10.Copy the new non-linear term to hc and the new rhs to rhc
 copyBuffer(hc,rhc);     copyBuffer(rhc,aux3);              return;
}// END NONLINEAR FUNCTION................................................./








// SET NONLINEAR........................................................./
void setNonlinear(double *y_h){
  // Stuff
  CHECK_CUDART(cudaMalloc(&y ,NY*sizeof(double)));
  CHECK_CUDART(cudaMemcpy(y,y_h,NY*sizeof(double),cudaMemcpyHostToDevice));	
  // Setting the size of buffers
  size_t size_pad=NY*NXP*NZP*sizeof(float2);  
  // Allocate the fields c_p c3_p
  CHECK_CUDART(cudaMalloc(&c_p,size_pad));
  // Allocate u_p 
  CHECK_CUDART(cudaMalloc(&u_p.x,size_pad));
  CHECK_CUDART(cudaMalloc(&u_p.y,size_pad));
  CHECK_CUDART(cudaMalloc(&u_p.z,size_pad));
  // Allocate derivatives 1
  CHECK_CUDART(cudaMalloc(&d1u_p.x,size_pad)); // d1u_p.x -> du/dy
  CHECK_CUDART(cudaMalloc(&d1u_p.y,size_pad)); // d1u_p.y -> du/dz
  CHECK_CUDART(cudaMalloc(&d1u_p.z,size_pad)); // d1u_p.z -> dv/dx
  // Allocate derivatives 2
  CHECK_CUDART(cudaMalloc(&d2u_p.x,size_pad)); // d2u_p.x -> dv/dy
  CHECK_CUDART(cudaMalloc(&d2u_p.y,size_pad)); // d2u_p.y -> dw/dx
  CHECK_CUDART(cudaMalloc(&d2u_p.z,size_pad)); // d2u_p.z -> dw/dy
  // Allocate derivatives 3
  CHECK_CUDART(cudaMalloc(&d3u_p.x,size_pad)); // d3u_p.x -> du/dx
  CHECK_CUDART(cudaMalloc(&d3u_p.y,size_pad)); // d3u_p.y -> dv/dy
  CHECK_CUDART(cudaMalloc(&d3u_p.z,size_pad)); // d3u_p.z -> dw/dz
  // Allocate derivatives of c
  CHECK_CUDART(cudaMalloc(&dc_p.x,size_pad));  // dc_p.x  -> dc/dx
  CHECK_CUDART(cudaMalloc(&dc_p.y,size_pad));  // dc_p.y  -> dv/dy
  CHECK_CUDART(cudaMalloc(&dc_p.z,size_pad));  // dc_p.z  -> dw/dz
  // Allocate for a plane plot
  size_pad=NY*NXP*sizeof(float);
  CHECK_CUDART(cudaMalloc(&c_pln,size_pad));
  // Allocate auxiliary spectral field vectors
  size_pad=NY*NX*NZ*sizeof(float2);
  CHECK_CUDART(cudaMalloc(&aux1,size_pad));
  CHECK_CUDART(cudaMalloc(&aux2,size_pad));  
  CHECK_CUDART(cudaMalloc(&aux3,size_pad)); 
  // Allocate auxiliary vectors for the BC's
  size_pad=4*NXP*NZP*sizeof(float2);
  CHECK_CUDART(cudaMalloc(&BC_p,size_pad));
  size_pad=4*NX*NZ*sizeof(float2);
  CHECK_CUDART(cudaMalloc(&auxBC,size_pad));
  // Dedicated scratch for the M_int mass diagnostic: (c+1) per physical cell
  size_pad=NY*NXP*2*NZP*sizeof(float);
  CHECK_CUDART(cudaMalloc(&wint,size_pad));  return;
}// END SETTER.........................................................../









// DESTROY NONLINEAR...................................................../
void NonDestroy(void){   cudaFree(aux1);      cudaFree(aux2);
    cudaFree(u_p.x);     cudaFree(u_p.y);     cudaFree(u_p.z);
    cudaFree(d1u_p.x);   cudaFree(d1u_p.y);   cudaFree(d1u_p.z);
    cudaFree(d2u_p.x);   cudaFree(d2u_p.y);   cudaFree(d2u_p.z); 
    cudaFree(d3u_p.x);   cudaFree(d3u_p.y);   cudaFree(d3u_p.z);
    cudaFree(dc_p.x);    cudaFree(dc_p.y);    cudaFree(dc_p.z);
    cudaFree(c_p);       cudaFree(aux3);      cudaFree(y);
    cudaFree(BC_p);      cudaFree(auxBC);     cudaFree(c_pln);
    cudaFree(wint); return;
}// END NONLINEAR DESTROYER............................................../







// PAD FORWARD.........................................................../
void padForward(float2 *pad,float2 *u){
    // Dimension
    dim3 grid,block; block.x=block_size; 
    grid.x = (NY*NXP*NZP + block.x - 1)/block.x;
    // Call kernel
    padForward_k<<<grid,block>>>(pad,u); return;
}// END PAD FORWARD....................................................../

// PAD BACKWARD........................................................../
void padBackward(float2 *u,float2 *pad){
    // Dimension
    dim3 grid,block; block.x=block_size;
    grid.x = (NY*NXP*NZP + block.x - 1)/block.x;
    // Call kernel
    padBackward_k<<<grid,block>>>(u,pad); return;
}// END PAD BACKWARD...................................................../

// PAD BACKWARD BC......................................................./
void padBackwardBC(float2 *u,float2 *pad){
    // Dimension
    dim3 grid,block; block.x=block_size;
    grid.x = (4*NXP*NZP + block.x - 1)/block.x;
    // Call kernel
    padBackward_BC_k<<<grid,block>>>(u,pad); return;
}// END PAD BACKWARD BC................................................../








// FUNCTION TO INTRODUCE THE NOISE........................................./
void add_pert(vfield u,double *y){
  // Initialize the noise
  float *rv_h=(float *)malloc(3*NY*2*NXP*NZP*sizeof(float)); float* rv;
  // Initialize seed
  srand(time(NULL));
  // Write to vector a random number from 0 to 1 
 for(int i=0;i<3*NY*2*NXP*NZP;i++){rv_h[i]=(float)rand()/RAND_MAX*2.0-1.0;}
  // Allocate and copy in GPU 
  CHECK_CUDART(cudaMalloc(&rv,3*NY*2*NXP*NZP*sizeof(float)));
  CHECK_CUDART(cudaMemcpy(rv,rv_h,3*NY*2*NXP*NZP*sizeof(float),cudaMemcpyHostToDevice));
  // Pad forward the velocity
  padForward(u_p.x,u.x); padForward(u_p.y,u.y);  
  padForward(u_p.z,u.z); 
  // fftBackward of u
  fftBackward(u_p.x);    fftBackward(u_p.y);     
  fftBackward(u_p.z);    
  // Grid dimension
  dim3 block,grid; block.x=block_size;
  grid.x=(2*NXP*NZP+block.x-1)/block.x; grid.y=NY;
  // Call kernel
  add_pert_k<<<grid,block>>>((float*)u_p.x,(float*)u_p.y,
		             (float*)u_p.z,rv,y);
  // fftForward of the velocity and T
  fftForward(u_p.x);      fftForward(u_p.y);      
  fftForward(u_p.z);      
  // Pad Backward the velocity and T
  padBackward(u.x,u_p.x); padBackward(u.y,u_p.y); 
  padBackward(u.z,u_p.z);  
  cudaFree(rv); free(rv_h);
  return;
}// END FUNCTION TO INTRODUCE THE NOISE..................................../

// INIT CH FIELD.........................................................../
void initCh(float2 *ch, double *y_h,float th_s){
   // Read .bin files Careful they must have the same length u input
   if(restart>0){ printf("Reading initial condition of ch\n");
    rBufferBinary((float*)ch,"ch_s.bin",sizeof(float2),NY*NX*NZ);}
   // Usually u initialize with -1 in the wholde domain
   else{ printf("Preparing initial condition\n");
   // Auxiliary variables
   double *y; 
   // Copy to device
   CHECK_CUDART(cudaMalloc(&y ,NY*sizeof(double)));
   CHECK_CUDART(cudaMemcpy(y,y_h,NY*sizeof(double),cudaMemcpyHostToDevice));
   // Grid dimension
   dim3 block,grid; block.x=block_size;
   grid.x=(2*NXP*NZP+block.x-1)/block.x; grid.y=NY;
   // Call kernel
   add_drop_k<<<grid,block>>>((float*)c_p,y,th_s);
   // fftForward and pad backward
   fftForward(c_p); padBackward(ch,c_p);  cudaFree(y);
   }// end no restarting
   // Return
   return;
}// END INIT FIELD........................................................./





