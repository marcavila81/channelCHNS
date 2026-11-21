/* MAIN FILE OF NSRAYBEN IN CUDA ---------------------------------------- */
/* Purpose: It integrates the Navier-Stokes equations in 
            cartesian coordinates using a pseudo-spectral formulation and a 
            fractional step method.
            - This version doesnot include a dynamic time step
-  Authors: Daniel Morón Montesdeoca. 
-  Contact: daniel.moron@zarm.uni-bremen.de
-  Date   : 17/07/2024                                                    */
#include"head.h"
#include <string.h>

char OUT[300]={"output/"};   // output directory (trailing slash); see extern in head.h

/* Write a commented run-info header (used for the ycg file and run_info.txt) */
static void write_run_header(FILE *f,float th_s,float Bo,float We){
  time_t now=time(NULL); char host[64]={0}; gethostname(host,sizeof(host)-1);
  fprintf(f,"# channelCHNS run info\n");
  fprintf(f,"# date (UTC)   : %s",asctime(gmtime(&now)));
  fprintf(f,"# host         : %s\n",host);
  fprintf(f,"# code version : %s\n",GIT_COMMIT);
  fprintf(f,"# grid         : NY=%d NX=%d NZ=%d (NXP=%d NZP=%d), domain LX=%g LY=%g LZ=%g\n",
          NY,NX,NZ,NXP,NZP,(double)LX,(double)LY,(double)LZ);
  fprintf(f,"# integrator   : dt=%g t_tot=%g d_im=%g\n",(double)dt,(double)t_tot,(double)d_im);
  fprintf(f,"# parameters   : Cn=%g Re=%g Fr=%g rho_r=%g mu_r=%g WE_REF=%g Rad=%g\n",
          (double)Cn,(double)Re,(double)Fr,(double)rho_r,(double)mu_r,(double)WE_REF,(double)Rad);
  fprintf(f,"# case         : th_s=%g Bo=%g (physical) We=%g (Weber slot)\n",
          (double)th_s,(double)Bo,(double)We);
  fprintf(f,"# output rates : dt_mp=%g dt_rms=%g dt_pln=%g dt_bin=%g\n",
          (double)dt_mp,(double)dt_rms,(double)dt_pln,(double)dt_bin);
  fprintf(f,"# columns of the ycg file:\n");
  fprintf(f,"#   t      : time\n");
  fprintf(f,"#   <y>_cg : center-of-gravity height of the drop (c>=0 region)\n");
  fprintf(f,"#   V_drop : drop volume, indicator c>=0 (code units)\n");
}
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int main(int argc, const char* argv[]){
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 printf("\n++++++++++++++  Welcome to nsWallChan in CUDA  +++++++++++++++");
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 // CASE INPUTS FROM FILE (mandatory, no defaults): ./wnCH_NS [input_file]
 const char *fname = (argc>1) ? argv[1] : "case.in";
 float th_s=0.0, Bo=0.0, v_ttot=0.0, v_mp=0.0, v_rms=0.0, v_pln=0.0, v_bin=0.0;
 int seen[7]={0,0,0,0,0,0,0};
 FILE *fp=fopen(fname,"r");
 if(!fp){
   fprintf(stderr,"\n ERROR: cannot open input file '%s' -- this code requires an input file (usage: ./wnCH_NS [input_file])\n",fname);
   exit(1);
 }
 char line[256],key[64],outdir_buf[256]={"output"}; double val;
 while(fgets(line,sizeof(line),fp)){
   if(line[0]=='#' || line[0]=='\n' || line[0]=='\r') continue;
   if(sscanf(line,"%63s",key)!=1){
     fprintf(stderr,"\n ERROR: malformed line in input file '%s': %s",fname,line); exit(1);
   }
   if(!strcmp(key,"outdir")){   // optional string key
     if(sscanf(line,"%*63s %255s",outdir_buf)!=1){
       fprintf(stderr,"\n ERROR: malformed outdir line in '%s': %s",fname,line); exit(1);
     }
     continue;
   }
   if(sscanf(line,"%*63s %lf",&val)!=1){
     fprintf(stderr,"\n ERROR: malformed line in input file '%s': %s",fname,line); exit(1);
   }
   if     (!strcmp(key,"th_s"))  {th_s=val;  seen[0]=1;}
   else if(!strcmp(key,"Bo"))    {Bo=val;    seen[1]=1;}
   else if(!strcmp(key,"t_tot")) {v_ttot=val;seen[2]=1;}
   else if(!strcmp(key,"dt_mp")) {v_mp=val;  seen[3]=1;}
   else if(!strcmp(key,"dt_rms")){v_rms=val; seen[4]=1;}
   else if(!strcmp(key,"dt_pln")){v_pln=val; seen[5]=1;}
   else if(!strcmp(key,"dt_bin")){v_bin=val; seen[6]=1;}
   else fprintf(stderr,"\n WARNING: unknown key '%s' in '%s' ignored\n",key,fname);
 }
 fclose(fp);
 const char *keys[7]={"th_s","Bo","t_tot","dt_mp","dt_rms","dt_pln","dt_bin"};
 for(int i=0;i<7;i++) if(!seen[i]){
   fprintf(stderr,"\n ERROR: input file '%s' is missing required key '%s'\n",fname,keys[i]); exit(1);
 }
 // Runtime run/output parameters (extern in head.h, used in mod_integrate.cu)
 t_tot=v_ttot; dt_mp=v_mp; dt_rms=v_rms; dt_pln=v_pln; dt_bin=v_bin;
 // Matched densities -> no buoyancy -> the physical Bond number must be 0
 if(rho_r==1.0){
   printf("\n NOTE: matched densities (rho_r = 1): no buoyancy, setting Bo = 0\n");
   Bo=0.0;
 }
 // Internal Weber number (capillary slot); lengths in units of the drop
 // radius Rad:  Bo = (rho_r-1)*Rad^2*We/Fr^2
 float We = (rho_r==1.0) ? float(WE_REF)
              : Bo*float(Fr)*float(Fr)/(float(rho_r-1.0)*float(Rad)*float(Rad));
 printf("\n Case: th_s=%03.2f  Bo=%02.2f (physical)  We=%.6f (Weber slot)\n",th_s,Bo,We);
 printf(" Run: t_tot=%f  dt_mp=%f dt_rms=%f dt_pln=%f dt_bin=%f\n",t_tot,dt_mp,dt_rms,dt_pln,dt_bin);
 // Startup assertion: all case values must be valid (finite, positive intervals)
 if(!isfinite(th_s) || !isfinite(Bo) || !isfinite(We) || !(t_tot>0.0f) ||
    !(dt_mp>0.0f) || !(dt_rms>0.0f) || !(dt_pln>0.0f) || !(dt_bin>0.0f)){
   fprintf(stderr,"\n ERROR: invalid case values from '%s': th_s=%f Bo=%f We=%f t_tot=%f dt_mp=%f dt_rms=%f dt_pln=%f dt_bin=%f\n",
           fname,(double)th_s,(double)Bo,(double)We,(double)t_tot,(double)dt_mp,(double)dt_rms,(double)dt_pln,(double)dt_bin);
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
 // Output directory: optional 'outdir' key from the input file (default "output")
 snprintf(OUT,sizeof(OUT),"%s%s",outdir_buf,(outdir_buf[strlen(outdir_buf)-1]=='/')?"":"/");
 { struct stat sto; if(stat(OUT,&sto)==-1){mkdir(OUT,0700);} }
 snprintf(Nme,sizeof(Nme),"%s",OUT);
 // Run-info header: prepended to the ycg file, and stored as run_info.txt
 // next to the binary outputs (raw .bin files stay headerless)
 { char ycgfile[300], rif[300];
   sprintf(ycgfile,"%sycg.txt",Nme);
   sprintf(rif,"%srun_info.txt",Nme);
   FILE *f1=fopen(ycgfile,"w"); if(f1){write_run_header(f1,th_s,Bo,We); fclose(f1);}
   FILE *f2=fopen(rif,"w");     if(f2){write_run_header(f2,th_s,Bo,We); fclose(f2);}
   // diag_v_int.txt: one header line, then one row per output event (append in wrt_diag)
   char dfile[300];
   sprintf(dfile,"%sdiag_v_int.txt",Nme);
   FILE *f3=fopen(dfile,"w");
   if(f3){ fprintf(f3,"# t  umax_cap  u_int_rms_cap  Ifrac\n"); fclose(f3);}
   setDiagDir(Nme);
 }
 // Initialize the velocity field and the scalar field
 initField(u);  initCh(ch,y_h,th_s); 
 // INTEGRATION+++++++++++++++++++++++++++++++++++++++++++++++
 integrate(u,ch,Bo,We,th_s);   
 // OUTRO ++++++++++++++++++++++++++++++++++++++++++++++++++++
 printf("\n FINISHED TH=%03.2f AND BO=%02.2f\n",th_s,Bo);
 // Last velocity field 
 sprintf(Nme,"%sch.bin",OUT);
 wBufferBinary((float*)ch,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"%sux.bin",OUT);
 wBufferBinary((float*)u.x,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"%suy.bin",OUT);
 wBufferBinary((float*)u.y,Nme,sizeof(float2),NY*NX*NZ);
 sprintf(Nme,"%suz.bin",OUT); 
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
