# channelCHNS
Current developer & maintainer: Prof. Dr. Marc Avila with help of Vibe Code (Mistral Large 4)
Developer of the original solver: Dr.-Ing. Daniel Morón Montesdeoca while at ZARM, University of Bremen
Two-phase flow code for a Cartesian channel geometry 
Formulation: wall-normal vorticit-velocity
Discretisation: Fourier-Fourier + compact finite differences

## Building and running
Build with `make` in `code/` (requires CUDA: nvcc, cufft, cublas, curand, cusparse).

The code requires an input file at run time (no built-in defaults):

    ./wnCH_NS [input_file]        # default input file: ./case.in

Input format: one `key value` per line, `#` starts a comment. Required keys:

    th_s    # case contact angle (degrees)
    Bo      # physical Bond number of the case
    t_tot   # total simulation time
    dt_mp   # write interval: mean profile
    dt_rms  # write interval: RMS profiles
    dt_pln  # write interval: plane cuts
    dt_bin  # write interval: binary dumps

If the file is missing or a key is absent, the code prints an error and exits.
Each run writes to `./Results/TH_<th_s>_BO_<Bo>/` (Bo = physical Bond number;
if `rho_r = 1` in `head.h`, Bo is forced to 0 since there is no buoyancy).

Numerical/physics setup (grid, `dt`, `Re`, `Cn`, `rho_r`, `mu_r`, ...) is set
via the `#define`s in `code/src/head.h` and needs a recompile. Case parameters
and output rates come from the input file, so sweeps are done externally, e.g.:

    for th in 20 40; do for Bo in 0.5 1.0; do
      printf 'th_s %s\nBo %s\nt_tot 5.0\ndt_mp 0.05\ndt_rms 0.05\ndt_pln 0.25\ndt_bin 50.0\n' $th $Bo > case.in
      ./wnCH_NS case.in
    done; done
