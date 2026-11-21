# channelCHNS
Two-phase flow code for a Cartesian channel geometry 
Formulation: wall-normal vorticity-velocity formulation
Discretisation: Fourier-Fourier + compact finite differences

Current developer & maintainer: Prof. Dr. Marc Avila with help of Vibe Code (Mistral Large 4)

Developer of the original solver: Dr.-Ing. Daniel Morón Montesdeoca while at ZARM, University of Bremen

## Building and running
Build with `make` in `code/` (requires CUDA: nvcc, cufft, cublas, curand, cusparse).

The code requires an input file at run time (no built-in defaults):

    ./wnCH_NS [input_file]        # default input file: ./case.in

Input format: one `key value` per line, `#` starts a comment. Required keys:

    th_s    # case contact angle (degrees)
    Bo      # physical Bond number of the case
    t_tot   # total simulation time
    dt_mp   # console progress-print interval (mean-profile file output is currently disabled)
    dt_rms  # interval for the time series: ycg.txt (center of gravity + volume)
            #   and diag_v_int.txt (spurious-current diagnostics, capillary units)
    dt_pln  # interval for pln_*.bin plane cuts (x-y plane through the drop center, physical space)
    dt_bin  # interval for ch_*.bin full-field dumps (final ch/ux/uy/uz.bin are always written)

If the file is missing or a required key is absent, the code prints an error
and exits. Optional key: `outdir <dir>` sets the output directory (default
`output`); all results go directly into it (no TH/Bo subfolder):
`run_info.txt` (run header), `ycg.txt`, `diag_v_int.txt`, `pln_*.bin`,
`ch_*.bin`, and the final `ch.bin ux.bin uy.bin uz.bin` (restart fields).
Bo = physical Bond number; if `rho_r = 1` in `head.h`, Bo is forced to 0 since
there is no buoyancy.

Numerical/physics setup (grid, `dt`, `Re`, `Cn`, `rho_r`, `mu_r`, ...) is set
via the `#define`s in `code/src/head.h` and needs a recompile. Case parameters
and output rates come from the input file, so sweeps are done externally, e.g.:

    for th in 20 40; do for Bo in 0.5 1.0; do
      printf 'th_s %s\nBo %s\nt_tot 5.0\ndt_mp 0.05\ndt_rms 0.05\ndt_pln 0.25\ndt_bin 50.0\noutdir run_th%s_Bo%s\n' $th $Bo $th $Bo > case.in
      ./wnCH_NS case.in
    done; done

Some `head.h` setup constants can be overridden at build time without editing
the source, e.g. a cheap coarse run:

    make -C code FLAGS="-DNY=65 -DNX=64 -DNZ=33 -DCn=0.05 -Ddt=1e-4"

## Restarting
The final fields `ch.bin ux.bin uy.bin uz.bin` in the output directory are the
restart state (spectral representation). To restart, copy them into the run
directory as `ch_s.bin ux_s.bin uy_s.bin uz_s.bin` and rebuild with
`#define restart 1` in `code/src/head.h` (`restart 0` = fresh start).

## Validation and sanity checks
See `validation/README.md` for details. Quick start (requires a GPU build):

    cd validation
    # full validation case (reference results in validation/reference/):
    python run_case.py case1_static_cap_th50_Bo0.json --workdir /path/to/run
    # fast smoke test for solver changes (~seconds on one GPU):
    python run_case.py sanity_th50_Bo0_coarse.json --workdir /path/to/sanity \
        --build-flags "-DNY=65 -DNX=64 -DNZ=33 -DCn=0.05 -Ddt=1e-4"

`run_case.py` writes `case.in` from the case JSON, runs the binary, checks that
all expected outputs exist, and runs `analysis.py`, which prints each metric
with PASS/FAIL against the case oracle and the delta vs the reference metrics,
and writes `metrics.json` + plots into the output directory. Exit code 0 means
the oracle passed.
